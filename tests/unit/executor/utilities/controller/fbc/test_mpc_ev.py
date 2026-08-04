"""L2 — MPC `Ev` component physics.

Two defects are pinned here:

* the state-of-charge trajectory ignored the battery capacity and could go negative;
* the `min_soc` charging scheme demanded the target at the wrong timesteps, so the car left
  below its minimum state of charge.

All energies are in Wh and all powers in W, matching the executor's internal units.
"""
import numpy as np
import pytest
from linopy import Model

import hamlet.constants as c
from hamlet.executor.utilities.controller.fbc.mpc.linopy.components import Ev

NAME = 'ev1'
CAPACITY = 50_000  # Wh
CHARGING_POWER = 11_000  # W


def make_ev(timesteps, delta, *, soc_init, energy_consumed, availability=None,
            scheme=None, efficiency=1.0, v2g=0):
    """Build an `Ev` directly, bypassing the executor."""
    n = len(timesteps)
    availability = [1] * n if availability is None else availability
    scheme = scheme or {'method': 'full'}
    forecasts = {
        f'{NAME}_availability': availability,
        f'{NAME}_energy_consumed': energy_consumed,
    }
    return Ev(
        NAME,
        forecasts=forecasts,
        timesteps=timesteps,
        delta=delta,
        socs={NAME: [soc_init]},
        charging_scheme=scheme,
        sizing={
            'capacity': CAPACITY,
            'charging_home': CHARGING_POWER,
            'charging_AC': CHARGING_POWER,
            'charging_DC': CHARGING_POWER,
            'charging_efficiency': efficiency,
            'v2g': v2g,
        },
    )


def build_model(ev, timesteps):
    """Define the EV's variables and constraints on a fresh linopy model."""
    model = Model(force_dim_names=True)
    model = ev.define_variables(model, comp_type=c.P_EV)
    model = ev.define_constraints(model)
    return model


class TestStateOfCharge:
    """The SoC trajectory the MPC starts from, before any charging decision."""

    def test_initial_soc_is_capped_at_capacity(self, timesteps, delta):
        """Regression: the capacity cap was computed and then immediately overwritten.

        A stale `socs` entry above the battery's capacity was carried into the horizon
        verbatim, letting the optimiser discharge energy the battery never held.
        """
        ev = make_ev(timesteps, delta,
                     soc_init=CAPACITY + 10_000,
                     energy_consumed=[0, 0, 0, 0])

        assert np.max(ev.soc) <= CAPACITY

    def test_soc_never_goes_negative(self, timesteps, delta):
        """Regression: cumulative driving consumption was subtracted without a floor.

        Once the trip consumption exceeded the starting charge the SoC went negative, which is
        not a physical state and silently relaxes the storage constraint.
        """
        ev = make_ev(timesteps, delta,
                     soc_init=CAPACITY,
                     energy_consumed=[10_000, 10_000, 10_000, 40_000])

        assert np.min(ev.soc) >= 0

    def test_soc_follows_cumulative_consumption(self, timesteps, delta):
        """The trajectory is the starting charge minus cumulative consumption, floored at 0."""
        ev = make_ev(timesteps, delta,
                     soc_init=CAPACITY,
                     energy_consumed=[10_000, 10_000, 10_000, 40_000])

        assert list(ev.soc) == [40_000, 30_000, 20_000, 0]


class TestMinSocScheme:
    """The `min_soc` charging scheme must actually reach the requested minimum."""

    @pytest.mark.solver
    def test_min_soc_is_reached_before_departure(self, timesteps, delta):
        """Regression: the remaining-charging-energy array was reversed instead of shifted.

        `target_soc[t]` is "the SoC needed at t so that the car can still reach its target by
        the time it leaves", i.e. the target minus the energy chargeable *strictly after* t.
        Reversing the cumulative array put the full block energy at t=0 and one timestep's
        worth at the end, so the constraint at the last available timestep asked for far less
        than the target and the car left under-charged.
        """
        target_fraction = 0.8
        ev = make_ev(timesteps, delta,
                     soc_init=20_000,
                     energy_consumed=[0, 0, 0, 0],
                     scheme={'method': 'min_soc', 'min_soc': {'val': target_fraction}})
        model = build_model(ev, timesteps)

        target_soc = model.constraints[f'{NAME}_soc_scheme'].rhs.values

        # By the last timestep no further charging is possible, so the constraint must already
        # demand the full target.
        assert target_soc[-1] == pytest.approx(CAPACITY * target_fraction)

    @pytest.mark.solver
    def test_min_soc_is_reached_when_the_car_leaves_mid_horizon(self, timesteps, delta):
        """Same defect, exercised across an availability gap.

        The car is plugged in for the first two timesteps and gone for the last two, so the
        target of 40 kWh must be met by timestep 1.

        At timestep 0 the soc need only be 20 kWh: the reachable ceiling there is 31 kWh
        (20 kWh start + one hour at 11 kW) and a further 11 kWh can still be added at
        timestep 1. The constraint therefore reads [20, 40, -, -] kWh. The reversed array gave
        [9, 29, -, -], so the car left 11 kWh short of its minimum.
        """
        target_fraction = 0.8
        ev = make_ev(timesteps, delta,
                     soc_init=20_000,
                     energy_consumed=[0, 0, 0, 0],
                     availability=[1, 1, 0, 0],
                     scheme={'method': 'min_soc', 'min_soc': {'val': target_fraction}})
        model = build_model(ev, timesteps)

        target_soc = model.constraints[f'{NAME}_soc_scheme'].rhs.values

        last_available = 1
        assert target_soc[last_available] == pytest.approx(CAPACITY * target_fraction)
        # One timestep earlier the soc may still be one timestep of charging below the ceiling.
        reachable_at_0 = 20_000 + CHARGING_POWER
        assert target_soc[0] == pytest.approx(reachable_at_0 - CHARGING_POWER)


class TestDrivingEntersTheSocRecursion:
    """The soc variable and the charging-scheme targets must be on the same scale.

    The soc recursion used to model charging only, while the targets were computed from a
    trajectory that did account for driving. The two therefore drifted apart the moment the car
    went anywhere, and the `min_soc` constraint stopped binding for the rest of the horizon.
    """

    @pytest.mark.solver
    def test_the_car_recharges_exactly_what_it_drove(self, timesteps, delta):
        """Regression: after a trip the minimum-SoC constraint was satisfied by arithmetic.

        The car starts full-enough at 40 kWh, spends 15 kWh driving while away, and comes back
        to a charger with a minimum SoC of 80 % (40 kWh). It therefore has to put back exactly
        the 15 kWh it used.

        With driving missing from the recursion the soc variable never dropped, so charging
        nothing already satisfied the constraint and the car ended the horizon 15 kWh short.
        """
        ev = make_ev(timesteps, delta,
                     soc_init=40_000,
                     energy_consumed=[0, 15_000, 0, 0],
                     availability=[1, 0, 1, 1],
                     scheme={'method': 'min_soc', 'min_soc': {'val': 0.8}})
        model = build_model(ev, timesteps)

        # Charging is otherwise free, so price it to make the optimiser charge as little as the
        # constraints allow
        charge = model.variables[f'{NAME}_{c.P_EV}_{c.ET_ELECTRICITY}_{c.PF_OUT}']
        model.add_objective(-1 * charge.sum(), overwrite=True)
        model.solve(solver_name='highs', output_flag=False, log_to_console=False)

        assert model.status == 'ok'
        soc = model.variables[f'{NAME}_{c.P_EV}_soc'].solution.values

        # The trip shows up in the trajectory the model solves for, not just in the reference
        assert soc[1] == pytest.approx(soc[0] - 15_000, abs=1)
        # ... the minimum is reached before the car leaves again ...
        assert soc[-1] == pytest.approx(CAPACITY * 0.8, abs=1)
        # ... and it charged exactly the trip's worth, no more
        assert -charge.solution.values.sum() == pytest.approx(15_000, abs=1)

    @pytest.mark.solver
    def test_the_soc_trajectory_matches_the_reference_when_nothing_charges(self, timesteps, delta):
        """With the charger unavailable the soc variable must reproduce the reference exactly.

        This is what "same scale" means: the trajectory the targets are built from and the one
        the model solves for coincide when no charging is possible.
        """
        ev = make_ev(timesteps, delta,
                     soc_init=40_000,
                     energy_consumed=[5_000, 5_000, 5_000, 5_000],
                     availability=[0, 0, 0, 0],
                     scheme={'method': 'min_soc', 'min_soc': {'val': 0.8}})
        model = build_model(ev, timesteps)
        model.add_objective(0 * model.variables[f'{NAME}_{c.P_EV}_soc'].sum(), overwrite=True)
        model.solve(solver_name='highs', output_flag=False, log_to_console=False)

        assert model.status == 'ok'
        soc = model.variables[f'{NAME}_{c.P_EV}_soc'].solution.values

        assert soc == pytest.approx(list(ev.soc), abs=1)

    def test_driving_is_clamped_to_what_the_battery_holds(self, timesteps, delta):
        """A forecast that would drive the soc below zero must not make the problem infeasible.

        Per-timestep consumption is taken as the decrease of the floored reference trajectory,
        so it can never exceed what the car actually has.
        """
        ev = make_ev(timesteps, delta,
                     soc_init=10_000,
                     energy_consumed=[6_000, 6_000, 6_000, 6_000],
                     availability=[0, 0, 0, 0],
                     scheme={'method': 'full'})

        assert ev.consumption.min() >= 0
        assert ev.consumption.sum() == pytest.approx(10_000)  # never more than it started with
        assert list(ev.soc) == [4_000, 0, 0, 0]
