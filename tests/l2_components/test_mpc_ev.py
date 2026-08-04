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
        target must be met by timestep 1. With 11 kW over one hour only 11 kWh can still be
        added after timestep 0, so the constraint reads [20, 40, -, -] kWh; the reversed array
        gave [9, 29, -, -] and the car left 11 kWh short.
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
        # One timestep earlier the shortfall may be at most one timestep of charging.
        assert target_soc[0] == pytest.approx(CAPACITY * target_fraction - CHARGING_POWER)
