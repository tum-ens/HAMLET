"""Integration — a whole agent MPC model, built from components and solved.

Unit tests pin one component at a time. This builds the model the way the executor does
(several components sharing one balance equation) and solves it, so it catches the failures
that only appear when the pieces are combined: an unclosed energy balance, a slack quietly
absorbing energy that should have been traded, or a sign error that only shows up in dispatch.
"""
import polars as pl
import pytest
from linopy import Model

import hamlet.constants as c
from hamlet.executor.utilities.controller.fbc.mpc.linopy.components import (
    InflexibleLoad, Market, Pv,
)

MARKET = 'continuous'
LOAD = 'load1'
PV = 'pv1'

RETAIL_PRICE = 3200
FEED_IN_PRICE = 800
GRID_FEE = 400
LEVY = 1800
MARKET_ENERGY_LIMIT = 1_000_000  # Wh per timestep, as in the shipped retailer data


@pytest.fixture
def market_forecasts(timesteps):
    n = len(timesteps)
    return {
        f'{c.TC_ENERGY}_{c.TC_ENERGY}_{c.PF_IN}': [MARKET_ENERGY_LIMIT] * n,
        f'{c.TC_ENERGY}_{c.TC_ENERGY}_{c.PF_OUT}': [MARKET_ENERGY_LIMIT] * n,
        f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_OUT}': [RETAIL_PRICE] * n,
        f'{c.TC_ENERGY}_{c.TC_PRICE}_{c.PF_IN}': [FEED_IN_PRICE] * n,
        f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_OUT}': [GRID_FEE] * n,
        f'{c.TT_GRID}_{c.TT_MARKET}_{c.PF_IN}': [0] * n,
        f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_OUT}': [LEVY] * n,
        f'{c.TT_LEVIES}_{c.TC_PRICE}_{c.PF_IN}': [0] * n,
    }


def build(timesteps, delta, market_forecasts, load_w, pv_w, slack_penalty):
    """Assemble load + PV + market on one electricity balance, with slacks, and solve."""
    model = Model(force_dim_names=True)

    # The components consume polars Series, as they come from the forecaster
    load = InflexibleLoad(LOAD, forecasts={f'{LOAD}_{c.ET_ELECTRICITY}': pl.Series(load_w)},
                          timesteps=timesteps, delta=delta)
    pv = Pv(PV, forecasts={f'{PV}_{c.ET_ELECTRICITY}': pl.Series(pv_w)},
            timesteps=timesteps, delta=delta, sizing={'controllable': True})
    market = Market(MARKET, forecasts=market_forecasts, timesteps=timesteps, delta=delta)

    for comp, comp_type in ((load, c.P_INFLEXIBLE_LOAD), (pv, c.P_PV)):
        model = comp.define_variables(model, comp_type=comp_type)
        model = comp.define_constraints(model)
    model = market.define_variables(model, comp_type=c.ET_ELECTRICITY)
    model = market.define_constraints(model)

    balance = (model.variables[f'{LOAD}_{c.P_INFLEXIBLE_LOAD}_{c.ET_ELECTRICITY}']
               + model.variables[f'{PV}_{c.P_PV}_{c.ET_ELECTRICITY}']
               + model.variables[f'{MARKET}_{c.ET_ELECTRICITY}_{c.PF_IN}']
               + model.variables[f'{MARKET}_{c.ET_ELECTRICITY}_{c.PF_OUT}'])

    gen_slack = model.add_variables(name=f'{c.ET_ELECTRICITY}_{c.OM_GENERATION}_slack',
                                    lower=0, coords=[timesteps])
    load_slack = model.add_variables(name=f'{c.ET_ELECTRICITY}_{c.OM_LOAD}_slack',
                                     lower=0, coords=[timesteps])
    model.add_constraints(balance + gen_slack - load_slack == 0, name='balance_electricity')

    objective = (model.variables[f'{MARKET}_costs'].sum()
                 - model.variables[f'{MARKET}_revenue'].sum()
                 + slack_penalty * gen_slack.sum()
                 + slack_penalty * load_slack.sum())
    model.add_objective(objective, overwrite=True)
    model.solve(solver_name='highs', output_flag=False, log_to_console=False)

    return model


@pytest.mark.solver
def test_import_covers_a_deficit_without_slack(timesteps, delta, market_forecasts):
    """With no generation the agent must import, not shed load."""
    model = build(timesteps, delta, market_forecasts,
                  load_w=[2000] * len(timesteps), pv_w=[0] * len(timesteps),
                  slack_penalty=c.FBC_DEFAULT_SLACK_PENALTY)

    assert model.status == 'ok'
    assert model.variables[f'{c.ET_ELECTRICITY}_{c.OM_LOAD}_slack'].solution.values.max() \
        == pytest.approx(0, abs=1e-6)
    assert model.variables[f'{c.ET_ELECTRICITY}_{c.OM_GENERATION}_slack'].solution.values.max() \
        == pytest.approx(0, abs=1e-6)
    assert model.variables[f'{MARKET}_{c.ET_ELECTRICITY}_{c.PF_IN}'].solution.values.min() > 0


@pytest.mark.solver
def test_the_balance_closes(timesteps, delta, market_forecasts):
    """Every timestep's flows must sum to zero — the defining invariant of the model."""
    model = build(timesteps, delta, market_forecasts,
                  load_w=[2000, 3000, 1000, 2500], pv_w=[0, 500, 4000, 100],
                  slack_penalty=c.FBC_DEFAULT_SLACK_PENALTY)

    assert model.status == 'ok'
    total = (model.variables[f'{LOAD}_{c.P_INFLEXIBLE_LOAD}_{c.ET_ELECTRICITY}'].solution.values
             + model.variables[f'{PV}_{c.P_PV}_{c.ET_ELECTRICITY}'].solution.values
             + model.variables[f'{MARKET}_{c.ET_ELECTRICITY}_{c.PF_IN}'].solution.values
             + model.variables[f'{MARKET}_{c.ET_ELECTRICITY}_{c.PF_OUT}'].solution.values
             + model.variables[f'{c.ET_ELECTRICITY}_{c.OM_GENERATION}_slack'].solution.values
             - model.variables[f'{c.ET_ELECTRICITY}_{c.OM_LOAD}_slack'].solution.values)

    assert total == pytest.approx([0] * len(timesteps), abs=1e-6)


@pytest.mark.solver
def test_surplus_is_exported_rather_than_dumped_into_slack(timesteps, delta, market_forecasts):
    """A PV surplus must leave through the market, not through the slack variable.

    This is the check that catches a slack quietly absorbing energy the agent should have sold:
    with export available and priced above zero, dumping is both physically wrong and, at a
    value-of-lost-load penalty, wildly more expensive.
    """
    model = build(timesteps, delta, market_forecasts,
                  load_w=[500] * len(timesteps), pv_w=[5000] * len(timesteps),
                  slack_penalty=c.FBC_DEFAULT_SLACK_PENALTY)

    assert model.status == 'ok'
    assert model.variables[f'{c.ET_ELECTRICITY}_{c.OM_LOAD}_slack'].solution.values.max() \
        == pytest.approx(0, abs=1e-6)
    assert model.variables[f'{MARKET}_{c.ET_ELECTRICITY}_{c.PF_OUT}'].solution.values.min() < 0
