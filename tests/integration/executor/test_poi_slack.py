"""Integration — the PyOptInterface backends build and penalise balance slacks.

Both backends are selectable per agent, so a scenario that switches `framework` must get the
same model. The POI backends previously had no slack at all: an infeasible agent aborted the
run under `poi` and was absorbed under `linopy`, with no warning either way.

PyOptInterface needs a solver shared library, which `highspy` does not provide -- it bundles HiGHS
inside its extension rather than exposing a .dll, so HAMLET ships `highsbox` for this. These tests
skip when no backend can actually solve, which is not the same as "no library loaded": Gurobi's
library loads without a licence and only fails at `optimize()`. See `available_backend`.

They also skip on Windows, where the library crashes the interpreter under pytest. See
`skip_on_windows` for what was measured and what was ruled out.
"""
import numpy as np
import pytest

import hamlet.constants as c
from tests.poi_support import available_backend, skip_on_windows

# Safe to import the helpers above first: they touch `pyoptinterface` only inside their bodies.
poi = pytest.importorskip('pyoptinterface')


@pytest.fixture(scope='session')
def backend():
    """The solver these tests run against, resolved on first use rather than at import.

    Deliberately a fixture and not a module-level constant, so that importing this file never
    loads a solver library. Collection then stays clean even on a platform where loading one is
    hazardous, and a machine with no solver at all skips rather than errors.
    """
    module = available_backend()
    if module is None:
        pytest.skip('no PyOptInterface solver library is loadable')
    return module


@skip_on_windows
@pytest.mark.solver
def test_slack_closes_an_otherwise_infeasible_balance(backend):
    """A demand the bounded market cannot cover must be shed, not raised as infeasible.

    Mirrors what the POI controllers now build: a balance equation with a `gen`/`load` slack
    pair, each penalised at the value of lost load.
    """
    model = backend.Model()

    # Market able to supply at most 2 kW against a 5 kW load
    market = model.add_variable(name='market', lb=0, ub=2000)
    gen_slack = model.add_variable(name=f'{c.ET_ELECTRICITY}_{c.OM_GENERATION}_slack',
                                   lb=0, ub=np.inf)
    load_slack = model.add_variable(name=f'{c.ET_ELECTRICITY}_{c.OM_LOAD}_slack',
                                    lb=0, ub=np.inf)

    model.add_linear_constraint(market + gen_slack - load_slack,
                                poi.ConstraintSense.Equal, 5000, name='balance_electricity')
    model.set_objective(1.0 * market
                        + c.FBC_DEFAULT_SLACK_PENALTY * gen_slack
                        + c.FBC_DEFAULT_SLACK_PENALTY * load_slack,
                        poi.ObjectiveSense.Minimize)
    model.optimize()

    # The market is used to its bound first, and only the shortfall is shed
    assert model.get_value(market) == pytest.approx(2000)
    assert model.get_value(gen_slack) == pytest.approx(3000)
    assert model.get_value(load_slack) == pytest.approx(0)


@skip_on_windows
@pytest.mark.solver
def test_slack_stays_at_zero_when_the_market_can_cover_the_load(backend):
    """Adding a penalised slack must not change a problem that was already feasible."""
    model = backend.Model()

    market = model.add_variable(name='market', lb=0, ub=10_000)
    gen_slack = model.add_variable(name='gen_slack', lb=0, ub=np.inf)
    load_slack = model.add_variable(name='load_slack', lb=0, ub=np.inf)

    model.add_linear_constraint(market + gen_slack - load_slack,
                                poi.ConstraintSense.Equal, 5000, name='balance')
    model.set_objective(1.0 * market
                        + c.FBC_DEFAULT_SLACK_PENALTY * gen_slack
                        + c.FBC_DEFAULT_SLACK_PENALTY * load_slack,
                        poi.ObjectiveSense.Minimize)
    model.optimize()

    assert model.get_value(market) == pytest.approx(5000)
    assert model.get_value(gen_slack) == pytest.approx(0)
    assert model.get_value(load_slack) == pytest.approx(0)


def test_both_poi_controllers_build_slack_variables():
    """Static check that the wiring exists, so it holds even without a solver library."""
    import inspect

    from hamlet.executor.utilities.controller.fbc.mpc.poi import mpc_poi
    from hamlet.executor.utilities.controller.rtc.optim.poi import optim_poi

    for module in (mpc_poi, optim_poi):
        source = inspect.getsource(module)
        assert 'self.slack_enabled' in source, f'{module.__name__} never checks slack_enabled'
        assert '_slack' in source, f'{module.__name__} builds no slack variables'
        assert 'self.slack_penalty' in source, f'{module.__name__} never penalises the slack'


def test_the_poi_objective_tests_for_slack_before_market_names():
    """A market named `electricity` must not swallow `electricity_gen_slack`.

    Market names are user-defined, so the prefix test has to come second or the penalty never
    reaches the objective and shedding becomes free.
    """
    import inspect

    from hamlet.executor.utilities.controller.fbc.mpc.poi import mpc_poi

    source = inspect.getsource(mpc_poi.POI.define_objective)

    assert source.index("endswith('_slack')") < source.index('startswith(tuple(self.market_names))')
