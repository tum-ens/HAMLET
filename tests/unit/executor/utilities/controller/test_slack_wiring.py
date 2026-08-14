"""Unit — the balance slack machinery in the controllers themselves.

These pin two properties that are easy to break silently and that no solve would reveal:

* the slack penalty must actually reach the objective, in every backend;
* when a slack is used, the run must say so — the shed energy is never written to the setpoints,
  so a silent slack makes an infeasible agent indistinguishable from a healthy one.
"""
import inspect
import logging
from types import SimpleNamespace

import pytest

import hamlet.constants as c
from hamlet.executor.utilities.controller.fbc.mpc.linopy import mpc_linopy
from hamlet.executor.utilities.controller.fbc.mpc.poi import mpc_poi
from hamlet.executor.utilities.controller.rtc.optim.linopy import optim_linopy
from hamlet.executor.utilities.controller.rtc.optim.poi import optim_poi

OBJECTIVES = [
    (mpc_linopy.Linopy, 'mpc/linopy'),
    (mpc_poi.POI, 'mpc/poi'),
    (optim_linopy.Linopy, 'rtc/linopy'),
    (optim_poi.POI, 'rtc/poi'),
]


@pytest.mark.parametrize('controller, name', OBJECTIVES, ids=[n for _, n in OBJECTIVES])
def test_the_slack_test_precedes_the_market_name_test(controller, name):
    """Market names are user-defined, so the prefix test must not run first.

    A market called `electricity` would otherwise capture `electricity_gen_slack` in the
    market branch, the penalty would never enter the objective, and shedding would be free --
    the optimiser would then dump load at no cost and report an optimal solution.
    """
    source = inspect.getsource(controller.define_objective)

    if 'market_names' not in source:
        pytest.skip(f'{name} does not classify by market name')

    assert source.index("endswith('_slack')") < source.index('market_names'), (
        f'{name}: the market-name test runs before the slack test, so a market named after an '
        f'energy type would silence the slack penalty')


@pytest.mark.parametrize('controller, name', OBJECTIVES, ids=[n for _, n in OBJECTIVES])
def test_the_objective_applies_the_configured_penalty(controller, name):
    """Not a hardcoded constant: the penalty is configurable per agent."""
    source = inspect.getsource(controller.define_objective)

    assert 'self.slack_penalty' in source, f'{name} does not use the configured slack penalty'


class TestSlackIsReported:
    """A slack that nobody hears about is worse than a crash."""

    @pytest.mark.parametrize('module', [mpc_linopy, optim_linopy], ids=['mpc', 'rtc'])
    def test_it_reports_through_logging_not_warnings(self, module):
        """Regression: the slack report must not go through `warnings.warn`.

        It originally could not: `hamlet/executor/setup.py` installed a blanket
        `warnings.filterwarnings("ignore")` at import, so a warning raised here reached nobody
        and the shed energy was completely silent. That filter is gone (#199), so this is no
        longer the *only* thing standing between the report and the user -- but `warnings`
        deduplicates by source line, so a single `warn` here would report the first agent that
        shed and stay quiet for every one after it. `logging` is the right channel for a
        per-agent, per-timestep report, and this pins it.
        """
        source = inspect.getsource(module.Linopy._warn_on_slack)

        assert 'LOGGER' in source
        assert 'warnings.warn' not in source

    @pytest.mark.parametrize('module', [mpc_linopy, optim_linopy], ids=['mpc', 'rtc'])
    def test_the_report_actually_comes_out_of_the_controller(self, module, caplog):
        """The end-to-end property: a used slack produces a record naming the agent and the peak.

        This used to log to `logging.getLogger(mpc_linopy.__name__)` itself and assert `caplog`
        saw it, which tested the standard library — it called no HAMLET code, and a review panel
        confirmed it stayed green with the blanket `filterwarnings("ignore")` reinstated, i.e.
        under the exact regression its own docstring named. (It could never have caught that in
        any case: warning filters do not affect `logging`.)

        `_warn_on_slack` is now driven on a stub model instead, so the assertion is that *the
        controller* reports, at the threshold it claims to use.
        """
        import numpy as np

        peak = module.Linopy.SLACK_REPORTING_THRESHOLD + 1000.0
        probe = object.__new__(module.Linopy)
        probe.agent = SimpleNamespace(agent_id='agent_under_test')
        probe.model = SimpleNamespace(variables={
            'balance_electricity_slack': SimpleNamespace(
                solution=SimpleNamespace(values=np.array([0.0, -peak, 0.0]))),
            # A large ordinary variable, to pin that the report is selected by the `_slack`
            # suffix rather than by magnitude. Its name must not end in `_slack` -- the first
            # draft called it `not_a_slack`, which does, and the test duly reported it.
            'power_electricity': SimpleNamespace(
                solution=SimpleNamespace(values=np.array([1e9]))),
        })

        with caplog.at_level(logging.WARNING, logger=module.__name__):
            probe._warn_on_slack()

        messages = [record.getMessage() for record in caplog.records]
        assert len(messages) == 1, (
            f'expected exactly one report -- the slack variable, not the ordinary one -- '
            f'got {messages}')
        assert 'agent_under_test' in messages[0] and 'balance_electricity_slack' in messages[0], (
            f'the report does not name the agent and the variable: {messages[0]}')

    @pytest.mark.parametrize('module', [mpc_linopy, optim_linopy], ids=['mpc', 'rtc'])
    def test_the_solve_path_reports_before_returning(self, module):
        """The check has to run on every solve, not only on a failure branch."""
        source = inspect.getsource(module.Linopy.run)

        assert '_warn_on_slack' in source
