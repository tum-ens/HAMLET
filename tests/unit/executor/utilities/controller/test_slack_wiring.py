"""Unit — the balance slack machinery in the controllers themselves.

These pin two properties that are easy to break silently and that no solve would reveal:

* the slack penalty must actually reach the objective, in every backend;
* when a slack is used, the run must say so — the shed energy is never written to the setpoints,
  so a silent slack makes an infeasible agent indistinguishable from a healthy one.
"""
import inspect
import logging

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
        """Regression: `warnings.warn` is dead in this package.

        `hamlet/executor/setup.py` installs a blanket `warnings.filterwarnings("ignore")` at
        import time, so a warning raised here never reaches the user. The shed energy would be
        completely silent, which is exactly the failure the slack is meant to make survivable
        rather than invisible.
        """
        source = inspect.getsource(module.Linopy._warn_on_slack)

        assert 'LOGGER' in source
        assert 'warnings.warn' not in source

    def test_a_warning_actually_escapes_the_package_filters(self, caplog):
        """The end-to-end property: importing hamlet must not silence this message."""
        import hamlet  # noqa: F401  -- installs the blanket warnings filter

        logger = logging.getLogger(mpc_linopy.__name__)
        with caplog.at_level(logging.WARNING, logger=mpc_linopy.__name__):
            logger.warning('energy balance closed with %.1f W of slack', 3000.0)

        assert any('slack' in record.message for record in caplog.records)

    @pytest.mark.parametrize('module', [mpc_linopy, optim_linopy], ids=['mpc', 'rtc'])
    def test_the_solve_path_reports_before_returning(self, module):
        """The check has to run on every solve, not only on a failure branch."""
        source = inspect.getsource(module.Linopy.run)

        assert '_warn_on_slack' in source
