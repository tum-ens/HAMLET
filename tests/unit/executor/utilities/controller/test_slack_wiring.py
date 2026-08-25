"""Unit — the balance slack machinery in the controllers themselves.

These pin two properties that are easy to break silently and that no solve would reveal:

* the slack penalty must actually reach the objective, in every backend;
* when a slack is used, the run must say so — the shed energy is never written to the setpoints,
  so a silent slack makes an infeasible agent indistinguishable from a healthy one.
"""
import inspect
import logging
import re
from types import SimpleNamespace

import numpy as np
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


def _linopy_probe(module, values):
    """A stub Linopy controller: `_warn_on_slack` reads `model.variables[name].solution.values`."""
    probe = object.__new__(module.Linopy)
    probe.agent = SimpleNamespace(agent_id='agent_under_test')
    probe.model = SimpleNamespace(variables={
        name: SimpleNamespace(solution=SimpleNamespace(values=np.asarray(series)))
        for name, series in values.items()})
    return probe


def _mpc_poi_probe(module, values):
    """A stub FBC POI controller.

    POI reads a different shape from linopy: the variables live on the *controller*
    (`self.variables`), each entry is a sequence of opaque solver handles, and the number behind
    a handle comes from `self.model.get_value(handle)`. The handles here are plain strings, which
    is all the controller assumes about them -- it only passes them back to `get_value`.
    """
    handles = {name: [f'{name}[{i}]' for i in range(len(series))]
               for name, series in values.items()}
    numbers = {f'{name}[{i}]': value
               for name, series in values.items() for i, value in enumerate(series)}
    probe = object.__new__(module.POI)
    probe.agent = SimpleNamespace(agent_id='agent_under_test')
    probe.variables = handles
    probe.model = SimpleNamespace(get_value=lambda handle: numbers[handle])
    return probe


def _optim_poi_probe(module, values):
    """A stub RTC POI controller.

    Same read path as the FBC one, but the RTC controller holds a single handle per variable
    rather than a sequence -- `get_solution` is `int(self.model.get_value(var))`, not a
    comprehension. `_warn_on_slack` spans both via `np.atleast_1d`, so the stub must too, or the
    scalar case would never be exercised. A series collapses to its largest value by magnitude,
    which is the number this controller would be holding for that variable.

    A consequence worth naming: this probe cannot test the aggregation. `np.atleast_1d` always
    yields one element here, so `np.max` and `np.min` are indistinguishable through it -- and
    that is faithful, because the RTC controller genuinely holds one handle per variable. The
    FBC probe is where a broken reduction would show.
    """
    handles = {name: f'{name}[]' for name in values}
    numbers = {f'{name}[]': max(series, key=abs) for name, series in values.items()}
    probe = object.__new__(module.POI)
    probe.agent = SimpleNamespace(agent_id='agent_under_test')
    probe.variables = handles
    probe.model = SimpleNamespace(get_value=lambda handle: numbers[handle])
    return probe


# The four controllers that report, with the stub each one's read path needs. `threshold` is read
# off the controller rather than hardcoded, so the probe stays honest if the constant moves.
REPORTERS = [
    (mpc_linopy, mpc_linopy.Linopy, _linopy_probe, 'mpc/linopy'),
    (mpc_poi, mpc_poi.POI, _mpc_poi_probe, 'mpc/poi'),
    (optim_linopy, optim_linopy.Linopy, _linopy_probe, 'rtc/linopy'),
    (optim_poi, optim_poi.POI, _optim_poi_probe, 'rtc/poi'),
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


class _UnreadableVariable:
    """A linopy variable whose solution cannot be read, for the failing-read test."""

    @property
    def solution(self):
        raise RuntimeError('solver handle went away')


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

    @pytest.mark.parametrize('module, controller, build_probe, name', REPORTERS,
                             ids=[n for *_, n in REPORTERS])
    def test_the_report_actually_comes_out_of_the_controller(
            self, module, controller, build_probe, name, caplog):
        """The end-to-end property: every used slack produces one record, on the controller's own
        logger, naming that agent, that variable and that peak in watts -- and nothing else does.

        This used to log to `logging.getLogger(mpc_linopy.__name__)` itself and assert `caplog`
        saw it, which tested the standard library — it called no HAMLET code, and a review panel
        confirmed it stayed green with the blanket `filterwarnings("ignore")` reinstated, i.e.
        under the exact regression its own docstring named. (It could never have caught that in
        any case: warning filters do not affect `logging`.)

        `_warn_on_slack` is now driven on a stub model instead, so the assertion is that *the
        controller* reports, at the threshold it claims to use.

        It also ran on the two linopy controllers only. POI is the default backend, and its
        `LOGGER.warning` had never executed in any test or any scenario (#231): across two full
        scenarios the scan ran 216 times over 864 slack variables and the branch was never once
        taken. POI reads a different shape, which is why it was left out; see the probe builders
        above.

        **The probe's shape is the test.** Every entry below was put there by a mutation that
        survived without it, and none of them is decoration:

        * the ordinary variable comes **first**, because with it last, turning the filter's
          `continue` into a `break` stopped the scan before any slack and still produced the one
          message the earlier version asserted. In the real controllers the non-slack variables
          come first, so that mutation silenced essentially every report in production;
        * it is enormous, pinning that selection is by suffix and not by magnitude -- the first
          draft called it `not_a_slack`, which *ends in* `_slack`, so the test duly reported it;
        * `..._slack_headroom` **contains** `_slack` without ending in it, pinning `endswith`
          against a widening to `in`, which is the same mistake from the other side;
        * **two** slacks are above the threshold, in different energy types, because with one, a
          `break` after `LOGGER.warning` reported the first and stayed green;
        * one slack sits below the threshold, pinning that the comparison is load-bearing:
          without it, the suffix alone still yielded the expected count;
        * the peaks are **absolute watts, not offsets from the threshold**. Deriving them from
          `SLACK_REPORTING_THRESHOLD` made the constant unfalsifiable -- the inputs scaled with
          it, so setting it to 1e6 W silenced a megawatt of shed energy and nothing failed. These
          values bracket it instead: raise it past 1500 and the heat report vanishes, drop it
          below 1e-4 and the hydrogen decoy starts reporting. The sibling test below pins the
          order of magnitude directly.

        Slack is `lb=0` in all four controllers, so the values here are positive: a negative
        slack is not a state any of them can reach, and a probe that fed one would be pinning
        `np.abs` against a case production cannot produce.
        """
        electricity = f'{c.ET_ELECTRICITY}_{c.OM_GENERATION}_slack'
        heat = f'{c.ET_HEAT}_{c.OM_LOAD}_slack'
        hydrogen = f'{c.ET_H2}_{c.OM_GENERATION}_slack'
        probe = build_probe(module, {
            'power_electricity': [1e9],
            f'{electricity}_headroom': [1e9],
            electricity: [0.0, 3000.0, 0.0],
            heat: [0.0, 0.0, 1500.0],
            hydrogen: [0.0, 1e-4, 0.0],
        })

        with caplog.at_level(logging.WARNING, logger=module.__name__):
            probe._warn_on_slack()

        messages = sorted(record.getMessage() for record in caplog.records)
        assert len(messages) == 2, (
            f'{name}: expected one report per above-threshold slack and nothing else -- not the '
            f'sub-threshold {hydrogen}, not {electricity}_headroom, not power_electricity -- '
            f'got {messages}')

        # Substring presence is not enough: with three `in` checks, swapping the agent and the
        # variable in the format arguments produced `Agent electricity_gen_slack: ... of
        # "agent_under_test"` and passed. The order and the unit are asserted with them.
        for variable, peak in ((electricity, 3000.0), (heat, 1500.0)):
            pattern = rf'Agent agent_under_test: .*\b{peak:.1f} W\b.*"{variable}".*'
            assert sum(bool(re.fullmatch(pattern, m)) for m in messages) == 1, (
                f'{name}: exactly one report must name the agent, then {peak:.1f} W, then '
                f'"{variable}", in that order: {messages}')

        # The logger the record carries is how a user tells the backends apart and how any
        # per-module level or filter finds it. `caplog.at_level(..., logger=...)` sets a level;
        # it does not restrict capture, so without this a report moved to another logger, or
        # dropped to a level nobody watches, is invisible here.
        for record in caplog.records:
            assert record.name == module.__name__, (
                f"{name}: reported on {record.name!r}, not this controller's own logger")
            assert record.levelno == logging.WARNING, (
                f'{name}: reported at {record.levelname}, not WARNING')

    @pytest.mark.parametrize('controller, name', OBJECTIVES, ids=[n for _, n in OBJECTIVES])
    def test_the_threshold_is_a_solver_tolerance(self, controller, name):
        """The constant's *value*, which the report's inputs must not be derived from.

        `SLACK_REPORTING_THRESHOLD` exists to separate solver tolerance from a real imbalance, and
        its comment says so. Nothing pinned the number: raising it to 1e6 W silenced a megawatt of
        shed energy with the whole suite green, because the one test that drove the report built
        its inputs as `threshold + 1000`. A watt is already far above any solver's feasibility
        tolerance and far below anything worth calling an imbalance.
        """
        assert 0 < controller.SLACK_REPORTING_THRESHOLD <= 1.0, (
            f'{name}: a threshold of {controller.SLACK_REPORTING_THRESHOLD} W is not a solver '
            f'tolerance -- above about a watt it starts hiding real shed energy')

    @pytest.mark.parametrize('module, controller, build_probe, name', REPORTERS,
                             ids=[n for *_, n in REPORTERS])
    def test_a_failing_read_is_not_swallowed(
            self, module, controller, build_probe, name, caplog):
        """A slack value that cannot be read must raise, not be skipped in silence.

        The POI controllers wrapped the read in `except Exception: continue`. That decision --
        skip this variable and carry on -- is the wrong one whichever exception reaches it: the
        result is an agent that shed energy and reported nothing, which is exactly what
        `_warn_on_slack` exists to make impossible. Why that is safe in POI is argued where it
        belongs, in the source comment on `_warn_on_slack` itself.

        Runs on all four so that no backend can grow a *broad* handler. Read the limit before
        trusting it: the two linopy controllers keep a narrow `except (AttributeError,
        ValueError)`, and this test is deliberately blind to it. `RuntimeError` is therefore
        load-bearing, not arbitrary -- with `ValueError` here the two linopy parametrisations
        fail against unmodified code. Widening either linopy handler to `except Exception` does
        fail this test; tightening the POI ones back does too. Narrowing linopy's further, or
        judging whether those two types are the right ones, is #231's out-of-scope neighbour.
        """
        probe = build_probe(module, {'balance_electricity_slack': [1e6]})

        def explode(*_args, **_kwargs):
            raise RuntimeError('solver handle went away')

        # Dispatched on the probe builder, not on `hasattr(probe, 'variables')`: were a linopy
        # controller ever to grow a `self.variables`, that test would silently take the POI
        # branch, break nothing that is read, and pass for the wrong reason.
        if build_probe is _linopy_probe:          # linopy: the read goes through `.solution`
            probe.model.variables['balance_electricity_slack'] = _UnreadableVariable()
        else:                                     # POI: the read goes through `model.get_value`
            probe.model = SimpleNamespace(get_value=explode)

        with caplog.at_level(logging.WARNING, logger=module.__name__):
            with pytest.raises(RuntimeError, match='solver handle went away'):
                probe._warn_on_slack()

        assert not caplog.records, (
            f'{name}: a failed read reported something anyway: '
            f'{[r.getMessage() for r in caplog.records]}')

    @pytest.mark.parametrize('module', [mpc_linopy, optim_linopy], ids=['mpc', 'rtc'])
    def test_the_solve_path_reports_before_returning(self, module):
        """The check has to run on every solve, not only on a failure branch."""
        source = inspect.getsource(module.Linopy.run)

        assert '_warn_on_slack' in source
