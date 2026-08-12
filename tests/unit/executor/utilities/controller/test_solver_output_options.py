"""Unit — the "keep the solver quiet" options use each solver's own names.

`mpc_linopy` and `optim_linopy` sent the literal `{'OutputFlag': 0, 'LogToConsole': 0}` to
whichever solver was configured. Those are Gurobi's names; HiGHS discards them unrecognised, so
under the default backend the flags did nothing and the `sys.stdout = open(os.devnull, 'w')`
around the solve was the only thing suppressing output (#199, roadmap item #11).

That is the same defect as #204's `TimeLimit`, which was also a Gurobi key sent to HiGHS, and
which made a solver time limit invisible for years. The difference is only in consequence: a
discarded log flag prints, a discarded time limit changes results. Both are cured by looking the
name up per solver, which is what `solver_options` now does for all three options.

These assertions are about *names and types*, deliberately. HiGHS does not accept an unknown
option quietly -- it answers `ERROR: getOptionIndex: Option "OutputFlag" is unknown`, from C,
on every solve -- but linopy discards the return status and the message went to a file
descriptor the old stdout hijack never touched. So nothing in the call tells the caller which
spelling was accepted, and the spelling is the thing to pin.
"""
import io
import sys

import pytest

from hamlet.executor.utilities.controller.solver_options import (KNOWN_SOLVERS, quiet_options,
                                                                 reproducibility_options)


class TestEachSolverGetsItsOwnSpelling:

    def test_highs_gets_the_lower_case_snake_case_names(self):
        """HiGHS' option names, from its own options table. `OutputFlag` is not one of them."""
        assert quiet_options('highs') == {'output_flag': False, 'log_to_console': False}

    def test_gurobi_gets_the_capitalised_camel_case_names(self):
        """Gurobi's names. This is the pair that was being sent to both solvers."""
        assert quiet_options('gurobi') == {'OutputFlag': 0, 'LogToConsole': 0}

    def test_neither_solver_is_sent_the_other_s_names(self):
        """The property behind the two cases above, stated so a third solver cannot be added
        by copying one of the tables and forgetting to rename it."""
        highs, gurobi = set(quiet_options('highs')), set(quiet_options('gurobi'))

        assert not highs & gurobi, (
            f'{sorted(highs & gurobi)} is sent to both solvers under the same name, which is the '
            f'defect this module exists to prevent')

    def test_the_types_match_what_each_solver_declares(self):
        """PyOptInterface dispatches `set_raw_parameter` on the Python type: HiGHS declares these
        options bool and rejects `0`/`1`, Gurobi declares them int. The same trap as the time
        limit having to be a float in `reproducibility_options`."""
        assert all(isinstance(value, bool) for value in quiet_options('highs').values())
        assert all(isinstance(value, int) and not isinstance(value, bool)
                   for value in quiet_options('gurobi').values())


class TestItRefusesWhatItDoesNotKnow:

    def test_an_unknown_solver_raises(self):
        """Rather than returning an empty dict, which would suppress nothing and look fine."""
        with pytest.raises(ValueError, match='Unsupported solver'):
            quiet_options('cplex')

    @pytest.mark.parametrize('solver', KNOWN_SOLVERS)
    def test_every_known_solver_has_both_flags(self, solver):
        """A half-filled table would quieten one stream and leave the other printing."""
        assert len(quiet_options(solver)) == 2


class TestTheTwoOptionSetsStaySeparate:
    """`quiet_options` must not be able to change a result, and the split is what says so."""

    @pytest.mark.parametrize('solver', KNOWN_SOLVERS)
    def test_they_do_not_overlap(self, solver):
        """If a name appeared in both, merging them in the callers would make the order matter."""
        quiet = set(quiet_options(solver))
        reproducibility = set(reproducibility_options(solver, time_limit=120))

        assert not quiet & reproducibility


CONTROLLERS = ['hamlet.executor.utilities.controller.fbc.mpc.linopy.mpc_linopy',
               'hamlet.executor.utilities.controller.rtc.optim.linopy.optim_linopy']


def probe_controller(module_path, solver, raise_on_solve=False):
    """A linopy controller with just enough around it to reach the solve, and no further.

    Constructed with `object.__new__` so the real `__init__` -- which wants an AgentDB, a
    forecast and a built model -- is not involved. `_warn_on_slack` and `process_solution` are
    stubbed because the solve is what is under test here.

    This exists because the assertions it replaces were `inspect.getsource` substring checks, and
    a review panel defeated all four of them at once: reverting both controllers to the pre-fix
    code while leaving a comment naming `quiet_options(solver)` and `redirect_stdout` kept them
    green, and one of them was additionally beaten by deleting a single space (`sys.stdout=`).
    Source text is not behaviour.
    """
    import importlib
    from types import SimpleNamespace

    controller = importlib.import_module(module_path).Linopy
    probe = object.__new__(controller)
    probe.ems = {'solver': solver, 'time_limit': 120}
    probe.agent = SimpleNamespace(agent_id='probe')
    probe.sent = {}

    class Model:
        def solve(inner, solver_name=None, **kwargs):
            probe.sent = dict(kwargs, solver_name=solver_name)
            if raise_on_solve:
                raise RuntimeError('the solver blew up')
            return ('ok', 'optimal')

    probe.model = Model()
    probe._warn_on_slack = lambda: None
    probe.process_solution = lambda: probe.agent
    return probe


class TestTheCallersUseThem:
    """Two correct dicts that nobody sends are worth nothing."""

    @pytest.mark.parametrize('module_path', CONTROLLERS, ids=['mpc', 'rtc'])
    @pytest.mark.parametrize('solver', KNOWN_SOLVERS)
    def test_the_controller_sends_that_solver_its_own_names(self, module_path, solver):
        """Read off the call, so hardcoding either solver's spelling fails on the other arm."""
        probe = probe_controller(module_path, solver)

        probe.run()

        expected = {**quiet_options(solver), **reproducibility_options(solver, 120),
                    'solver_name': solver}
        assert probe.sent == expected, (
            f'{module_path} sent {probe.sent} for solver {solver!r}, expected {expected}')

    @pytest.mark.parametrize('module_path', CONTROLLERS, ids=['mpc', 'rtc'])
    def test_stdout_is_restored_to_the_callers_object(self, module_path, monkeypatch):
        """Not to `sys.__stdout__`.

        The old code ended with `sys.stdout = sys.__stdout__`, which is the *process's* original
        stdout rather than whatever the caller had installed -- so the first solve under pytest
        tore down pytest's capture for everything after it.
        """
        sentinel = io.StringIO()
        monkeypatch.setattr(sys, 'stdout', sentinel)

        probe_controller(module_path, 'highs').run()

        assert sys.stdout is sentinel, (
            'the solve replaced the caller\'s stdout with something else and did not put it back')

    @pytest.mark.parametrize('module_path', CONTROLLERS, ids=['mpc', 'rtc'])
    def test_stdout_is_restored_when_the_solve_raises(self, module_path, monkeypatch):
        """The failure that made the old form more than untidy.

        `sys.stdout = open(os.devnull, 'w')` with the restore on the line *after* the solve meant
        a raising solve left the whole process writing to devnull permanently, and leaked the
        handle every time.
        """
        sentinel = io.StringIO()
        monkeypatch.setattr(sys, 'stdout', sentinel)

        probe = probe_controller(module_path, 'highs', raise_on_solve=True)
        with pytest.raises(RuntimeError, match='blew up'):
            probe.run()

        assert sys.stdout is sentinel, (
            'stdout was not restored after the solve raised, so everything printed from here on '
            'goes to wherever the controller pointed it')


class TestThePoiBackendUsesThemToo:
    """`create_model` had the names right before this change; the point of sharing the table is
    that it cannot drift away from the linopy side. Deleting its loop left the whole unit suite
    green until this test existed."""

    @pytest.mark.parametrize('solver', KNOWN_SOLVERS)
    def test_create_model_applies_the_quiet_options(self, solver, monkeypatch):
        """Stubs the PyOptInterface module, so it runs without either solver being installed."""
        from types import SimpleNamespace

        from hamlet.executor.utilities.controller import poi_solver

        applied = {}

        class Model:
            def __init__(self, *args):
                pass

            def set_raw_parameter(self, name, value):
                applied[name] = value

            def set_model_attribute(self, attribute, value):
                pass

        class Env:
            def __init__(self, **kwargs):
                pass

            def set_raw_parameter(self, name, value):
                pass

            def start(self):
                pass

        monkeypatch.setattr(poi_solver, 'get_solver_module',
                            lambda _: SimpleNamespace(Model=Model, Env=Env))

        poi_solver.create_model(solver)

        assert applied == quiet_options(solver), (
            f'create_model applied {applied} for {solver!r}, expected {quiet_options(solver)}')
