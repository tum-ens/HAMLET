"""Unit — the "keep the solver quiet" options use each solver's own names.

`mpc_linopy` and `optim_linopy` sent the literal `{'OutputFlag': 0, 'LogToConsole': 0}` to
whichever solver was configured. Those are Gurobi's names; HiGHS discards them unrecognised, so
under the default backend the flags did nothing and the `sys.stdout = open(os.devnull, 'w')`
around the solve was the only thing suppressing output (#199, roadmap item #11).

That is the same defect as #204's `TimeLimit`, which was also a Gurobi key sent to HiGHS, and
which made a solver time limit invisible for years. The difference is only in consequence: a
discarded log flag prints, a discarded time limit changes results. Both are cured by looking the
name up per solver, which is what `solver_options` now does for all three options.

These assertions are about *names and types*, deliberately. A solver either recognises an option
or silently ignores it, and there is no return value that says which -- so the spelling is the
thing to pin, and pinning it here is cheaper than a solver round trip.
"""
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


class TestTheCallersUseThem:
    """Two correct dicts that nobody sends are worth nothing."""

    @pytest.mark.parametrize('module_path', [
        'hamlet.executor.utilities.controller.fbc.mpc.linopy.mpc_linopy',
        'hamlet.executor.utilities.controller.rtc.optim.linopy.optim_linopy',
    ])
    def test_the_linopy_controllers_send_quiet_options(self, module_path):
        import importlib
        import inspect

        source = inspect.getsource(importlib.import_module(module_path).Linopy.run)

        assert 'quiet_options(solver)' in source, (
            'the controller builds its solver options some other way; the literal '
            "{'OutputFlag': 0, 'LogToConsole': 0} is what this replaced")
        assert "'OutputFlag'" not in source, (
            'a Gurobi option name is written out in the controller again')

    @pytest.mark.parametrize('module_path', [
        'hamlet.executor.utilities.controller.fbc.mpc.linopy.mpc_linopy',
        'hamlet.executor.utilities.controller.rtc.optim.linopy.optim_linopy',
    ])
    def test_the_stdout_redirect_is_exception_safe(self, module_path):
        """The other half of item #11: the previous form assigned `sys.stdout`, restored it only
        on the success path, and leaked the devnull handle on every solve. A solve that raised
        left the process writing to devnull for good."""
        import importlib
        import inspect

        source = inspect.getsource(importlib.import_module(module_path).Linopy.run)

        assert 'with redirect_stdout(' in source
        assert 'sys.stdout =' not in source, (
            'stdout is assigned rather than redirected, so it is not restored when the solve '
            'raises')
