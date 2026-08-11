"""Unit -- the solver options that decide whether a run is reproducible (#204).

Three separate defects made the shipped example's results a function of machine load, and each
one is pinned here:

1. Nothing set a thread count, so both solvers picked their own. A parallel MIP's incumbent
   depends on how its threads interleave.
2. The configured `time_limit` was divided by 60 on its way to the solver, so the example's
   `time_limit: 120` arrived as 2 seconds.
3. `TerminationStatusCode.TIME_LIMIT` was whitelisted alongside `OPTIMAL`, so a solve that ran out
   of time had its incumbent accepted silently -- and every other bad status was ignored too,
   because the `raise` was commented out.

Measured on the shipped example under artificial CPU load before the fix: 1 of 192 solves returned
`TIME_LIMIT` and was accepted. That is why (2) and (3) are not theoretical.

The last two tests apply the options to a real solver rather than to a dict, because the mapping
being right on paper is not the claim that matters -- PyOptInterface dispatches `set_raw_parameter`
on the *type* of the value, so a plausible-looking dict can still be rejected at the solver.
"""
import pytest

from hamlet.executor.utilities.controller.poi_solver import (apply_reproducibility_options,
                                                             create_model, raise_unless_optimal)
from hamlet.executor.utilities.controller.solver_options import reproducibility_options
from tests.poi_support import available_backend

CONFIGURED_TIME_LIMIT = 120  # what examples/create_simple_scenario ships


class TestOptionNames:
    """Each solver spells these two options its own way and rejects the other's spelling."""

    def test_highs_pins_one_thread(self):
        assert reproducibility_options('highs')['threads'] == 1

    def test_gurobi_pins_one_thread(self):
        assert reproducibility_options('gurobi')['Threads'] == 1

    def test_an_unknown_solver_is_refused(self):
        with pytest.raises(ValueError, match='cplex'):
            reproducibility_options('cplex')


class TestTheTimeLimitIsInSeconds:
    """The #204 unit bug. A division reappearing anywhere on this path fails these."""

    @pytest.mark.parametrize('solver, key', [('highs', 'time_limit'), ('gurobi', 'TimeLimit')])
    def test_the_configured_value_arrives_unscaled(self, solver, key):
        options = reproducibility_options(solver, CONFIGURED_TIME_LIMIT)
        assert options[key] == pytest.approx(CONFIGURED_TIME_LIMIT)

    @pytest.mark.parametrize('solver, key', [('highs', 'time_limit'), ('gurobi', 'TimeLimit')])
    def test_no_limit_is_set_when_none_is_configured(self, solver, key):
        """None must leave the solver's own default alone rather than become a limit of zero."""
        assert key not in reproducibility_options(solver, None)

    def test_the_limit_is_a_float_and_the_thread_count_an_int(self):
        """PyOptInterface dispatches on the Python type, so these are not interchangeable."""
        options = reproducibility_options('highs', CONFIGURED_TIME_LIMIT)
        assert isinstance(options['time_limit'], float)
        assert isinstance(options['threads'], int) and not isinstance(options['threads'], bool)


class TestOnlyAProvenOptimumIsAccepted:
    """`raise_unless_optimal` is where a load-dependent answer becomes a loud failure."""

    @staticmethod
    def status(name):
        import pyoptinterface as poi

        return getattr(poi.TerminationStatusCode, name)

    def test_an_optimal_solve_passes(self):
        assert raise_unless_optimal(self.status('OPTIMAL'), 'agent-1', CONFIGURED_TIME_LIMIT) is None

    def test_hitting_the_time_limit_is_an_error(self):
        """It used to be whitelisted next to OPTIMAL, which is the whole of #204."""
        with pytest.raises(ValueError, match='TIME_LIMIT'):
            raise_unless_optimal(self.status('TIME_LIMIT'), 'agent-1', CONFIGURED_TIME_LIMIT)

    def test_the_time_limit_message_names_the_limit_that_was_hit(self):
        """So the reader is not left guessing which number to raise."""
        with pytest.raises(ValueError, match=str(CONFIGURED_TIME_LIMIT)):
            raise_unless_optimal(self.status('TIME_LIMIT'), 'agent-1', CONFIGURED_TIME_LIMIT)

    def test_an_infeasible_solve_is_an_error(self):
        """The `raise` for this was present but commented out, so it only ever printed."""
        with pytest.raises(ValueError, match='agent-1'):
            raise_unless_optimal(self.status('INFEASIBLE'), 'agent-1', CONFIGURED_TIME_LIMIT)


@pytest.mark.solver
class TestTheOptionsReachTheSolver:
    """Applied to a real model, because a name the solver does not know is discarded silently.

    That is not hypothetical: `OutputFlag` and `LogToConsole` are sent to HiGHS on the linopy path
    to this day and HiGHS rejects both, which is how the time limit went years without applying.
    """

    @pytest.fixture
    def model(self):
        """A fresh model per test, so one test cannot leave an option set for the next."""
        if available_backend() is None:
            pytest.skip('no PyOptInterface solver available')
        return create_model('highs')

    def test_the_thread_count_is_pinned_on_the_model(self, model):
        assert model.get_raw_parameter('threads') == 0, 'HiGHS should still default to 0 here'
        apply_reproducibility_options(model, 'highs', CONFIGURED_TIME_LIMIT)
        assert model.get_raw_parameter('threads') == 1

    def test_the_time_limit_on_the_model_is_the_configured_number_of_seconds(self, model):
        apply_reproducibility_options(model, 'highs', CONFIGURED_TIME_LIMIT)
        assert model.get_raw_parameter('time_limit') == pytest.approx(CONFIGURED_TIME_LIMIT)
