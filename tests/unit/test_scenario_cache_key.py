"""Unit — the run cache's key, and what happens to a consumer when it is wrong.

A shared run is a way for a test to stop exercising what it believes it does: ask for `linopy`,
be handed a cached `poi`, and agree with yourself. `tests/scenario_cache.py` claims two defences
against that, and they catch different things. This file breaks each one on purpose and shows the
*other* one firing, because a defence that has never been observed rejecting anything is a
comment.

The runner is stubbed rather than real. That is the point rather than a shortcut: what is under
test is the key and the read-backs, and a stub lets the failure be *constructed* instead of hoped
for. The stub writes a receipt in the real on-disk format and every rejection below comes from
production code — `assert_backend_honoured` for the receipt, `ScenarioRuns` for the request.
"""
import inspect
import json

import pytest

from tests.scenario_cache import NOT_PART_OF_THE_REQUEST, ScenarioRuns, request_key
from tests.scenario_run import run_example

EXAMPLE = 'examples/create_simple_scenario'
SCENARIO = 'simple_scenario'

#: Every argument that decides *what* a run does, written out rather than derived. Deriving it the
#: way `request_key` does would make the test below a tautology: both sides would shrink together
#: when a name was added to `NOT_PART_OF_THE_REQUEST`, which is the one change that can silently
#: merge two different requests, and the test would go on passing. Spelled out, that change fails
#: here and has to be argued for.
KEYED_ARGUMENTS = {'example_dir', 'scenario_name', 'framework', 'solver', 'edits', 'config_edits',
                   'creator_method'}


def stub_runner(base, example_dir, scenario_name, framework=None, solver=None, edits=(),
                config_edits=None, record_backends=None,
                creator_method='new_scenario_from_configs', solved_by=None):
    """Stand in for `run_example`: write the artefacts a consumer reads, run nothing.

    `solved_by` overrides what the receipt claims solved the run, so a run that *ignores* the
    backend it was handed — #206's failure, which no comparison of requests can see — can be
    constructed. Everything else mirrors what a real run leaves behind.
    """
    (base / 'results' / scenario_name).mkdir(parents=True, exist_ok=True)
    if record_backends is not None:
        record_backends.write_text(
            json.dumps([list(solved_by or (framework or 'poi', solver or 'highs'))]),
            encoding='utf-8')
    return {'agents/sfh/meters.ft': {'files': 1, 'rows': 1, 'columns': {}}}


def key_without_framework(*args, **kwargs):
    """`request_key` with `framework` dropped — the omission the real key exists to prevent.

    Modelled on how this actually goes wrong here: not by someone deleting a field, but by an
    enumerated list that never grew a new one. `ROUNDING`, `KEYS` and `AGENT_TABLES` have each
    passed by omission in this repository.
    """
    return tuple((name, value) for name, value in request_key(*args, **kwargs)
                 if name != 'framework')


class TestTheKeyIsTheWholeRequest:

    def test_every_argument_that_decides_what_a_run_does_is_in_the_key(self):
        """The mechanical derivation, checked against a written-out list. See `KEYED_ARGUMENTS`."""
        keyed = {name for name, _ in request_key(EXAMPLE, SCENARIO)}
        signature = set(inspect.signature(run_example).parameters)

        assert keyed == KEYED_ARGUMENTS, (
            f'the key covers {sorted(keyed)}, not {sorted(KEYED_ARGUMENTS)}; an argument outside '
            f'the key lets two different requests share one run')
        assert signature - NOT_PART_OF_THE_REQUEST == KEYED_ARGUMENTS, (
            f'run_example now takes {sorted(signature)}, which no longer matches the written-out '
            f'list. A new parameter is keyed automatically -- add it to KEYED_ARGUMENTS. A newly '
            f'*excluded* one is the dangerous direction and needs an argument, not an edit')

    @pytest.mark.parametrize('difference', [
        {'example_dir': 'examples/create_scenario_with_grid'},
        {'scenario_name': 'grid_golden'},
        {'framework': 'linopy'},
        {'solver': 'gurobi'},
        {'creator_method': 'new_scenario_from_files'},
        {'edits': (('a', 'b'),)},
        {'config_edits': {'grids.yaml': [('a', 'b')]}},
    ])
    def test_a_request_differing_in_any_one_argument_is_a_different_key(self, difference):
        """One case per keyed argument, the two positional ones included.

        `example_dir` matters more than it looks: `test_solver_backend_smoke` records that no two
        shipped examples share a scenario folder name only as a coincidence, so the day that ends,
        `example_dir` is the sole thing telling two runs apart.

        Passed by keyword throughout so the two positional arguments can be varied the same way as
        the rest; `request_key` binds against the signature, so the two spellings are equivalent
        (`test_the_same_request_is_the_same_key` covers that they agree).
        """
        base = {'example_dir': EXAMPLE, 'scenario_name': SCENARIO}

        assert request_key(**base) != request_key(**{**base, **difference})

    def test_the_same_request_is_the_same_key(self):
        assert request_key(EXAMPLE, SCENARIO, framework='poi') == request_key(
            EXAMPLE, SCENARIO, framework='poi')

    def test_positional_and_keyword_spellings_of_one_request_agree(self):
        """Binding against the signature is what makes this true, and several tests rely on it."""
        assert request_key(EXAMPLE, SCENARIO) == request_key(
            example_dir=EXAMPLE, scenario_name=SCENARIO)

    def test_edits_are_order_significant(self):
        """`edits` is applied as sequential string replacements, so order is part of the request."""
        one = (('a', 'b'), ('c', 'd'))
        assert request_key(EXAMPLE, SCENARIO, edits=one) != request_key(
            EXAMPLE, SCENARIO, edits=one[::-1])


class TestAMisServedConsumerIsCaught:
    """The two defences, each shown catching what only it can catch."""

    def test_two_frameworks_do_not_share_a_run(self, tmp_path_factory):
        """The control. With the real key these are two requests and two runs."""
        runs = ScenarioRuns(tmp_path_factory, runner=stub_runner)

        runs.run(EXAMPLE, SCENARIO, framework='linopy')
        runs.run(EXAMPLE, SCENARIO, framework='poi')

        assert len(runs.log) == 2, (
            f'expected one run per framework, got {len(runs.log)}; if this is 1 the two requests '
            f'collided and the tests below are no longer demonstrating anything')

    def test_a_broken_key_is_caught_by_comparing_requests(self, tmp_path_factory):
        """Defence one. Break the key, and the second consumer must go red.

        This is the argument for comparing the full request rather than trusting the key: the key
        that decided the two requests were the same cannot also certify it, and here it is wrong.
        The comparison is between two *argument sets*, so it does not go through the key and
        covers every keyed argument -- including the four a receipt can say nothing about.
        """
        runs = ScenarioRuns(tmp_path_factory, runner=stub_runner, key=key_without_framework)

        runs.run(EXAMPLE, SCENARIO, framework='linopy')

        with pytest.raises(AssertionError, match=r"differs from this one in \['framework'\]"):
            runs.run(EXAMPLE, SCENARIO, framework='poi')

        assert len(runs.log) == 1, (
            'the premise of this test is that the broken key merged the two requests into one '
            'run; it did not, so the rejection above proves nothing')

    def test_a_run_that_ignored_its_backend_is_caught_by_the_receipt(self, tmp_path_factory):
        """Defence two, and the only one that can see this.

        Here the key is right and the request comparison passes — the consumer got exactly the
        entry it asked for. What is wrong is the *run*: it was told `linopy` and solved with
        `poi`. That is #206, and no amount of comparing requests to each other can detect it,
        because both sides agree. Only reading what the run reported does.
        """
        runs = ScenarioRuns(
            tmp_path_factory,
            runner=lambda base, **call: stub_runner(base, solved_by=('poi', 'highs'), **call))

        with pytest.raises(AssertionError, match=r"asked for framework 'linopy'"):
            runs.run(EXAMPLE, SCENARIO, framework='linopy')

    def test_a_collision_across_scenarios_is_caught_with_no_backend_named(self, tmp_path_factory):
        """Requests naming no framework or solver have only defence one, so it must hold alone.

        `assert_backend_honoured` deliberately checks only the axes a caller asked for, so it has
        nothing to say here — and this is the golden master's request shape, which makes it the
        one that most needs the other check.
        """
        runs = ScenarioRuns(tmp_path_factory, runner=stub_runner,
                            key=lambda *a, **k: 'every request is the same')

        runs.run(EXAMPLE, SCENARIO)

        with pytest.raises(AssertionError, match='differs from this one'):
            runs.run('tests/e2e/scenarios', 'grid_golden')


class TestTheReceiptIsWrittenOnlyWhenItIsNeeded:
    """`needs_receipt` keeps `BACKEND_PROBE` out of the runs that do not want it.

    The golden master's run must stay the run that produced the committed reference, and the probe
    monkeypatches `create_model` and `linopy.Model.solve`. Naming no backend and not asking for a
    receipt is what keeps it probe-free.
    """

    def test_a_request_naming_no_backend_gets_no_receipt(self, tmp_path_factory):
        runs = ScenarioRuns(tmp_path_factory, runner=stub_runner)

        assert runs.run(EXAMPLE, SCENARIO).record is None

    def test_asking_for_a_receipt_produces_one(self, tmp_path_factory):
        runs = ScenarioRuns(tmp_path_factory, runner=stub_runner)

        entry = runs.run(EXAMPLE, SCENARIO, needs_receipt=True)

        assert entry.record is not None and entry.record.exists()

    def test_the_receipt_is_part_of_the_key(self, tmp_path_factory):
        """Otherwise a probe-free run could be served to a caller that needs the receipt, or --
        the direction that matters -- a probe-on run to the golden master."""
        runs = ScenarioRuns(tmp_path_factory, runner=stub_runner)

        runs.run(EXAMPLE, SCENARIO)
        runs.run(EXAMPLE, SCENARIO, needs_receipt=True)

        assert len(runs.log) == 2


def test_the_one_duplicated_pair_in_the_suite_still_shares_its_run():
    """`test_grid_restrictions` and `test_golden_master[grid_golden]` must key identically.

    This is the entire measurable saving -- one `grid_golden` run -- and without this test
    it is asserted only by a wall-clock difference that this runner cannot resolve: the two paired
    arms measured for it came back -334 s and -28 s, a spread larger than the effect. Timing
    cannot tell "the run was shared" from "the runner was busy". Key equality can, it costs
    nothing, and it fails by name the day someone changes one call site and not the other.

    Deliberately built from the two modules' own constants rather than from a copy of the
    arguments, so it cannot agree with itself: if either module changes what it asks for, the two
    keys diverge here.
    """
    from tests.e2e import test_golden_master as golden
    from tests.e2e import test_grid_restrictions as restrictions

    pinned = [scenario for scenario in golden.SCENARIOS if scenario.name == restrictions.SCENARIO]
    assert len(pinned) == 1, (
        f'the golden master no longer pins {restrictions.SCENARIO!r} (it pins '
        f'{[scenario.name for scenario in golden.SCENARIOS]}), so there is nothing for '
        f'test_grid_restrictions to share a run with')

    assert request_key(pinned[0].config_dir, pinned[0].name,
                       creator_method=pinned[0].creator_method) == request_key(
        restrictions.CONFIG_ROOT, restrictions.SCENARIO,
        creator_method=restrictions.CREATOR_METHOD), (
        'test_grid_restrictions and test_golden_master no longer make the same request for '
        f'{restrictions.SCENARIO!r}, so they now pay for a run each. That is correct behaviour '
        f'from the cache -- two different requests must not share -- but it means the saving this '
        f'module exists to protect is gone, and whichever call site changed should say why')


def test_the_results_directory_is_checked(tmp_path_factory):
    """A run that writes no results fails here, naming that, rather than further downstream.

    `run_example` asserts only on `RUN_OK` and `fingerprint()` returns `{}` for a missing
    directory rather than raising, so without this an Executor that wrote nothing would surface as
    a puzzling empty comparison.
    """
    runs = ScenarioRuns(tmp_path_factory,
                        runner=lambda base, **call: {'agents/sfh/meters.ft': {'rows': 0}})

    with pytest.raises(AssertionError, match='wrote no results directory'):
        runs.run(EXAMPLE, SCENARIO)
