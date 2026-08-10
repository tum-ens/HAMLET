"""End to end — what `framework: linopy` and `framework: poi` do and do not share.

`framework` is a per-agent configuration option; `poi` is the default and `linopy` remains fully
supported. Both backends are handed the same scenario and the same solver, so the only thing that
differs is how the model is expressed.

**The models are equivalent and the whole-run outputs still differ, permanently.** Both controllers
were compared by exporting their models to LP and diffing by constraint shape -- sense, RHS and
coefficient multiset, invariant to the variable naming and ordering that differ between backends --
and they match, with first-timestep objectives agreeing to ~1e-12 (MPC) and exactly (RTC). What
diverges downstream is a *tie*: the agent MILPs are degenerate, the two backends present an equal
optimum at a different vertex, and `rtc_base.update_socs` quantises that vertex into a state of
charge the next timestep reads and amplifies.

**So these tests are `xfail(strict=True)` permanently, not pending a fix.** That is unusual enough
to spell out: an xfail normally marks something to repair, and this one marks a comparison that
cannot succeed on whole-run outputs however correct both backends are. It is kept because it is
still load-bearing in one direction -- it fails loudly if the divergence ever *disappears*, which
would mean the two arms had stopped being two arms (a framework key renamed, a switch silently
no-oping). The evidence that this is degeneracy rather than a defect lives in #198 and in
`tests/e2e/test_golden_master.py`'s "when it fails" guidance; do not delete the marker on the
strength of the models being equivalent, because that was already established.

    python -m pytest tests -m e2e

Each test runs the example twice, so budget a few minutes.
"""
import pytest

from tests.scenario_run import REPO_ROOT, compare, run_example

EXAMPLE = REPO_ROOT / 'examples' / 'create_simple_scenario'
SCENARIO_NAME = 'simple_scenario'

# The golden master's own tolerance. Solver output is bit-stable on a fixed platform, but HiGHS and
# polars versions move; this is loose enough to survive that and far tighter than any real
# modelling change. Deliberately *not* widened to accommodate the current divergence -- a band that
# admitted a 100 % difference would assert nothing.
RELATIVE_TOLERANCE = 1e-6

# See the module docstring: this is a permanent xfail, not a pending repair. The models are
# equivalent; whole-run outputs still differ because a degenerate MILP's tie breaks differently
# and `update_socs` feeds that vertex into the next timestep.
known_divergence = pytest.mark.xfail(
    strict=True,
    reason='POI and linopy diverge on whole-run output through degeneracy amplified by state '
           'feedback, though the models themselves are equivalent (#198)')


@pytest.fixture(scope='module')
def linopy_results(tmp_path_factory):
    """The reference implementation, still fully supported and selectable."""
    return run_example(tmp_path_factory.mktemp('linopy'), EXAMPLE, SCENARIO_NAME,
                       framework='linopy')


@pytest.fixture(scope='module')
def poi_results(tmp_path_factory):
    """The example as shipped. Also the arm that must match the golden master."""
    return run_example(tmp_path_factory.mktemp('poi'), EXAMPLE, SCENARIO_NAME, framework='poi')


@pytest.mark.e2e
@pytest.mark.solver
def test_the_poi_arm_reproduces_the_golden_master(poi_results):
    """The control, and it must pass even while the comparison below does not.

    Without this, a failing equivalence test is ambiguous: it could mean the backends diverge, or
    that this harness does not run the example the way the golden master does. This pins it to the
    second interpretation being false.

    **This was the linopy arm until `poi` became the default.** The control has to be whichever
    backend the shipped config selects, because that is what the golden master runs -- pointing it
    at the other one turns a passing control into an assertion that the two backends agree, which
    is precisely what the tests below record that they do not.
    """
    import json

    reference = json.loads(
        (REPO_ROOT / 'tests' / 'e2e' / 'golden' / f'{SCENARIO_NAME}.json').read_text(
            encoding='utf-8'))

    differences = compare(poi_results, reference, 'this harness', 'golden master',
                          RELATIVE_TOLERANCE)
    assert not differences, (
        'the poi arm no longer matches the committed golden master, so the backend comparison '
        'in this file cannot be trusted:\n  ' + '\n  '.join(differences[:20]))


@known_divergence
@pytest.mark.e2e
@pytest.mark.solver
def test_the_two_backends_produce_the_same_tables(linopy_results, poi_results):
    """A table or row count moving is a result change, not a rounding difference."""
    differences = [d for d in compare(linopy_results, poi_results, 'linopy', 'poi',
                                      RELATIVE_TOLERANCE)
                   if 'rows in' in d or 'tables only in' in d]
    assert not differences, 'structure differs between backends:\n  ' + '\n  '.join(differences)


@known_divergence
@pytest.mark.e2e
@pytest.mark.solver
def test_the_two_backends_produce_the_same_numbers(linopy_results, poi_results):
    """The substance: every numeric column's total, minimum and maximum."""
    differences = compare(linopy_results, poi_results, 'linopy', 'poi', RELATIVE_TOLERANCE)
    assert not differences, (
        f'{len(differences)} differences between backends:\n  ' + '\n  '.join(differences[:40]))
