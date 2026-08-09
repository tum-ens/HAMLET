"""End to end — `framework: linopy` and `framework: poi` must produce the same results.

`framework` is a per-agent configuration option, so a user can select `poi` today and get whatever
that backend computes. Both backends are handed the same scenario and the same solver, so the only
thing that differs is how the model is expressed; equal expressions must give equal results.

**These tests currently fail, and that is the point** -- they are `xfail(strict=True)`, so they
report the known defect without reddening the pipeline, and they will fail *loudly* the moment the
backends are reconciled, which is the signal to delete the marker. See #198 for the measurement:
3 row counts and 110 column statistics move, by up to 100 %.

    python -m pytest tests -m e2e

Each test runs the example twice, so budget a few minutes.
"""
import sys

import pytest

from tests.scenario_run import REPO_ROOT, compare, run_example

EXAMPLE = REPO_ROOT / 'examples' / 'create_simple_scenario'
SCENARIO_NAME = 'simple_scenario'

# The golden master's own tolerance. Solver output is bit-stable on a fixed platform, but HiGHS and
# polars versions move; this is loose enough to survive that and far tighter than any real
# modelling change. Deliberately *not* widened to accommodate the current divergence -- a band that
# admitted a 100 % difference would assert nothing.
RELATIVE_TOLERANCE = 1e-6

not_on_windows = pytest.mark.skipif(
    sys.platform == 'win32',
    reason='framework: poi segfaults on Windows; see tests/poi_support.py')

known_divergence = pytest.mark.xfail(
    strict=True,
    reason='the POI backend does not reproduce linopy (#198): 3 row counts and 85 column '
           'statistics still move, on agents owning a battery or an EV. Remove this when fixed.')


@pytest.fixture(scope='module')
def linopy_results(tmp_path_factory):
    """The example as shipped. Also the arm that must match the golden master."""
    return run_example(tmp_path_factory.mktemp('linopy'), EXAMPLE, SCENARIO_NAME,
                       framework='linopy')


@pytest.fixture(scope='module')
def poi_results(tmp_path_factory):
    return run_example(tmp_path_factory.mktemp('poi'), EXAMPLE, SCENARIO_NAME, framework='poi')


@pytest.mark.e2e
@pytest.mark.solver
def test_the_linopy_arm_reproduces_the_golden_master(linopy_results):
    """The control, and it must pass even while the comparison below does not.

    Without this, a failing equivalence test is ambiguous: it could mean the POI backend diverges,
    or that this harness does not run the example the way the golden master does. This pins it to
    the second interpretation being false.
    """
    import json

    reference = json.loads(
        (REPO_ROOT / 'tests' / 'e2e' / 'golden' / f'{SCENARIO_NAME}.json').read_text(
            encoding='utf-8'))

    differences = compare(linopy_results, reference, 'this harness', 'golden master',
                          RELATIVE_TOLERANCE)
    assert not differences, (
        'the linopy arm no longer matches the committed golden master, so the backend comparison '
        'in this file cannot be trusted:\n  ' + '\n  '.join(differences[:20]))


@not_on_windows
@known_divergence
@pytest.mark.e2e
@pytest.mark.solver
def test_the_two_backends_produce_the_same_tables(linopy_results, poi_results):
    """A table or row count moving is a result change, not a rounding difference."""
    differences = [d for d in compare(linopy_results, poi_results, 'linopy', 'poi',
                                      RELATIVE_TOLERANCE)
                   if 'rows in' in d or 'tables only in' in d]
    assert not differences, 'structure differs between backends:\n  ' + '\n  '.join(differences)


@not_on_windows
@known_divergence
@pytest.mark.e2e
@pytest.mark.solver
def test_the_two_backends_produce_the_same_numbers(linopy_results, poi_results):
    """The substance: every numeric column's total, minimum and maximum."""
    differences = compare(linopy_results, poi_results, 'linopy', 'poi', RELATIVE_TOLERANCE)
    assert not differences, (
        f'{len(differences)} differences between backends:\n  ' + '\n  '.join(differences[:40]))
