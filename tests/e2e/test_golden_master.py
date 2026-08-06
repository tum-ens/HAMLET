"""Golden master — the shipped example, compared against committed reference numbers.

Every other test here pins a property someone thought to check. This one pins the numbers
themselves, so a change that moves results has to be acknowledged rather than noticed. Of the
defects found while assembling this branch, most were caught by review or by measurement rather
than by a test failing; this is the test that would have caught them.

    python -m pytest tests -m golden

**When it fails**, the diff tells you which table and column moved. Decide whether that is the
change you meant to make. If it is, regenerate the reference and commit it *with* the change,
so the review sees the numbers move:

    HAMLET_UPDATE_GOLDEN=1 python -m pytest tests -m golden

Reproducibility rests on seeding `random` and `numpy.random` and pinning `PYTHONHASHSEED`. The
Creator draws agent ids, plant ownership and sizings from all three. Verified: two seeded runs
produce byte-identical scenarios and identical results.

The reference records per-table row counts and, for every numeric column, the sum, minimum and
maximum. Agent ids are random-but-seeded, so tables are grouped by kind rather than by agent --
that keeps the reference readable and stable against an id-scheme change, while still moving the
moment the physics does.
"""
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / 'examples' / 'create_simple_scenario'
SCENARIO_NAME = 'simple_scenario'
REFERENCE = Path(__file__).parent / 'golden' / f'{SCENARIO_NAME}.json'

SEED = 20260804
# Solver output is bit-stable on a fixed platform, but HiGHS and polars versions move; this is
# loose enough to survive that and far tighter than any real modelling change.
RELATIVE_TOLERANCE = 1e-6

RUNNER = """
import os, random, sys
import numpy as np
sys.path.insert(0, r"{repo}")
random.seed({seed})
np.random.seed({seed})
from hamlet import Creator, Executor
Creator(path=r"{config_dir}").new_scenario_from_configs()
Executor(r"{scenarios}/{name}", num_workers=1).run()
print("RUN_OK")
"""


def table_kind(path, root):
    """The table's identity, with the random agent id replaced by its type."""
    parts = Path(path).relative_to(root).parts
    # agents/<type>/<random id>/<table>.ft  ->  agents/<type>/<table>.ft
    if parts and parts[0] == 'agents' and len(parts) == 4:
        return f'agents/{parts[1]}/{parts[3]}'
    return '/'.join(parts)


def fingerprint(results_root):
    """Row counts and per-column statistics, grouped by table kind."""
    import polars as pl

    grouped = defaultdict(lambda: {'files': 0, 'rows': 0, 'columns': {}})
    for path in sorted(Path(results_root).rglob('*.ft')):
        frame = pl.read_ipc(path, memory_map=False)
        entry = grouped[table_kind(path, results_root)]
        entry['files'] += 1
        entry['rows'] += len(frame)
        for column, dtype in zip(frame.columns, frame.dtypes):
            if not dtype.is_numeric() or frame[column].null_count() == len(frame):
                continue
            stats = entry['columns'].setdefault(column, {'sum': 0.0, 'min': None, 'max': None})
            stats['sum'] += float(frame[column].sum() or 0)
            low, high = frame[column].min(), frame[column].max()
            if low is not None:
                stats['min'] = float(low) if stats['min'] is None else min(stats['min'], float(low))
                stats['max'] = float(high) if stats['max'] is None else max(stats['max'], float(high))

    return {kind: entry for kind, entry in sorted(grouped.items())}


@pytest.fixture(scope='module')
def actual(tmp_path_factory):
    """Run the example once, seeded, against a temp copy of the config."""
    import shutil

    base = tmp_path_factory.mktemp('golden')
    scenarios, results = base / 'scenarios', base / 'results'
    config = base / SCENARIO_NAME
    shutil.copytree(EXAMPLE / SCENARIO_NAME, config)
    scenarios.mkdir()
    results.mkdir()

    setup = config / 'setup.yaml'
    text = setup.read_text(encoding='utf-8')
    for old, new in (('input: ../../input_data', f'input: {(REPO_ROOT / "input_data").as_posix()}'),
                     ('scenarios: ../../scenarios', f'scenarios: {scenarios.as_posix()}'),
                     ('results: ../../results', f'results: {results.as_posix()}')):
        assert old in text, f'{old!r} not found in setup.yaml'
        text = text.replace(old, new)
    setup.write_text(text, encoding='utf-8')

    script = RUNNER.format(repo=REPO_ROOT, config_dir=config.as_posix(), seed=SEED,
                           scenarios=scenarios, name=SCENARIO_NAME)
    completed = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=3600,
        env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONIOENCODING': 'utf-8',
             'PYTHONHASHSEED': '0'})
    assert 'RUN_OK' in completed.stdout, completed.stderr[-4000:]

    try:
        yield fingerprint(results / SCENARIO_NAME)
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope='module')
def expected(actual):
    """The committed reference, regenerated in place when explicitly asked for."""
    if os.environ.get('HAMLET_UPDATE_GOLDEN'):
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE.write_text(json.dumps(actual, indent=2, sort_keys=True) + '\n',
                             encoding='utf-8')
        pytest.skip(f'reference regenerated at {REFERENCE.relative_to(REPO_ROOT)}; '
                    f'review the diff and commit it with the change that caused it')

    assert REFERENCE.exists(), (
        f'no golden reference at {REFERENCE.relative_to(REPO_ROOT)}. Create one with '
        f'HAMLET_UPDATE_GOLDEN=1 python -m pytest tests -m golden')

    return json.loads(REFERENCE.read_text(encoding='utf-8'))


@pytest.mark.golden
def test_the_same_tables_are_produced(actual, expected):
    """A table appearing or disappearing is a result change like any other."""
    assert sorted(actual) == sorted(expected)


@pytest.mark.golden
def test_row_counts_match(actual, expected):
    """Catches trades appearing or vanishing, which several defects here did."""
    differences = {kind: (entry['rows'], expected[kind]['rows'])
                   for kind, entry in actual.items()
                   if kind in expected and entry['rows'] != expected[kind]['rows']}

    assert not differences, f'row counts moved (actual, expected): {differences}'


@pytest.mark.golden
def test_column_statistics_match(actual, expected):
    """The substance: every numeric column's total, minimum and maximum.

    Reported all at once rather than failing on the first, because when the model changes it is
    the shape of the difference across tables that tells you whether it was intended.
    """
    differences = []
    for kind, entry in actual.items():
        if kind not in expected:
            continue
        for column, stats in entry['columns'].items():
            reference = expected[kind]['columns'].get(column)
            if reference is None:
                differences.append(f'{kind}:{column} is new')
                continue
            for statistic, value in stats.items():
                other = reference.get(statistic)
                if value is None or other is None:
                    if value != other:
                        differences.append(f'{kind}:{column}.{statistic} {value} != {other}')
                    continue
                if abs(value - other) > RELATIVE_TOLERANCE * max(1.0, abs(other)):
                    differences.append(
                        f'{kind}:{column}.{statistic} {value:,.3f} != {other:,.3f} '
                        f'(delta {value - other:+,.3f})')

    assert not differences, (
        'the shipped example now produces different numbers:\n  '
        + '\n  '.join(differences[:40])
        + (f'\n  ... and {len(differences) - 40} more' if len(differences) > 40 else '')
        + '\n\nIf this change was intended, regenerate the reference with '
          'HAMLET_UPDATE_GOLDEN=1 and commit it alongside the change.')


@pytest.mark.golden
def test_no_column_was_dropped(actual, expected):
    """A column disappearing is easy to miss when only the ones present are compared."""
    missing = [f'{kind}:{column}'
               for kind, entry in expected.items() if kind in actual
               for column in entry['columns']
               if column not in actual[kind]['columns']]

    assert not missing, f'columns no longer produced: {missing}'
