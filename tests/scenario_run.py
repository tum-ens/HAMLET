"""Running a shipped example and reducing its results to a comparable fingerprint.

Extracted from `tests/e2e/test_golden_master.py` so that the golden master and the backend
equivalence tests reduce results the *same* way. Two different fingerprints would make the two
tests incomparable, and the whole value of the equivalence test is that its linopy arm reproduces
the golden reference exactly.

Reproducibility rests on seeding `random` and `numpy.random` and pinning `PYTHONHASHSEED`: the
Creator draws agent ids, plant ownership and sizings from all three.
"""
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260804

# Runs Creator then Executor in a subprocess. A subprocess rather than an in-process call because
# `PYTHONHASHSEED` only takes effect at interpreter start, and the Creator's file selection depends
# on it.
RUNNER = """
import os, random
import numpy as np
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


def run_example(base, example_dir, scenario_name, framework=None, edits=(), config_edits=None):
    """Run one example end to end under `base`, and return the fingerprint of its results.

    `framework` switches `framework: linopy` to another backend everywhere it appears; None leaves
    the config as shipped. `edits` is a sequence of (old, new) string replacements applied to
    `agents.yaml`, and `config_edits` is a {filename: [(old, new), ...]} mapping for any other
    config file -- `grids.yaml`, say, to switch a grid restriction on.

    Every replacement must match at least once. That is deliberate: a renamed config key then
    fails the test loudly, instead of the test quietly running an unmodified scenario and passing
    for the wrong reason.
    """
    scenarios, results = base / 'scenarios', base / 'results'
    config = base / scenario_name
    shutil.copytree(example_dir / scenario_name, config)
    scenarios.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)

    setup = config / 'setup.yaml'
    text = setup.read_text(encoding='utf-8')
    for old, new in (('input: ../../input_data', f'input: {(REPO_ROOT / "input_data").as_posix()}'),
                     ('scenarios: ../../scenarios', f'scenarios: {scenarios.as_posix()}'),
                     ('results: ../../results', f'results: {results.as_posix()}')):
        assert old in text, f'{old!r} not found in setup.yaml'
        text = text.replace(old, new)
    setup.write_text(text, encoding='utf-8')

    agents = config / 'agents.yaml'
    agents_text = agents.read_text(encoding='utf-8')
    if framework is not None:
        assert 'framework: linopy' in agents_text, 'no framework key to switch'
        agents_text = agents_text.replace('framework: linopy', f'framework: {framework}')
    for old, new in edits:
        assert old in agents_text, f'{old!r} not found in agents.yaml'
        agents_text = agents_text.replace(old, new)
    agents.write_text(agents_text, encoding='utf-8')

    for filename, replacements in (config_edits or {}).items():
        target = config / filename
        assert target.exists(), f'{filename} not found in {scenario_name}'
        content = target.read_text(encoding='utf-8')
        for old, new in replacements:
            assert old in content, f'{old!r} not found in {filename}'
            content = content.replace(old, new)
        target.write_text(content, encoding='utf-8')

    script = RUNNER.format(config_dir=config.as_posix(), seed=SEED,
                           scenarios=scenarios, name=scenario_name)
    completed = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=3600,
        env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONIOENCODING': 'utf-8',
             'PYTHONHASHSEED': '0'})
    assert 'RUN_OK' in completed.stdout, completed.stderr[-4000:]

    return fingerprint(results / scenario_name)


def compare(a, b, label_a, label_b, tolerance):
    """Differences between two fingerprints, as a list of human-readable strings.

    Empty means the two runs agree to `tolerance`. Reported all at once rather than failing on the
    first, because when a model changes it is the shape of the difference across tables that says
    whether it was intended.
    """
    differences = []

    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a:
        differences.append(f'tables only in {label_a}: {only_a}')
    if only_b:
        differences.append(f'tables only in {label_b}: {only_b}')

    for kind in sorted(set(a) & set(b)):
        if a[kind]['rows'] != b[kind]['rows']:
            differences.append(
                f"{kind}: {a[kind]['rows']} rows in {label_a}, {b[kind]['rows']} in {label_b}")
        for column, stats in a[kind]['columns'].items():
            reference = b[kind]['columns'].get(column)
            if reference is None:
                differences.append(f'{kind}:{column} is present in {label_a} only')
                continue
            for statistic, value in stats.items():
                other = reference.get(statistic)
                if value is None or other is None:
                    continue
                scale = max(abs(value), abs(other))
                if scale < 1e-9 or abs(value - other) / scale <= tolerance:
                    continue
                differences.append(
                    f'{kind}:{column}.{statistic}  {value:.6g} ({label_a}) '
                    f'vs {other:.6g} ({label_b})')

    return differences
