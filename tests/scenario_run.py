"""Running a shipped example and reducing its results to a comparable fingerprint.

Extracted from `tests/e2e/test_golden_master.py` so that the golden master and the backend
equivalence tests reduce results the *same* way. Two different fingerprints would make the two
tests incomparable, and the whole value of the equivalence test is that its linopy arm reproduces
the golden reference exactly.

Reproducibility rests on seeding `random` and `numpy.random` and pinning `PYTHONHASHSEED`: the
Creator draws agent ids, plant ownership and sizings from all three.
"""
import os
import re
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
{probe}
Creator(path=r"{config_dir}").{creator_method}()
Executor(r"{scenarios}/{name}", num_workers=1).run()
print("RUN_OK")
"""

# Records every (framework, solver) pair that actually built or solved a model during the run, so
# a caller can assert the run used what it asked for instead of trusting that a config edit took.
#
# This is not defensiveness. In !212 the `framework:` switch below matched a literal shipped value
# and silently became a no-op when the default flipped, which would have made both arms of a
# backend comparison run the same backend and agree with each other. A config edit is a *request*;
# this is the receipt. `num_workers=1` is what makes it work: every solve happens in this process.
#
# The two POI modules are patched by name rather than `poi_solver.create_model`, because each does
# `from ... import create_model`, which binds the function into its own namespace -- rebinding it
# on `poi_solver` afterwards would patch nothing, record an empty set, and fail open.
BACKEND_PROBE = '''
import atexit, json
import linopy
from hamlet.executor.utilities.controller.fbc.mpc.poi import mpc_poi
from hamlet.executor.utilities.controller.rtc.optim.poi import optim_poi

_used = set()

for _module in (mpc_poi, optim_poi):
    _original_create = _module.create_model

    def _create_model(solver, _original_create=_original_create):
        _used.add(('poi', solver))
        return _original_create(solver)

    _module.create_model = _create_model

_original_solve = linopy.Model.solve


def _solve(self, *args, **kwargs):
    _used.add(('linopy', kwargs.get('solver_name') or (args[0] if args else None)))
    return _original_solve(self, *args, **kwargs)


linopy.Model.solve = _solve

atexit.register(lambda: open(r"{record}", "w", encoding="utf-8").write(json.dumps(sorted(_used))))
'''


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


def run_example(base, example_dir, scenario_name, framework=None, solver=None, edits=(),
                config_edits=None, record_backends=None,
                creator_method='new_scenario_from_configs'):
    """Run one example end to end under `base`, and return the fingerprint of its results.

    `creator_method` is the Creator entry point the example's own notebook calls — the three
    shipped ones differ (`new_scenario_from_configs` reads the YAML, `new_scenario_from_files`
    reads `agents.xlsx`, `new_scenario_from_grids` derives the agents from the grid file), and
    which one is used decides whether the agent ids are drawn fresh or come from a file. That
    matters for the `topology` grid method, whose topology file names agents by id.

    `framework` switches every `framework:` key to the named backend and `solver` every `solver:`
    key to the named solver, whatever the config ships; None leaves either alone. Both match the
    *key* rather than the shipped value on purpose -- the framework switch used to look for the
    literal `framework: linopy`, which silently became a no-op the moment the default flipped to
    `poi`, and a no-op here means both arms of a backend comparison run the same backend and
    agree. `edits` is a sequence of (old, new) string replacements applied to `agents.yaml`, and
    `config_edits` is a {filename: [(old, new), ...]} mapping for any other config file --
    `grids.yaml`, say, to switch a grid restriction on.

    Every replacement must match at least once. That is deliberate: a renamed config key then
    fails the test loudly, instead of the test quietly running an unmodified scenario and passing
    for the wrong reason.

    `record_backends` is a path to write the JSON list of (framework, solver) pairs the run
    actually used. Matching the key is what stops the *request* being lost; this is what proves it
    was honoured. See `BACKEND_PROBE`.
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
        agents_text, switched = re.subn(r'^(\s*)framework: *\w+', rf'\g<1>framework: {framework}',
                                        agents_text, flags=re.MULTILINE)
        assert switched, 'no framework key to switch'
    if solver is not None:
        agents_text, switched = re.subn(r'^(\s*)solver: *\w+', rf'\g<1>solver: {solver}',
                                        agents_text, flags=re.MULTILINE)
        assert switched, 'no solver key to switch'
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

    probe = '' if record_backends is None else BACKEND_PROBE.format(record=record_backends)
    script = RUNNER.format(config_dir=config.as_posix(), seed=SEED, scenarios=scenarios,
                           name=scenario_name, probe=probe, creator_method=creator_method)
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
