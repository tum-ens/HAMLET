"""End-to-end — the shipped example, Creator to Executor to Analyzer.

This is the test the README names as the installation check, and the only one that exercises
the real file layout, the forecaster, the market clearing and the grid stage together. It takes
a few minutes, so it is marked `e2e` and deselected by default:

    python -m pytest tests -m e2e

It is deliberately a smoke test rather than a golden master: it asserts that the run completes
and that the results are structurally sound, not that specific numbers are reproduced. A real
golden master needs committed reference tables and a fixed seed.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / 'examples' / 'create_simple_scenario'
SCENARIO_NAME = 'simple_scenario'

RUNNER = """
import os, sys
sys.path.insert(0, r"{repo}")
os.chdir(r"{example}")
from hamlet import Creator, Executor, Analyzer
Creator(path="./{name}").new_scenario_from_configs()
Executor(r"{scenarios}/{name}", num_workers=1).run()
sim = Analyzer({{"{name}": r"{results}/{name}"}})
sim.agents.plot_all_meters_data(save_path=None)
sim.markets.plot_total_balancing()
print("E2E_OK")
"""


@pytest.fixture(scope='module')
def run_dirs(tmp_path_factory):
    """Run the example once, with scenarios and results directed into a temp tree."""
    base = tmp_path_factory.mktemp('e2e')
    scenarios, results = base / 'scenarios', base / 'results'
    scenarios.mkdir()
    results.mkdir()

    # The example's setup.yaml puts scenarios and results two levels above the config folder;
    # point them at the temp tree instead so the repo is never written to.
    setup = EXAMPLE / SCENARIO_NAME / 'setup.yaml'
    original = setup.read_text(encoding='utf-8')
    patched = (original
               .replace('scenarios: ../../scenarios', f'scenarios: {scenarios.as_posix()}')
               .replace('results: ../../results', f'results: {results.as_posix()}'))
    generated_xlsx = EXAMPLE / SCENARIO_NAME / 'agents.xlsx'

    setup.write_text(patched, encoding='utf-8')
    try:
        script = RUNNER.format(repo=REPO_ROOT, example=EXAMPLE, name=SCENARIO_NAME,
                               scenarios=scenarios, results=results)
        completed = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                   text=True, encoding='utf-8', errors='replace', timeout=3600,
                                   env={**__import__('os').environ, 'MPLBACKEND': 'Agg',
                                        'PYTHONIOENCODING': 'utf-8'})
        yield completed, results / SCENARIO_NAME
    finally:
        setup.write_text(original, encoding='utf-8')
        generated_xlsx.unlink(missing_ok=True)  # Creator output, not a source file
        shutil.rmtree(base, ignore_errors=True)


@pytest.mark.e2e
def test_the_example_runs_to_completion(run_dirs):
    """Creator, Executor and Analyzer all complete without raising."""
    completed, _ = run_dirs

    assert 'E2E_OK' in completed.stdout, completed.stderr[-4000:]
    assert completed.returncode == 0


@pytest.mark.e2e
def test_market_transactions_are_produced(run_dirs):
    """The run must actually trade, not just finish."""
    import polars as pl

    _, results = run_dirs
    transactions = list(results.rglob('market_transactions.ft'))

    assert transactions, 'no market transactions were written'
    frame = pl.read_ipc(transactions[0], memory_map=False)
    assert len(frame) > 0


@pytest.mark.e2e
def test_grid_fees_and_levies_are_charged_on_consumption(run_dirs):
    """The in/out convention, verified on real output rather than in a fixture.

    Grid fees and levies are owed on what an agent draws from the grid. If the retailer columns
    are read the wrong way round these land on feed-in instead, which is the defect this
    convention has repeatedly produced.
    """
    import polars as pl
    import hamlet.constants as c

    _, results = run_dirs
    frame = pl.read_ipc(list(results.rglob('market_transactions.ft'))[0], memory_map=False)

    for trade_type in (c.TT_GRID, c.TT_LEVIES):
        rows = frame.filter(pl.col(c.TC_TYPE_TRANSACTION) == trade_type)
        if rows.is_empty():
            pytest.skip(f'no {trade_type} transactions in this run')
        # Charges are levied on the consumption side, so the per-unit import price must be set
        assert rows[c.TC_PRICE_PU_IN].max() > 0, f'{trade_type} is not charged on consumption'
