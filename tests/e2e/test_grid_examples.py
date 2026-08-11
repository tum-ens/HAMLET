"""End-to-end — the two grid-enabled examples, which is to say: does the grid stage run at all.

Neither of these ran on `develop` until #205. `create_scenario_with_grid` builds its network from
a complete grid file and `create_scenario_with_topology` from a topology plus per-bus agent
assignments, so between them they cover both branches of `GridDB.register_grid` and both grid-file
conventions — and unlike `create_simple_scenario`, which sets `electricity.active: False`, they
actually solve a power flow.

Marked `e2e` and deselected by default:

    python -m pytest tests -m e2e

A smoke test, not a golden master: it asserts the run completes and that the grid stage produced
results, not that specific numbers come back. The unit-level behaviour of registration is pinned
in `tests/integration/executor/test_grid_registration.py`, which is fast; this is the layer that
would notice the two being wired together wrongly.
"""
import shutil
from pathlib import Path

import pytest

from tests.scenario_run import run_example

REPO_ROOT = Path(__file__).resolve().parents[2]

# (example folder, scenario name, the Creator entry point that example's notebook calls)
GRID_EXAMPLES = [
    pytest.param(('create_scenario_with_grid', 'scenario_with_grid', 'new_scenario_from_grids'),
                 id='from_grid_file'),
    pytest.param(('create_scenario_with_topology', 'scenario_with_topology',
                  'new_scenario_from_files'), id='from_topology'),
]


@pytest.fixture(scope='module', params=GRID_EXAMPLES)
def grid_run(request, tmp_path_factory):
    """Run one grid example exactly as shipped, in a temp copy of its config tree.

    Nothing is overridden — no framework, no solver, no config edit. That is the point: the
    acceptance criterion for #205 is that the examples run as a user finds them, and an override
    here would test a configuration nobody ships.
    """
    example, scenario_name, creator_method = request.param
    base = tmp_path_factory.mktemp('e2e_grid')
    try:
        fingerprint = run_example(base, REPO_ROOT / 'examples' / example, scenario_name,
                                  creator_method=creator_method)
        yield scenario_name, fingerprint, base / 'results' / scenario_name
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.mark.e2e
def test_the_grid_example_runs_to_completion(grid_run):
    """Creator and Executor both complete. `run_example` asserts on RUN_OK and raises otherwise."""
    scenario_name, fingerprint, _ = grid_run

    assert fingerprint, f'{scenario_name} produced no result tables at all'


@pytest.mark.e2e
def test_the_grid_stage_produced_power_flow_results(grid_run):
    """The power flow ran and wrote results for every timestep.

    Without this the test above would pass on a run whose grid was silently inactive, which is
    exactly the shape of the gap #205 exists to close: `create_simple_scenario` finishes happily
    and calculates no grid whatsoever.
    """
    import pandas as pd

    scenario_name, _, results = grid_run
    grid_results = results / 'grids' / 'electricity'
    written = sorted(path.name for path in grid_results.glob('res_*.csv'))

    assert 'res_bus.csv' in written and 'res_line.csv' in written, (
        f'{scenario_name} wrote no power flow results, so the grid stage did not run: '
        f'found {written}')

    # A results file that exists but is empty would satisfy the check above.
    for name in ('res_bus.csv', 'res_line.csv'):
        frame = pd.read_csv(grid_results / name)
        assert not frame.empty, f'{name} is empty, so no power flow was solved'


@pytest.mark.e2e
def test_every_grid_element_belongs_to_an_agent(grid_run):
    """Registration assigned every load and sgen, rather than leaving some unowned.

    This is the assertion the old failure modes traded away: both #201's `TypeError` and #205's
    `KeyError` sat where the alternative was to skip the unmatched element and solve the power
    flow on a feeder missing one of its participants. An unassigned element would understate the
    loading, silently.
    """
    import pandapower as pp

    scenario_name, _, results = grid_run
    grid_file = next((results / 'grids' / 'electricity').glob('*.xlsx'))
    net = pp.from_excel(str(grid_file))

    # Stated as a count and not as `if table.empty: continue`. A registration that produced no
    # loads at all, or that lost the id column in the round-trip, would otherwise satisfy an
    # "every element has an agent" check by having no elements.
    assert len(net.load) >= 4, f'{scenario_name}: only {len(net.load)} loads in the saved network'

    for table_name in ('load', 'sgen'):
        table = getattr(net, table_name)
        if table.empty:  # a scenario need not contain any generation
            continue
        assert 'id_agent' in table.columns, (
            f'{scenario_name}: the saved network has no id_agent column on {table_name}, so '
            f'nothing records which agent each element belongs to')
        unassigned = table[table['id_agent'].isna() | (table['id_agent'] == 0)]
        assert unassigned.empty, (
            f'{scenario_name}: {len(unassigned)} of {len(table)} {table_name} elements have no '
            f'agent after registration (rows {list(unassigned.index)})')
