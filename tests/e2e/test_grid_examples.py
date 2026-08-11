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
    here would test a configuration nobody ships. `record_backends` is not an override; it is the
    receipt, and `test_the_grid_example_needs_no_commercial_licence` says why one is needed.
    """
    example, scenario_name, creator_method = request.param
    base = tmp_path_factory.mktemp('e2e_grid')
    record = base / 'backends.json'
    try:
        fingerprint = run_example(base, REPO_ROOT / 'examples' / example, scenario_name,
                                  creator_method=creator_method, record_backends=record)
        yield scenario_name, fingerprint, base / 'results' / scenario_name, record
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.mark.e2e
def test_the_grid_example_runs_to_completion(grid_run):
    """Creator and Executor both complete. `run_example` asserts on RUN_OK and raises otherwise."""
    scenario_name, fingerprint, _, _ = grid_run

    assert fingerprint, f'{scenario_name} produced no result tables at all'


@pytest.mark.e2e
def test_the_grid_example_needs_no_commercial_licence(grid_run):
    """The examples run on HiGHS, checked rather than inherited from whatever the machine has.

    Both were switched from Gurobi to HiGHS so that anyone can run them, and nothing pinned it: a
    reviewer reverted both `agents.yaml` files to `solver: gurobi` and this file still passed, in
    85 s, on a machine with a licence. The property held only on machines incapable of breaking
    it, which is the wrong way round.
    """
    import json

    scenario_name, _, _, record = grid_run

    assert record.exists(), (
        f'{scenario_name} completed but wrote no backend record, so what solved it is unknown')
    used = {tuple(pair) for pair in json.loads(record.read_text(encoding='utf-8'))}

    assert used == {('poi', 'highs')}, (
        f'{scenario_name} was solved by {sorted(used)} rather than poi + highs as its config '
        f'specifies, so the example would need a commercial licence')


@pytest.mark.e2e
def test_the_grid_stage_produced_power_flow_results(grid_run):
    """The power flow ran and wrote results for every timestep.

    Without this the test above would pass on a run whose grid was silently inactive, which is
    exactly the shape of the gap #205 exists to close: `create_simple_scenario` finishes happily
    and calculates no grid whatsoever.
    """
    import pandas as pd

    scenario_name, _, results, _ = grid_run
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

    scenario_name, _, results, _ = grid_run
    grid_file = next((results / 'grids' / 'electricity').glob('*.xlsx'))
    net = pp.from_excel(str(grid_file))

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


@pytest.mark.e2e
def test_every_agent_in_the_scenario_reached_the_network(grid_run):
    """Every agent the scenario created is present in the saved network.

    The element-side check above cannot see this, and a reviewer proved it: the `topology` branch
    ends with `self.grid.load.dropna(subset=['id_agent'])`, so an element that lost its agent is
    *deleted* rather than left unassigned. Removing one agent's plants entirely took that example
    from 10 loads and 4 agents to 7 loads and 3 agents, and every assertion above still passed —
    a whole participant missing from the feeder, silently, which is the exact harm they claim to
    guard. Counting elements cannot fix it either; the count that would have caught it is the one
    nobody knows until after the run.

    So this asks the question from the other end: the agents are enumerated from the results tree,
    which is what the Executor actually simulated, and every one of them must own something in the
    network. Both shipped grid examples give every agent an inflexible load, so "every agent"
    is the right bar -- an agent owning nothing electrical would legitimately be absent, and no
    shipped example has one.
    """
    import pandapower as pp

    scenario_name, _, results, _ = grid_run
    grid_file = next((results / 'grids' / 'electricity').glob('*.xlsx'))
    net = pp.from_excel(str(grid_file))

    simulated = {path.name
                 for type_dir in (results / 'agents').iterdir() if type_dir.is_dir()
                 for path in type_dir.iterdir() if path.is_dir()}
    in_network = set(net.load.get('id_agent', [])) | set(net.sgen.get('id_agent', []))

    assert simulated, f'{scenario_name}: no agents found under {results / "agents"}'
    missing = sorted(simulated - in_network)
    assert not missing, (
        f'{scenario_name}: {len(missing)} of {len(simulated)} simulated agents own nothing in the '
        f'saved network, so the power flow ran on a feeder missing them: {missing}')
