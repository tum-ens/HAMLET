"""End to end — the `ctsp_industry` fixture, and the #206 read-back that now rides on it.

**Why this scenario exists.** Until it did, `ctsp` and `industry` had no coverage of any kind:
no `agents.yaml` outside `config_templates/` declared either type, and no test imported either
class, so ~1900 lines of Creator and both Executor agent classes were never executed (#213). The
Executor's `Ctsp` and `Industry` carry a `# TODO: Not yet tested and implemented` comment; this
file is what settles which half of that sentence was true. They are `AgentBase` subclasses with no
overrides, and **they run** — see `test_both_agent_types_produced_results`.

**One run, two jobs, and that is the point of the fixture.** The run is the same either way, so
paying for it twice would be waste:

1. *#206* — the scenario is built with `new_scenario_from_files`, so the Creator reads
   `agents.xlsx` and never looks at `agents.yaml`. A backend switch that edits only the YAML is
   accepted, satisfies its own `assert switched`, and evaporates. Asking this scenario for a
   backend it does not ship, then reading back what actually solved, is what catches that.
2. *ctsp / industry* — the same run is the only place either agent type is executed at all.

**The switch runs from `linopy` to `poi`, and that direction is deliberate.** `grid_golden` — which
carried this test before — ships `poi`, so asking it for something it does not ship meant asking
for `linopy`, and linopy builds a model per solve. This fixture ships `linopy` instead, so the
value it does *not* ship is `poi`, the fast one and the project default.

Measured through `run_example` on the development laptop, same harness both sides:
**26–36 s (4 runs) against 232–272 s (2 runs)** for the `grid_golden` call this replaces. Quote the
band rather than a number — this runner's spread on identical work is wide. Shipping the
non-default backend here is therefore load-bearing, not an oversight: see the header of
`scenarios/ctsp_industry/agents.yaml`, and do not "correct" it to `poi` without moving this test's
request the other way, or the saving goes with it.

**The fixture declares no EV, and that is not tidiness.** Raising `ev` share turns this file red:
the Creator writes `NaN` into every nested `charging_scheme` parameter for both classes
(`_add_info_indexed` does not descend into nested config the way `_add_info_simple` does), the
`config_templates` `ctsp` block names a forecast model that does not exist, and POI's
`__constraint_cs_full` passes a Series where a scalar is required. Each is filed separately;
`tests/README.md` names them. Until they are fixed an EV here would cover nothing and block
everything.
"""
import json

import pytest

from tests.backend_matrix import require
from tests.scenario_run import REPO_ROOT, run_example

CONFIG_ROOT = REPO_ROOT / 'tests' / 'e2e' / 'scenarios'
SCENARIO = 'ctsp_industry'

#: What the fixture's `agents.xlsx` pins. Asserted rather than assumed by
#: `test_the_fixture_still_ships_a_backend_worth_switching_away_from`.
SHIPPED_FRAMEWORK = 'linopy'

#: What this file asks for instead. Needs no licence, so the whole file runs everywhere.
REQUESTED_FRAMEWORK = 'poi'
REQUESTED_SOLVER = 'highs'

#: The result tables every agent writes. Named, so "the agent produced results" cannot be
#: satisfied by a folder that exists and is empty.
AGENT_TABLES = ('meters.ft', 'setpoints.ft', 'socs.ft', 'timeseries.ft', 'forecasts.ft')


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    """Run the fixture once, switched to `poi`, and hand back the receipt and the fingerprint."""
    require(REQUESTED_FRAMEWORK, REQUESTED_SOLVER)
    base = tmp_path_factory.mktemp('ctsp_industry')
    record = base / 'backends.json'

    fingerprint = run_example(base, CONFIG_ROOT, SCENARIO, framework=REQUESTED_FRAMEWORK,
                              solver=REQUESTED_SOLVER, record_backends=record,
                              creator_method='new_scenario_from_files')

    used = {tuple(pair) for pair in json.loads(record.read_text(encoding='utf-8'))}
    return used, fingerprint


def test_the_fixture_still_ships_a_backend_worth_switching_away_from():
    """The premise, checked against the files rather than assumed.

    If the fixture ever starts shipping `poi`, every assertion below still passes while proving
    nothing: the run would use `poi` whether the switch worked or not. Stated as its own test so
    that a fixture change fails *here*, naming the cause, instead of quietly hollowing out the
    #206 coverage — which is the precise failure mode #206 was filed about.

    Reads the workbook the way `create_agents_from_file` does, because the workbook is the file
    this scenario is built from. The YAML is checked too: it must agree (#214), and
    `tests/integration/test_shipped_configs_agree_with_their_workbooks.py` is what enforces that.
    """
    import pandas as pd

    values = []
    with pd.ExcelFile(CONFIG_ROOT / SCENARIO / 'agents.xlsx') as book:
        sheets = list(book.sheet_names)
        for sheet in sheets:
            frame = book.parse(sheet, index_col=0)
            for column in frame.columns:
                if str(column).rsplit('/', 1)[-1] == 'framework':
                    values.extend(frame[column].dropna().tolist())

    assert values, f'{SCENARIO}/agents.xlsx has no framework column at all'
    assert set(values) == {SHIPPED_FRAMEWORK}, (
        f'{SCENARIO}/agents.xlsx ships framework={sorted(set(values))}, not {SHIPPED_FRAMEWORK!r}. '
        f'This file asks it for {REQUESTED_FRAMEWORK!r}; asking for what it already ships proves '
        f'nothing')
    assert SHIPPED_FRAMEWORK != REQUESTED_FRAMEWORK
    # Two sheets, because the per-sheet half of the backend switch (#206 one level down) has no
    # other real fixture -- `grid_golden` has one sheet and the unit tests synthesise a second.
    assert sorted(sheets) == ['ctsp', 'industry'], sheets


@pytest.mark.e2e
@pytest.mark.solver
def test_the_requested_backend_is_what_solved(run):
    """#206 — the request reached the file the Creator actually reads.

    **The assertion is on what solved, not on what was asked for.** `used` comes from
    `scenario_run.BACKEND_PROBE`, which wraps `linopy.Model.solve` and both POI `create_model`
    functions inside the run's own interpreter, so it reports the backend that really built and
    solved the agents' models. Asserting that `agents.xlsx` now says `poi` would only re-check the
    edit; this checks the consequence.

    Deliberately redundant with `run_example`'s own `assert_backend_honoured`, which fires first.
    Keeping it is what makes this test independent of that guard: with the workbook step reverted
    *and* the compulsory receipt disabled, this assertion is the one that fails.
    """
    used, _ = run

    assert used == {(REQUESTED_FRAMEWORK, REQUESTED_SOLVER)}, (
        f'asked {SCENARIO} for {REQUESTED_FRAMEWORK} + {REQUESTED_SOLVER}, but the run actually '
        f'used {sorted(used) or "no modelling backend at all"}. This scenario is built from '
        f'agents.xlsx, so a switch that only edits agents.yaml is lost -- see #206')


@pytest.mark.e2e
@pytest.mark.solver
def test_both_agent_types_produced_results(run):
    """`ctsp` and `industry` execute — the answer to the Executor's "not yet tested" TODO.

    Both types are asserted **separately and by name**. A count, or "at least one agent wrote
    something", would be satisfied by either type alone, and the whole reason this fixture declares
    two is that the two classes are 92 % identical and have drifted (#213); a check that one of
    them vouches for the other reproduces the problem it is meant to detect.
    """
    _, fingerprint = run

    missing = [f'agents/{agent_type}/{table}'
               for agent_type in ('ctsp', 'industry') for table in AGENT_TABLES
               if f'agents/{agent_type}/{table}' not in fingerprint]

    assert not missing, (
        f'the run completed but wrote no {missing} -- the agent type did not execute. '
        f'It produced: {sorted(kind for kind in fingerprint if kind.startswith("agents/"))}')


@pytest.mark.e2e
@pytest.mark.solver
def test_both_agent_types_wrote_a_row_for_every_timestep(run):
    """Existing is not running. The tables must hold the run's 24 timesteps, not be empty.

    `setpoints` is one row per timestep per agent and `forecasts` likewise, so with one agent of
    each type both are exactly 24. Pinning the number rather than `> 0` is what makes this fail if
    the scenario is silently shortened — a fixture that stopped after one step would still write
    every table named above.
    """
    _, fingerprint = run

    rows = {f'agents/{agent_type}/{table}': fingerprint[f'agents/{agent_type}/{table}']['rows']
            for agent_type in ('ctsp', 'industry') for table in ('setpoints.ft', 'forecasts.ft')}

    assert set(rows.values()) == {24}, rows


@pytest.mark.e2e
@pytest.mark.solver
def test_the_agents_traded(run):
    """The run reached the market, not only the controllers.

    Without this the file would pass on a run in which every agent solved its own model and no
    transaction was ever cleared — which is a plausible way for a new agent type to "work" while
    being invisible to the rest of the simulation.
    """
    _, fingerprint = run

    transactions = [kind for kind in fingerprint if kind.endswith('market_transactions.ft')]
    assert transactions, f'the run wrote no market transactions; it produced {sorted(fingerprint)}'
    assert sum(fingerprint[kind]['rows'] for kind in transactions) > 0, (
        'market_transactions exists but is empty, so no agent traded')
