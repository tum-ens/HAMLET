"""End to end — the `ctsp_industry` fixture, and the #206 read-back that now rides on it.

**Why this scenario exists.** Until it did, `ctsp` and `industry` had no coverage of any kind:
no `agents.yaml` outside `config_templates/` declared either type, and no test imported either
class (#213). The Executor's `Ctsp` and `Industry` carry a `# TODO: Not yet tested and implemented`
comment; this file is what settles which half of that sentence was true. They are `AgentBase`
subclasses with no behavioural overrides, and **they run** — see
`test_both_agent_types_produced_results`.

**What this file covers is the Executor half, and only that.** The scenario is built with
`new_scenario_from_files`, which reads `agents.xlsx` through `Agents.create_agents_from_file` —
and that method never consults `Agents.types`, the registry holding the *Creator's* per-type
classes (`agents.py:139`, used only at `:208` and `:267`). Traced on a real run: the classes
instantiated are `executor.Ctsp` and `executor.Industry` and **no Creator class at all**. So the
~1900 lines of `creator/agents/{ctsp,industry}.py` are not reached from here;
`tests/integration/creator/test_ctsp_industry_creator.py` is what reaches them, by building the
same config through `new_scenario_from_configs`. Neither file substitutes for the other, and
believing this one covers the Creator is the easy mistake.

**One run, two jobs, and that is the point of the fixture.** The run is the same either way, so
paying for it twice would be waste:

1. *#206* — the scenario is built with `new_scenario_from_files`, so the Creator reads
   `agents.xlsx` and never looks at `agents.yaml`. A backend switch that edits only the YAML is
   accepted, satisfies its own `assert switched`, and evaporates. Asking this scenario for a
   backend it does not ship, then reading back what actually solved, is what catches that.
2. *ctsp / industry* — the same run is the only place either agent type is **simulated**.

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
    # The premise, before spending 30 s on a run whose result would prove nothing without it.
    check_the_premise()
    base = tmp_path_factory.mktemp('ctsp_industry')
    record = base / 'backends.json'

    fingerprint = run_example(base, CONFIG_ROOT, SCENARIO, framework=REQUESTED_FRAMEWORK,
                              solver=REQUESTED_SOLVER, record_backends=record,
                              creator_method='new_scenario_from_files')

    used = {tuple(pair) for pair in json.loads(record.read_text(encoding='utf-8'))}
    return used, fingerprint


def check_the_premise():
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

    # Per sheet, never workbook-wide. One sheet must not vouch for the other -- that is #206's
    # shape one level down, and it is live here precisely because this fixture ships two sheets.
    # Read workbook-wide, renaming the `industry` sheet's framework column left this test green
    # and only the integration module noticed; demonstrated.
    per_sheet = {}
    with pd.ExcelFile(CONFIG_ROOT / SCENARIO / 'agents.xlsx') as book:
        for sheet in book.sheet_names:
            frame = book.parse(sheet, index_col=0)
            per_sheet[sheet] = {value for column in frame.columns
                                if str(column).rsplit('/', 1)[-1] == 'framework'
                                for value in frame[column].dropna()}
    sheets = sorted(per_sheet)

    blind = sorted(sheet for sheet, values in per_sheet.items() if not values)
    assert not blind, (
        f'{SCENARIO}/agents.xlsx sheets {blind} carry no framework column at all, so the switch '
        f'would be silently lost for those agents')
    wrong = {sheet: sorted(values) for sheet, values in per_sheet.items()
             if values != {SHIPPED_FRAMEWORK}}
    assert not wrong, (
        f'{SCENARIO}/agents.xlsx ships framework={wrong}, not {SHIPPED_FRAMEWORK!r} throughout. '
        f'This file asks it for {REQUESTED_FRAMEWORK!r}; asking for what it already ships proves '
        f'nothing')
    assert SHIPPED_FRAMEWORK != REQUESTED_FRAMEWORK
    # Two sheets, because the per-sheet half of the backend switch (#206 one level down) has no
    # other real fixture -- `grid_golden` has one sheet, and the two integration tests that need
    # a second synthesise it (test_shipped_configs_agree_with_their_workbooks,
    # test_scenario_run_backend_switch).
    assert sheets == ['ctsp', 'industry'], sheets


def test_the_fixture_still_ships_a_backend_worth_switching_away_from():
    """The premise as a named test, so it runs in the fast tier and fails there by name.

    It is *also* called from the `run` fixture below, which is what stops `pytest -m e2e` -- the CI
    job, and what a developer reaches for locally -- from running the #206 read-back with its
    premise unchecked. A premise that only holds in a tier nobody runs before pushing is the same
    fail-open shape as the defect this file is about.
    """
    check_the_premise()


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
    two is that the two classes are near-identical -- 48 changed lines across 13 hunks, per
    #213 -- and have drifted; a check that one of them vouches for the other reproduces the
    problem it is meant to detect.
    """
    _, fingerprint = run

    # `AGENT_TABLES` is asserted non-empty first: `missing` is a comprehension over it, so an
    # empty tuple makes the assertion below vacuous and this test its only consumer. Demonstrated.
    assert AGENT_TABLES, 'AGENT_TABLES is empty, so the check below asserts nothing'

    # Empty is not present. `scenario_run.fingerprint` creates a key per *file*, so a table that
    # exists with zero rows satisfies a membership test -- and zeroing meters, socs and timeseries
    # left this green when the check was `not in fingerprint`. Demonstrated.
    missing = [f'agents/{agent_type}/{table}'
               for agent_type in ('ctsp', 'industry') for table in AGENT_TABLES
               if fingerprint.get(f'agents/{agent_type}/{table}', {'rows': 0})['rows'] == 0]

    assert not missing, (
        f'the run completed but {missing} are absent or empty -- the agent type did not execute, '
        f'or executed and produced nothing. It produced: '
        f'{ {kind: entry["rows"] for kind, entry in sorted(fingerprint.items()) if kind.startswith("agents/")} }')


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
