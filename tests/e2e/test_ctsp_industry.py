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

**The fixture declares an EV on both sheets, and that took three fixes.** It shipped `ev.share: 0`
until #218, #219 and #220 were closed, because raising it turned this file red three times over:
the `config_templates` `ctsp` block named a forecast model that is not registered and carried the
pre-nesting flat `charging_scheme` spelling (#218); the Creator wrote `NaN` into every nested
`charging_scheme` parameter for *both* classes and reported success (#219, `_add_info_indexed` did
not descend into nested config the way `_add_info_simple` does); and POI's `__constraint_cs_full`
passed a Series where a scalar is required (#220).

This is now the only EV coverage either agent type has, so `check_the_ev_premise` guards it: a
revert of `ev.share` to 0 fails by name here rather than quietly emptying the file. The fixture
ships `method: ["full", "min_soc"]` deliberately — `full` is the arm #220 broke, and reverting that
fix makes this module fail with the original `TypeError`.
"""
import json

import pytest

from tests.backend_matrix import require
from tests.scenario_run import REPO_ROOT

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

#: The workbook column holding each agent's drawn charging scheme, and the draw that exercises
#: #220. Asserted by `check_the_ev_premise` -- see there for why it is the drawn value.
SCHEME_COLUMN = 'ev/charging_scheme/method'
SCHEME_EXERCISING_220 = 'full'


@pytest.fixture(scope='module')
def run(scenario_runs):
    """Run the fixture once, switched to `poi`, and hand back the receipt and the fingerprint."""
    require(REQUESTED_FRAMEWORK, REQUESTED_SOLVER)
    # The premise, before spending 30 s on a run whose result would prove nothing without it.
    check_the_premise()

    entry = scenario_runs.run(CONFIG_ROOT, SCENARIO, framework=REQUESTED_FRAMEWORK,
                              solver=REQUESTED_SOLVER,
                              creator_method='new_scenario_from_files')

    used = {tuple(pair) for pair in json.loads(entry.record.read_text(encoding='utf-8'))}
    return used, entry.fingerprint


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

    # And an EV on each sheet. This fixture is the only place either agent type exercises the EV
    # path at all, and that path was broken in three independent ways until #218/#219/#220 -- so
    # reverting `ev.share` to 0 would take the coverage away and leave every test here green.
    check_the_ev_premise()


def check_the_ev_premise():
    """Both sheets must ship an EV, with its nested `charging_scheme` parameters filled.

    The halves are separate on purpose and each was real. `ev/owner` catches `ev.share` going back
    to 0. The nested parameters catch a workbook **built while #219 was present** -- note the
    scope: this reads the committed `agents.xlsx`, so it catches a bad fixture, never a bad helper.
    `tests/unit/creator/agents/test_add_info_indexed.py` is what catches the defect itself, and it
    is the only thing that does; a panel confirmed that reverting the helper leaves this module
    green, because `new_scenario_from_files` never runs the Creator.

    #219 is worth guarding here anyway because of *how* it failed: the Creator completed
    successfully and wrote `NaN`, so the workbook had the columns and the scenario built. Whether
    it then crashed depended on the draw -- `min_soc` reads the nested values and raises
    `IntCastingNaNError`, while `full` never reads them and the run completes with a silently
    wrong plant. A committed workbook full of NaN is exactly the thing no run would have told us
    about.

    Per sheet, never workbook-wide, for the same reason as `check_the_premise`: the ctsp block was
    the broken one, and letting the industry sheet vouch for it is how #218 stayed invisible.
    """
    import pandas as pd

    # Written only when `_add_info_indexed` descends into nested config (#219). The flat spelling
    # (`min_soc_val`) is what the ctsp block carried until #218, and the Executor cannot read it.
    nested = ('ev/charging_scheme/min_soc/val',
              'ev/charging_scheme/min_soc_time/val',
              'ev/charging_scheme/min_soc_time/time',
              'ev/charging_scheme/price_sensitive/threshold')

    # Both loops below are comprehensions over `nested`, so emptying it turns both assertions into
    # tautologies and this whole guard into a no-op -- demonstrated by a panel, which emptied it
    # and got a green run. `AGENT_TABLES` already carries this guard; this tuple did not.
    assert nested, 'the nested column list is empty, so the checks below assert nothing'

    with pd.ExcelFile(CONFIG_ROOT / SCENARIO / 'agents.xlsx') as book:
        # And that the loop runs at all. `test_the_fixture_still_ships_an_ev_on_both_sheets` calls
        # this directly, without `check_the_premise`'s sheet-list assertion in front of it, so an
        # empty or renamed workbook would otherwise skip every check below in silence.
        assert sorted(book.sheet_names) == ['ctsp', 'industry'], (
            f'{SCENARIO}/agents.xlsx has sheets {sorted(book.sheet_names)}; the per-sheet checks '
            f'below iterate that list, so anything else silently checks nothing')

        for sheet in book.sheet_names:
            frame = book.parse(sheet, index_col=0)

            assert 'ev/owner' in frame.columns and frame['ev/owner'].fillna(0).sum() > 0, (
                f'the {sheet} sheet declares no EV owner, so this scenario no longer covers the EV '
                f'path for {sheet} -- and nothing else in the repository does (#218/#219/#220)')

            absent = [column for column in nested if column not in frame.columns]
            assert not absent, (
                f'the {sheet} sheet is missing {absent}; it carries the pre-nesting flat spelling '
                f'that #218 removed, which the Executor cannot read')

            # Present is not filled. #219 wrote the columns and left every value NaN, so a
            # membership check alone reproduces exactly the blind spot this guards against.
            empty = [column for column in nested if frame[column].isna().all()]
            assert not empty, (
                f'the {sheet} sheet has {empty} present but entirely NaN -- that is #219 exactly: '
                f'the Creator reports success and writes nothing')

    # And that some agent actually drew `full`, which nothing checked until it was pointed out:
    # the value is named load-bearing in `tests/README.md` and this module's header, and moving the
    # fixture to `min_soc` left every test here green while removing the only exercise of #220.
    # Demonstrated, not assumed -- the edit was made and the module still passed 5/5.
    #
    # This is deliberately a property of the *drawn* value rather than of the distribution the YAML
    # offers, and so it is seed-dependent on purpose. If a reseed makes both agents draw `min_soc`,
    # the #220 coverage is genuinely gone and that is worth a red test, not a shrug.
    drawn = set()
    with pd.ExcelFile(CONFIG_ROOT / SCENARIO / 'agents.xlsx') as book:
        for sheet in book.sheet_names:
            column = book.parse(sheet, index_col=0).get(SCHEME_COLUMN)
            if column is not None:
                drawn |= set(column.dropna())

    assert SCHEME_EXERCISING_220 in drawn, (
        f'no agent in {SCENARIO} draws {SCHEME_EXERCISING_220!r} -- the sheets draw {sorted(drawn)}. '
        f'That is the arm #220 broke, and it is the only place any scenario exercises it, so this '
        f'fixture no longer covers #220 even though every other assertion here still passes')


def test_the_fixture_still_ships_an_ev_on_both_sheets():
    """The EV premise as a named test, in the fast tier, for the reason below."""
    check_the_ev_premise()


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
def test_both_agent_types_modelled_their_ev(run):
    """The EV reached the Executor and was carried through the run, per agent type.

    **What this can and cannot claim.** It asserts the EV was modelled and tracked, not that it
    charged: both EVs enter the horizon already at capacity, so the `full` scheme correctly demands
    nothing and their `setpoints` column is legitimately all zero. Asserting a non-zero setpoint
    here would be asserting a property of the drawn driving profile, and it would fail the first
    time the seed moved.

    That the run *exercises* #220 rather than merely tolerating it is established by reverting the
    fix: `__constraint_cs_full` then raises `TypeError` and this whole module fails. Verified, and
    the reason the fixture ships `method: ["full", "min_soc"]` rather than `min_soc` alone.
    """
    _, fingerprint = run

    problems = []
    for agent_type in ('ctsp', 'industry'):
        meters = fingerprint.get(f'agents/{agent_type}/meters.ft', {}).get('columns', {})
        socs = fingerprint.get(f'agents/{agent_type}/socs.ft', {}).get('columns', {})

        # `socs` columns are keyed by bare plant id, which does not say what kind of plant it is --
        # and every agent here also owns a battery, so `any soc moved` is satisfied by the battery
        # alone and would pass with no EV in the scenario at all. `meters` columns carry the plant
        # type in the name, so the EV's id is taken from there and then required in `socs`.
        ev_ids = [column[:-len('_ev_electricity')] for column in meters
                  if column.endswith('_ev_electricity')]
        if not ev_ids:
            problems.append(f'{agent_type}: no EV appears in meters.ft, so none was modelled')
            continue

        for ev_id in ev_ids:
            if ev_id not in socs:
                problems.append(f'{agent_type}: EV {ev_id} has no state of charge')
            elif not socs[ev_id]['max'] > 0:
                problems.append(f'{agent_type}: EV {ev_id} soc never leaves zero, which is '
                                f'indistinguishable from never having been modelled')

    assert not problems, (
        'the EV did not reach the Executor: ' + '; '.join(problems) +
        '. This scenario is the only EV coverage either agent type has (#218/#219/#220)')


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
