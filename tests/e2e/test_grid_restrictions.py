"""End-to-end — §14a EnWG actually fires, and the agents actually obey it.

The golden master pins the *numbers* `grid_golden` produces. This file pins the thing that makes
those numbers worth pinning: that the restriction fired at all. A grid scenario in which nothing
ever overloads exercises the happy path and nothing else, and would keep passing while every
restriction mechanism underneath it rotted.

Both mechanisms are asserted separately, because they are independent code paths and one is far
easier to reach than the other:

* **Indirect** — variable grid fees, computed in `enwg_14a` and applied outside the solver in
  `agent_base.py`. Reached on every timestep once switched on; never causes a re-simulation.
* **Direct** — power control, which caps an agent through `apply_grid_commands` in the RTC. This
  is the only thing that can set `grid_ok = False`, so it is the only reason the
  `while not grid_ok` loop in `executor/setup.py` ever iterates. It is also the path that had
  never executed for anyone: it read a `hp_min_control` column nothing wrote, and no shipped
  example enables it.

A single "§14a ran" assertion would let a run where only the fees moved stand in for one where a
device was curtailed. These are different claims and they fail separately.

The assertions read the grid's own results rather than the agents' setpoints, deliberately: agent
setpoint tables only retain the forecast horizon, so by the end of a run the timestep a command
was issued at is no longer in them. `res_bus` is kept for every timestep, and in this fixture each
agent sits alone on its bus, so bus power *is* agent power as the power flow saw it.

**Direct power control has two methods, and everything above exercises only one of them.**
`grid_golden` selects `ems`, which asks the agent's own EMS to hit a net connection limit.
The other, `individual`, curtails named devices in a documented priority order -- batteries,
then EVs, then heat pumps -- and no shipped configuration reaches it: the three scenarios that
switch the restriction on run no electricity grid, and the two that run a grid apply no
restriction (#232). The `config_templates` default is `individual`, so the branch a new user
gets is the one nothing ran. The second half of this file covers it, on a fixture derived from
`grid_golden` by two config edits -- one selects the method, the other **lowers §14a's 4200 W
floor to 1000 W**, without which the priority order is not observable at all. That departure is
deliberate and is the price of the coverage; `INDIVIDUAL_EDITS` argues it, and the `ems` tests
above are what keep the real 4200 W under test.

**What that still does not reach.** Stated in full, with counts, because a partial list reads as
a complete one -- **34 of the method's 66 statements still never execute**:

* The **over-generation** branch (`enwg_14a.py:421-465`), **23 statements**. `grid_golden` never
  over-generates -- across six instrumented runs, 817 dispatches into the method entered
  over-consumption 169 times and neither branch 648 times, and over-generation not once. Nor
  would a threshold edit reach it: every battery ships `b2g_0 = 0.0` and every EV `v2g_0 = 0`, so
  the branch's two device filters cannot match as this fixture is configured. #233.
* The **heat-pump loop's body** (`enwg_14a.py:396-419`), **11 statements**, though the loop
  itself is entered. No heat pump is ever curtailed under the shipped order -- correctly, since
  the EV above it absorbs the whole budget first -- so the guard runs and the body does not. Two
  statements are unreachable outright rather than merely unreached: the §14a scaling arm at
  `:406` needs a heat pump drawing over 11 kW and all four here are rated 2800-9800 W, and the
  `hp_min_control` clamp at `:410` is identically false while every heat pump is under 11 kW,
  because `hp_min_control` then equals the threshold and the reduction is already capped at
  `p_mw - threshold`. #209 would change that arithmetic; it would not by itself make it reachable.

**Measured, not inferred: four mutations inside that body stay green** -- deleting the heat-pump
loop, disabling the `hp_min_control` clamp, forcing the over-11 kW arm, and changing the scaling
factor from 0.6 to 0.3. A review panel ran all four. Do not read the coverage below as protecting
them.

So this file pins *which* device §14a curtails first and *that* the device obeyed. It does not
pin how much is taken off a heat pump.
"""
import json
from pathlib import Path

import pytest

from tests.scenario_run import REPO_ROOT

SCENARIO = 'grid_golden'

#: Where the §14a fee lands once `agent_base` has applied it: the agent's own forecast, which is
#: what it then optimises against. `res_variable_grid_fee.csv` does *not* hold fees -- see
#: `test_the_variable_grid_fee_reached_the_agents`.
FEE_COLUMN = 'grid_market_out'
CONFIG_ROOT = REPO_ROOT / 'tests' / 'e2e' / 'scenarios'

#: `grid_golden`'s topology assigns agents to buses by id, and only this entry point keeps the ids
#: `agents.xlsx` declares. A module constant rather than a literal in the fixture so that
#: `test_scenario_cache_key` can check this request against the golden master's and fail if the
#: two stop being the one shared run.
CREATOR_METHOD = 'new_scenario_from_files'

#: Turns `grid_golden` into a fixture for the `individual` direct-power-control method.
#:
#: **The first edit selects the method; the second is what makes its priority order observable,
#: and without it this whole half of the file would be green under any ordering.** Measured over
#: one instrumented run with only the first edit applied: the method is dispatched into 152 times,
#: and 151 of those calls see at most one controllable device class above the 4200 W floor. The
#: one call that sees two is asked to shed 1749 W from devices that together have 957 W of
#: headroom, so both go to the floor and the order they are taken in changes nothing -- reversing
#: both documented priority orders leaves the commands of all 152 calls identical, and the written
#: `res_direct_power_control.csv` byte-identical. That is the "green for a new wrong reason" this
#: fixture exists to avoid. (The 817 figure in the module docstring is a different quantity: six
#: runs summed, not one.)
#:
#: The floor is what decides it: it sets how much headroom a device above it has, and at 4200 W
#: this fixture's devices sit too close to it for the reduction budget ever to run out mid-order.
#: At 1000 W they do not, and the order decides who gives way.
#:
#: **Lowering it cannot reorder the policy, because the sequence of the device loops is a literal
#: constant** (`enwg_14a.py:356`, `:376`, `:395` -- battery, then EV, then heat pump, in that
#: order in the source). What `threshold` does change is which devices enter those loops
#: (`:348`, `:352` filter on it) and how much each is asked to give; it also reaches
#: `grid_db._set_heat_pump_minimum_power`, which writes `hp_min_control` from the same key. So the
#: lower floor changes *who competes*, which is the point, and cannot change *who wins*, which is
#: what makes it a fair test of the order. What it does *not* test is §14a's actual 4200 W number,
#: which the `ems` tests above keep.
#:
#: Both replacements are unique in the fixture, which `test_the_individual_edits_are_unique`
#: asserts rather than inherits: `prepare_config` applies them with `str.replace`, so a second
#: occurrence would be rewritten silently.
INDIVIDUAL_EDITS = {'grids.yaml': [('method: ems', 'method: individual'),
                                   ('threshold: 4200', 'threshold: 1000')]}

#: The floor `INDIVIDUAL_EDITS` runs with, in watts. **Not §14a's 4200 W** -- deliberately lowered,
#: and the name says so because a reader who mistakes this for the regulation's number will
#: mistake what the tests below prove. See `INDIVIDUAL_EDITS` for why, and the `ems` tests above
#: for the fixture that does keep 4200 W. A device at or below this has nothing left to give,
#: which is what makes "was a higher-priority device passed over?" decidable at all.
LOWERED_THRESHOLD_W = 1000

#: The device priority order `__individual_device_control` documents for over-consumption
#: (`enwg_14a.py:302-309`), highest priority first: batteries first because curtailing one costs
#: no comfort, heat pumps last because it costs the most.
PRIORITY = ('battery', 'ev', 'hp')

#: Slack when asking whether a device is still above the floor, in watts. Commands are whole watts
#: (`int(... * c.MWH_TO_WH)`) while the power flow answers in floats, and `optim_poi`'s
#: `__apply_individual_control` pins the plant's target to the command rather than recomputing it,
#: so the realised power need not land on the command exactly. Measured margin is 0 W.
#:
#: Note the truncation runs the *other* way from the `ems` tests above: `int()` truncates toward
#: zero and this method's commands are negative for a consuming device, so rounding can only put a
#: device at or slightly *below* its floor, never above. Same `+ 1`, different mechanism -- the
#: `ems` case is a cap recomputed from each pass's power flow.
MARGIN_W = 1


@pytest.fixture(scope='module')
def grid_results(scenario_runs):
    """Run the fixture once and hand back its results directory.

    This request is byte-identical to `test_golden_master`'s for `grid_golden`, and going
    through `scenario_runs` is what lets the two share one run instead of paying for it twice
    (70-125 s; see `tests/scenario_cache.py` for the measurement and why it is a band).
    They only ever co-exist in a session that selects both markers, which no CI job does --
    see `tests/scenario_cache.py` for why the saving is local-only.
    """
    return scenario_runs.run(CONFIG_ROOT, SCENARIO, creator_method=CREATOR_METHOD).results


def read_csv(results, name):
    import pandas as pd

    path = results / 'grids' / 'electricity' / name
    assert path.exists(), f'{name} was not written, so that part of the grid stage did not run'
    return pd.read_csv(path)


@pytest.mark.e2e
def test_the_feeder_actually_overloads(grid_results):
    """The premise of everything below. Without an overload, §14a has nothing to do.

    Stated as its own test so that a fixture which quietly stops overloading -- a changed profile,
    a re-sized transformer, a solver that shifts charging -- fails *here*, naming the cause,
    rather than showing up as a puzzling absence of commands further down.
    """
    trafo = read_csv(grid_results, 'res_trafo.csv')
    overloaded = trafo[trafo['loading_percent'] > 100]

    assert not overloaded.empty, (
        f'the transformer never exceeded 100 % loading (peak '
        f'{trafo["loading_percent"].max():.1f} %), so no restriction could be triggered and every '
        f'assertion in this file would pass vacuously')


@pytest.mark.e2e
def test_the_variable_grid_fee_reached_the_agents(grid_results):
    """The indirect mechanism, read where it lands rather than where it is computed.

    **Not** from `res_variable_grid_fee.csv`, despite the name. `enwg_14a` writes
    `combined_loading_for_bus` under that key -- per-unit loading, roughly 0.1 to 2.3 -- while the
    fees themselves go only into `restriction_commands` and are never saved. Asserting that that
    table varies is asserting that the *loading* varies, which is what the overload test above
    already says, and it holds whether or not a single agent ever sees a fee.

    That is not hypothetical. Severing the mechanism entirely -- an early `return` in
    `agent_base.apply_grid_commands`, the only place a variable fee reaches an agent -- leaves the
    grid-side table completely unchanged, and a review panel demonstrated all four tests in this
    file passing against it.

    The fee lands in each agent's forecast, which is the whole point of an *indirect* control: it
    changes what the agent optimises against. With the mechanism live it varies over the horizon
    and rises above the flat base rate; with it severed the column is the shipped constant.
    """
    import polars as pl

    spreads = {}
    for type_dir in sorted((grid_results / 'agents').iterdir()):
        if not type_dir.is_dir():
            continue
        for agent_dir in sorted(type_dir.iterdir()):
            forecasts = pl.read_ipc(agent_dir / 'forecasts.ft', memory_map=False)
            if FEE_COLUMN not in forecasts.columns:
                continue
            column = forecasts[FEE_COLUMN]
            spreads[agent_dir.name] = float(column.max()) - float(column.min())

    assert spreads, (
        f'no agent forecast carries a {FEE_COLUMN!r} column, so the grid fee an agent optimises '
        f'against cannot be read and this test asserts nothing')
    assert max(spreads.values()) > 0, (
        f'every agent sees a constant grid fee across the whole horizon ({spreads}), so the '
        f'variable grid fee never reached them -- the indirect mechanism is not connected')


@pytest.mark.e2e
def test_direct_power_control_issued_a_command(grid_results):
    """The direct mechanism fired, which also means the `while not grid_ok` loop iterated.

    `EnWG14a.execute` starts from `grid_ok = True` and only `__calculate_direct_power_control` can
    set it False, so a command existing at all is proof the timestep was re-simulated: the command
    is written on one pass and the loop runs again because of it.
    """
    commands = read_csv(grid_results, 'res_direct_power_control.csv')

    assert not commands.empty, (
        'no direct power control command was issued, so the `while not grid_ok` loop never '
        'iterated and the restriction was never applied')
    assert (commands['control_result'] > 0).all(), (
        f'a command capped an agent at zero or below: {commands["control_result"].tolist()}')


@pytest.mark.e2e
def test_the_curtailment_command_was_respected(grid_results):
    """The one that matters: the agent drew no more than the grid operator allowed.

    Everything above can hold while the cap is silently discarded -- which is exactly what
    happened for `framework: poi` until the backend grew its own `apply_grid_commands`, because
    the base class's was a no-op. The grid stage cannot notice that by itself: it re-simulates,
    gets the same answer, and converges on an uncapped grid.

    Read from `res_bus` at the agent's own bus, taking the **last** row for the timestep. Not the
    converged one: `grids.yaml` sets `max_iteration: 3`, and at both restricted timesteps
    `executor/setup.py` forces `grid_ok = True` on the iteration cap rather than reaching a fixed
    point (`res_trafo.csv` carries 4 rows at each of them). The last row is the state the run
    actually kept, which is what the assertion is about.

    The cap is recomputed from each pass's power flow, so at the end of a converging sequence it
    sits exactly on the power drawn -- the margins here are 0 W, and the `+ 1` below is what
    absorbs the rounding. Under a backend that ignores the cap the sequence does not converge at
    all: the draw stays at its uncontrolled value while the cap keeps asking for less, and the
    gap is hundreds of watts.
    """
    commands = read_csv(grid_results, 'res_direct_power_control.csv')
    bus = read_csv(grid_results, 'res_bus.csv')
    bus_column = bus.columns[0]

    agent_bus = {path.name: json.loads((path / 'account.json').read_text(encoding='utf-8'))
                 ['general']['bus']
                 for type_dir in (grid_results / 'agents').iterdir() if type_dir.is_dir()
                 for path in type_dir.iterdir() if path.is_dir()}

    checked = 0
    for _, command in commands.iterrows():
        rows = bus[(bus[bus_column] == agent_bus[command['id_agent']])
                   & (bus['timestamp'] == command['timestamp'])]
        assert not rows.empty, (
            f'no power flow result at bus {agent_bus[command["id_agent"]]} for '
            f'{command["timestamp"]}, so the command cannot be checked against anything')

        drawn_w = rows['p_mw'].iloc[-1] * 1e6
        cap_w = command['control_result']
        assert drawn_w <= cap_w + 1, (
            f'{command["id_agent"]} was capped at {cap_w:.0f} W at {command["timestamp"]} but drew '
            f'{drawn_w:.0f} W, so the grid operator\'s command was accepted and then ignored')
        checked += 1

    # A loop over an empty table asserts nothing. The previous test already fails on that, but
    # this one must not be able to pass by iterating zero times.
    assert checked > 0, 'no command was checked'


# ----------------------------------------------------------------------------------------------
# The `individual` method: which devices get curtailed, and in what order. See the module
# docstring for why none of the above reaches it.
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def individual_results(scenario_runs):
    """`grid_golden` again, with direct power control switched to the `individual` method.

    A second run rather than a second scenario folder: the blocker #232 records is configuration,
    not code, and `grid_golden`'s own sizing is load-bearing in ways a sibling would have to
    reproduce exactly (see `.ai/context.md` -- the transformer sits between the uncontrolled peak
    and the guaranteed floor, and moving either stops it overloading at all).

    This is a *different request* from the fixture above, so it does not share that run: it costs
    the e2e job one additional scenario run.
    """
    return scenario_runs.run(CONFIG_ROOT, SCENARIO, creator_method=CREATOR_METHOD,
                             config_edits=INDIVIDUAL_EDITS).results


def controllable_devices(results, threshold_w):
    """Every §14a-controllable device's final draw at every timestep, in watts.

    Joined from three tables because no one of them carries both identity and power. The grid's
    own `topology.xlsx` -- written into the results, so this reads what the run actually built --
    names each element's bus, type and plant id; `res_load` and `res_sgen` carry the power flow's
    answer against a bare element index and nothing else.

    **The two tables sign power oppositely and the join normalises it.** A load consuming is
    positive, an sgen consuming (a battery charging) is negative. The priority order ranks a
    battery against an EV, so the two are only comparable once both are expressed as draw from
    the grid, and comparing raw `p_mw` would rank a charging battery below every load.

    A restricted timestep carries one row per power-flow pass (`grids.yaml` sets
    `max_iteration: 3`); the last is the state the run kept, which is the same choice
    `test_the_curtailment_command_was_respected` makes and for the same reason. **`.nth(-1)` and
    not `.last()`** -- pandas' `GroupBy.last()` is the last *non-NA* value per column
    independently, so a final pass that wrote a NaN power would silently hand back an earlier
    pass, and the caller would then assert about a state the run did not keep. No pass writes NaN
    today, which is what makes the distinction easy to get wrong and cheap to get right.
    """
    import pandas as pd

    book = pd.ExcelFile(results / 'grids' / 'electricity' / 'topology.xlsx')

    frames = []
    for sheet, kind_column, table in (('load', 'load_type', 'res_load'),
                                      ('sgen', 'plant_type', 'res_sgen')):
        identity = book.parse(sheet).rename(columns={'Unnamed: 0': 'element'})
        power = read_csv(results, f'{table}.csv').rename(columns={'Unnamed: 0': 'element'})
        power = power.groupby(['element', 'timestamp'], as_index=False).nth(-1)
        frame = power.merge(identity[['element', 'bus', kind_column, 'id_plant', 'id_agent']],
                            on='element', how='left').rename(columns={kind_column: 'kind'})
        frame['draw_w'] = frame['p_mw'] * 1e6 * (-1 if sheet == 'sgen' else 1)

        # A left join that matched nothing yields NaN identity and would then be filtered out by
        # `isin(PRIORITY)` below -- silently shrinking the device population rather than failing.
        # Only a *total* loss would reach the emptiness check, so partial drift is caught here.
        unmatched = frame['kind'].isna().sum()
        assert not unmatched, (
            f'{unmatched} of {len(frame)} {table} rows have no entry in topology.xlsx, so those '
            f'devices would vanish from the ordering check instead of failing it')
        frames.append(frame)

    devices = pd.concat(frames, ignore_index=True)
    devices = devices[devices['kind'].isin(PRIORITY)].copy()
    assert not devices.empty, (
        f'no device of any controllable type {PRIORITY} appears in the grid tables, so every '
        f'ordering assertion below would iterate over nothing')

    devices['rank'] = devices['kind'].map(PRIORITY.index)
    devices['headroom_w'] = devices['draw_w'] - threshold_w
    return devices


def test_the_individual_edits_are_unique():
    """`prepare_config` edits with `str.replace`, which rewrites *every* occurrence.

    Unmarked, so it runs in the default tier: it reads the tracked config and needs no scenario
    run, and a config edit that silently grew a second target should fail in seconds rather than
    after a two-minute run. `prepare_config` already fails loudly when a replacement matches
    nothing; nothing warns when it matches twice, and `threshold:` in particular is a key §14a
    could plausibly grow a second one of.
    """
    text = (CONFIG_ROOT / SCENARIO / 'grids.yaml').read_text(encoding='utf-8')
    counts = {old: text.count(old) for old, _ in INDIVIDUAL_EDITS['grids.yaml']}

    assert all(count == 1 for count in counts.values()), (
        f'{counts} -- every edit in INDIVIDUAL_EDITS must match exactly once in '
        f'{SCENARIO}/grids.yaml. A count of 0 means the fixture was renamed out from under this '
        f'file; a count above 1 means the edit now rewrites somewhere it was never meant to')


@pytest.mark.e2e
def test_the_run_used_the_individual_method(individual_results):
    """The config edits reached the configuration the run actually read.

    Read from the copy the run wrote into its own results rather than from the prepared input, so
    this is a receipt and not a restatement of the request.

    That receipt settles the dispatch, because the dispatch is nothing but a `match` on this key
    (`enwg_14a.py:250-261`) whose fallback arm raises: with `method: individual` in the config the
    run either entered `__individual_device_control` or died. There is no third outcome and no
    silent fallback to `ems`.
    """
    config = (individual_results / 'config' / 'grids.yaml').read_text(encoding='utf-8')

    assert 'method: individual' in config, (
        'the run read a grids.yaml that does not select the individual method, so every '
        'assertion below is about the ems path the tests above already cover')
    assert f'threshold: {LOWERED_THRESHOLD_W}' in config, (
        f'the run read a grids.yaml whose §14a floor is not {LOWERED_THRESHOLD_W} W; at the '
        f'shipped 4200 W the priority order is not observable at all -- see INDIVIDUAL_EDITS')


@pytest.mark.e2e
def test_both_priority_adjacencies_are_exercised(individual_results):
    """The premise of the ordering test below, and the reason this fixture lowers the floor.

    An ordering can only be observed where two device classes compete: one eligible device is
    curtailed whatever the order says. So this asserts that the run really does reach a bus and
    timestep where a battery and an EV are both above the floor, *and* one where an EV and a heat
    pump are -- and that a command was actually issued there, since two idle eligible devices
    decide nothing either.

    Stated as its own test, and as an equality against every documented adjacency rather than a
    count, so that a fixture which quietly stops producing one of the two pairs fails *here*,
    naming the pair it lost, rather than leaving the test below to pass over the half it can
    still see. `test_the_feeder_actually_overloads` exists for the same reason one layer up.

    **This one is a pinned trajectory and the test below deliberately is not.** Each arm rests on
    a single bus and timestep -- today (10:00, bus 3) for EV/heat-pump and (10:00, bus 4) for
    battery/EV -- so a solver or pandapower bump that moves either takes this red. That is the
    intended trade: a premise guard has to assert the fixture still produces the situation, and
    the cost of it failing loudly on an unrelated change is smaller than the cost of the ordering
    test below silently covering half of what it claims. Re-establish the premise before
    re-baselining it.
    """
    devices = controllable_devices(individual_results, LOWERED_THRESHOLD_W)
    commands = read_csv(individual_results, 'res_direct_power_control.csv')
    commanded = set(zip(commands['timestamp'], commands['id_plant']))

    exercised = set()
    for (timestamp, bus), group in devices[devices['headroom_w'] > 0].groupby(['timestamp',
                                                                               'bus']):
        here = devices[(devices['timestamp'] == timestamp) & (devices['bus'] == bus)]
        if not any((timestamp, plant) in commanded for plant in here['id_plant']):
            continue
        kinds = set(group['kind'])
        exercised |= {pair for pair in zip(PRIORITY, PRIORITY[1:]) if set(pair) <= kinds}

    expected = set(zip(PRIORITY, PRIORITY[1:]))
    assert exercised == expected, (
        f'exercised {sorted(exercised)} of the documented adjacencies {sorted(expected)}. Every '
        f'missing pair is a piece of the priority order that no assertion in this file can see: '
        f'reversing it would leave the suite green')


@pytest.mark.e2e
def test_devices_are_curtailed_in_priority_order(individual_results):
    """The policy itself: a device is only curtailed once everything above it has been used up.

    **An invariant over whatever the run produced, not a pinned trajectory.** Changing the
    priority order changes the commands, which changes the re-simulated power flow, which moves
    the whole run -- so a test that pinned specific timesteps would go red under the mutation for
    the wrong reason, and equally red under an unrelated solver change. This instead asserts the
    property the order *means*, at every command the run happens to issue, and it holds whichever
    timesteps overload.

    **A peer is judged by its own command where it has one, and by the measured power only where
    it has none.** The tempting simplification -- read every device's final power and call
    anything at the floor "used up" -- has a hole. Within one power-flow pass the restriction
    computes its commands *from* that pass's `res_*` rows, and those are the rows the pass then
    stores; a device commanded on the run's last pass is therefore recorded in its
    **pre-command** state, because no further pass ever re-simulates it. Reading the measured
    power alone would report such a device as passed over against a perfectly correct
    implementation, and whether that happens depends on whether the timestep converged or hit
    `max_iteration` -- a property of the fixture, not of the policy. Taking the command as the
    peer's power when one exists removes the dependency entirely.

    That the floor is a floor is what makes either reading decidable: a device that gave
    everything it had sits exactly on it, so "still had headroom" and "is still above the floor"
    are the same question. That is also why this test needs the lowered floor -- see
    `INDIVIDUAL_EDITS`.

    Verified by mutation, each adjacency separately, since a single full reversal can be caught by
    one arm while the other asserts nothing: swapping only battery with EV fails this on a battery
    left charging 4500 W above the floor while its owner's EV was curtailed, and swapping only EV
    with heat pump fails it on a heat pump curtailed while the EV beside it still drew 7200 W.

    **In the passing direction this makes exactly one comparison**, battery against EV, because
    under the shipped order no heat pump is ever curtailed -- which is the correct outcome, not a
    gap. An ordering violation only becomes visible when the lower-priority device *is* curtailed,
    so the EV/heat-pump arm is asserted by being sensitive to the mutation above rather than by
    comparing anything today. `test_both_priority_adjacencies_are_exercised` is what stops fixture
    drift from removing that sensitivity silently.
    """
    devices = controllable_devices(individual_results, LOWERED_THRESHOLD_W)
    commands = read_csv(individual_results, 'res_direct_power_control.csv')

    assert not commands.empty, (
        'the individual method issued no command at all, so it curtailed nothing and there is no '
        'order to check -- see test_direct_power_control_issued_a_command for the ems path')

    #: (timestamp, plant) -> the power the grid operator allowed it, as draw in watts. Commands
    #: are written negative for a consuming device under this method; see `read_csv`'s callers
    #: above for the `ems` method's opposite convention on the same column.
    allowed_w = {(row['timestamp'], row['id_plant']): abs(row['control_result'])
                 for _, row in commands.iterrows()}

    passed_over, compared = [], 0
    for _, command in commands.iterrows():
        curtailed = devices[(devices['id_plant'] == command['id_plant'])
                            & (devices['timestamp'] == command['timestamp'])]
        assert len(curtailed) == 1, (
            f'{command["id_plant"]} at {command["timestamp"]} matched {len(curtailed)} grid '
            f'results, so which device was curtailed cannot be decided')
        curtailed = curtailed.iloc[0]

        higher = devices[(devices['timestamp'] == command['timestamp'])
                         & (devices['bus'] == curtailed['bus'])
                         & (devices['rank'] < curtailed['rank'])]
        compared += len(higher)
        for _, peer in higher.iterrows():
            key = (peer['timestamp'], peer['id_plant'])
            peer_w = allowed_w.get(key, peer['draw_w'])
            if peer_w - LOWERED_THRESHOLD_W > MARGIN_W:
                source = 'was allowed' if key in allowed_w else 'still drew'
                passed_over.append(
                    f'{command["timestamp"]} bus {curtailed["bus"]}: {curtailed["kind"]} '
                    f'{curtailed["id_plant"]} was curtailed while the higher-priority '
                    f'{peer["kind"]} {peer["id_plant"]} {source} {peer_w:.0f} W, '
                    f'{peer_w - LOWERED_THRESHOLD_W:.0f} W above the '
                    f'{LOWERED_THRESHOLD_W} W floor')

    assert not passed_over, (
        'the documented priority order ' + ' -> '.join(PRIORITY) + ' was not followed:\n  '
        + '\n  '.join(passed_over))

    # Every command above may sit at the top of the order, in which case the loop compared
    # nothing and the assertion it guards is vacuous. The adjacency test states the stronger
    # premise; this makes *this* test unable to pass by never looking.
    assert compared > 0, (
        'no curtailed device had a higher-priority device beside it, so nothing in this test was '
        'ever compared and the priority order was not checked')


@pytest.mark.e2e
def test_the_individual_commands_were_respected(individual_results):
    """The `individual` analogue of `test_the_curtailment_command_was_respected`.

    Everything above can hold while the per-plant cap is accepted and then discarded, and the grid
    stage cannot notice: it re-simulates, gets the same answer, and converges on an uncapped grid.
    That is the defect `poi` shipped with on the `ems` path until !209, one method over -- the two
    take different routes into the backend (`optim_poi.__apply_individual_control` pins each named
    plant's target, where `ems` constrains the agent's net connection), so the `ems` test above
    does not cover this one.

    **Read per plant, not per bus.** The `ems` test can use `res_bus` because that method caps an
    agent's whole connection and each agent sits alone on its bus; this method names individual
    plants, so the assertion has to reach the plant's own row in `res_load`/`res_sgen`.

    **And note the opposite sign convention on the same column.** `ems` writes `control_result`
    positive for net consumption -- `test_direct_power_control_issued_a_command` asserts `> 0` on
    it -- while `individual` writes a per-plant target that is negative for a consuming device.
    `res_direct_power_control.csv` therefore cannot be read without knowing which method produced
    it, which is worth knowing before adding a third reader of that table.
    """
    devices = controllable_devices(individual_results, LOWERED_THRESHOLD_W)
    commands = read_csv(individual_results, 'res_direct_power_control.csv')

    assert (commands['control_result'] < 0).all(), (
        f'this method writes a per-plant target that is negative for a consuming device, and '
        f'every command in this fixture curtails consumption, so a non-negative value means the '
        f'convention changed and the comparison below is meaningless: '
        f'{commands["control_result"].tolist()}')

    checked = 0
    for _, command in commands.iterrows():
        plant = devices[(devices['id_plant'] == command['id_plant'])
                        & (devices['timestamp'] == command['timestamp'])]
        assert len(plant) == 1, (
            f'{command["id_plant"]} at {command["timestamp"]} matched {len(plant)} grid results')

        drawn_w = plant['draw_w'].iloc[0]
        allowed_w = abs(command['control_result'])

        # §14a is a *guarantee* as well as a cap: the floor is the power the operator may never
        # curtail below. Asserted here because nothing else does, and because every assertion in
        # this file is otherwise about curtailing too little or in the wrong order -- none would
        # notice the method curtailing too hard. The one device taken to its floor lands on it
        # exactly, so this is not slack being asserted against slack.
        assert allowed_w >= LOWERED_THRESHOLD_W - MARGIN_W, (
            f'{command["id_plant"]} ({plant["kind"].iloc[0]}) was capped at {allowed_w:.0f} W at '
            f'{command["timestamp"]}, below the {LOWERED_THRESHOLD_W} W §14a guarantees it')

        assert drawn_w <= allowed_w + MARGIN_W, (
            f'{command["id_plant"]} ({plant["kind"].iloc[0]}) was capped at {allowed_w:.0f} W at '
            f'{command["timestamp"]} but drew {drawn_w:.0f} W, so the grid operator\'s per-plant '
            f'command was accepted and then ignored')
        checked += 1

    assert checked > 0, 'no command was checked'
