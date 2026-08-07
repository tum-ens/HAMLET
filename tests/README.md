# HAMLET test suite

Split by scope, mirroring the package layout so a test sits next to the thing it covers.

```
tests/
  unit/          one class or function in isolation; no solver, no scenario
    creator/agents/            creator/markets/
    executor/agents/           executor/markets/
    executor/utilities/controller/{,fbc,rtc}/
    executor/utilities/grid_restrictions/
  integration/   several components wired together, or code that touches the filesystem
    analyzer/                  creator/            executor/
    creator/format/            committed scenario-format references, one per version
  e2e/           the shipped example, Creator -> Executor -> Analyzer
    golden/                    committed reference numbers
```

`unit/` and `integration/` mirror `hamlet/`, so the test for
`hamlet/executor/markets/electricity.py` lives at `tests/unit/executor/markets/`. Reading or
writing files puts a test in `integration/` regardless of how small it is.

## Running

```bash
uv sync                 # once: installs pytest and HAMLET itself, from uv.lock
uv run python -m pytest tests
```

That runs unit and integration — seconds, HiGHS only, no solver licence. The end-to-end layer
is deselected by default because it runs the whole example:

```bash
uv run python -m pytest tests -m e2e      # smoke: does the example still run
uv run python -m pytest tests -m golden   # golden master: does it still produce the same numbers
```

The `uv run` prefix is dropped in the rest of this file for readability; if you have activated
`.venv` yourself, `python -m pytest` is the same thing.

Markers: `solver` (builds and solves a real optimisation model), `e2e` (runs the example),
`golden` (compares against committed reference numbers). The last two take a couple of minutes
each and are deselected by default.

## The golden master

Every other test pins a property someone thought to check. `tests/e2e/test_golden_master.py`
pins the numbers themselves, so a change that moves results has to be acknowledged rather than
noticed later. It runs the shipped example under a fixed seed and compares per-table row counts
and per-column sum/min/max against `tests/e2e/golden/simple_scenario.json`.

**When it fails**, the message names the tables and columns that moved and by how much. Decide
whether that is the change you meant. If it is, regenerate the reference and commit it *with*
the change, so the review sees the numbers move:

```bash
HAMLET_UPDATE_GOLDEN=1 python -m pytest tests -m golden
```

Reproducibility rests on seeding `random` and `numpy.random` and pinning `PYTHONHASHSEED`; the
Creator draws agent ids, plant ownership and sizings from all three. Two seeded runs were
verified to produce byte-identical scenarios and identical results. The column names in the
reference contain those seeded agent and plant ids, so changing how ids are generated will fail
this test — correctly, since agent identities would genuinely have changed.

## The scenario format version

`hamlet.constants.SCENARIO_FORMAT_VERSION` is the version of the **on-disk scenario folder**, not
of HAMLET itself. The Creator stamps it into `general/general.json`; the Executor and the Analyzer
refuse to read a folder carrying anything else, including no stamp at all. It exists because the
retailer in/out convention fix spanned the Creator and the Executor, so a scenario generated
before it and executed after it silently applied grid fees and levies to feed-in — plausible
numbers, no error, no warning.

A version nobody remembers to bump is worse than no version at all, so
`tests/integration/creator/test_scenario_format_shape.py` remembers. It generates the shipped
example (Creator only, a couple of seconds, so it runs in the default suite) and fingerprints its
*shape* — which files exist, which columns each table has, which keys each JSON carries, with the
seeded-random agent and plant ids normalised away — against
`tests/integration/creator/format/scenario_format_v<N>.json`.

**When it fails**, the message names what moved and lays out the three possibilities: revert,
regenerate the current reference, or bump. If you bump, create the new reference and commit both:

```bash
HAMLET_UPDATE_SCENARIO_FORMAT=1 python -m pytest tests/integration/creator/test_scenario_format_shape.py
```

Older reference files are not disposable — they record what those scenarios looked like. Two
further tests guard the bookkeeping: a reference must exist for the current version, and none may
exist for a version above it (which would mean a bump was left half-done).

**What this cannot catch.** A column that keeps its name and changes its *meaning* is invisible to
a structural fingerprint — and that is precisely the change that motivated the versioning. What
sees that is the golden master, by the numbers moving. So the bump rule has two halves, written
out in full at `c.SCENARIO_FORMAT_VERSION`, and only the first is automated. When the golden
master moves, the question to ask is whether the cause was a change to what the *Creator writes*
rather than to what the Executor computes; if it was, bump.

## Provenance

This suite was seeded by porting bug fixes that had been stranded on the paper branch. Tests
name the defect they pin in their docstring.

The check that matters is not "does it pass" but "what would make it fail". The useful way to
answer that is to revert one source file at a time and see which tests notice:

```bash
git show origin/develop:PATH > PATH && python -m pytest tests -q; git checkout HEAD -- PATH
```

A production change that no test notices is a coverage gap. Several tests were deleted rather
than kept after that check showed they exercised no HAMLET code at all — they demonstrated
polars or solver behaviour while reading as regression tests.

## The in/out convention

The single most error-prone thing in this codebase, so it is stated once, here.

**Retailer input files** (`input_data/retailers/**/*.csv`) name every column from the
**retailer's** point of view:

- `_out` is the retailer selling, i.e. **the agent buying**. This is the expensive direction,
  and it is where the grid fee and the levy sit, because both are charged on consumption.
- `_in` is the retailer buying, i.e. the agent feeding in.

**Transaction and forecast columns** inside the executor are from the **agent's** point of view:
`energy_in` is energy flowing into the agent, i.e. the agent buying.

Therefore any code mapping a retailer column onto a transaction or forecast column **must cross
in↔out**. That applies uniformly to `energy`, `balancing`, `grid` and `levies` — there is no
per-file exception.

Buying is always more expensive than selling. A test that appears to show otherwise is showing a
convention bug, not an arbitrage opportunity.
