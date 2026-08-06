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
    executor/
  e2e/           the shipped example, Creator -> Executor -> Analyzer
```

`unit/` and `integration/` mirror `hamlet/`, so the test for
`hamlet/executor/markets/electricity.py` lives at `tests/unit/executor/markets/`. Reading or
writing files puts a test in `integration/` regardless of how small it is.

## Running

```bash
python -m pytest tests
```

That runs unit and integration — seconds, HiGHS only, no solver licence. The end-to-end layer
is deselected by default because it runs the whole example:

```bash
python -m pytest tests -m e2e
```

Markers: `solver` (builds and solves a real optimisation model), `e2e` (runs the example).

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
