# HAMLET test suite

Split by scope, mirroring the package layout so a test sits next to the thing it covers.

```
tests/
  unit/          one class or function in isolation; no file I/O, no scenario
    creator/agents/
    executor/agents/
    executor/markets/
    executor/utilities/controller/{fbc,rtc}/
    executor/utilities/database/
    executor/utilities/grid_restrictions/
  integration/   several components wired together and solved, still no file I/O
    executor/
  e2e/           the shipped example, Creator -> Executor -> Analyzer
```

`unit/` and `integration/` mirror `hamlet/`, so the test for
`hamlet/executor/markets/electricity.py` lives at `tests/unit/executor/markets/`.

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

This suite was seeded by porting bug fixes that had been stranded on the paper branch. Tests in
`unit/` and `integration/` name the defect they pin in their docstring, and each was verified to
fail before the corresponding fix and pass after — except where a fix introduced the seam the
test uses, which the docstring says explicitly.

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
