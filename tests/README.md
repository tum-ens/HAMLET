# HAMLET test suite

Layered, following the target described in the roadmap. Everything here runs in seconds and
needs no solver licence — HiGHS only.

| Layer | Directory | What it covers |
|---|---|---|
| L1 | `l1_functions/` | Pure functions: `hamlet/functions.py`, fee formulae, schema helpers |
| L2 | `l2_components/` | **Component physics.** Per component, a short single-plant linopy model with known inputs, asserting the resulting bounds/dispatch |
| L4 | `l4_accounting/` | Accounting invariants: net-vs-gross energy, fee and levy application, balance closure |
| — | `regression/` | Narrow regression tests for defects that are not component physics (I/O guards, config handling) |

L0 (contracts), L3 (market clearing) and L5 (golden-master integration) are not populated yet.

## Running

```bash
python -m pytest tests -q
```

Markers are declared in `pytest.ini`. `solver` marks tests that build and solve a real linopy
model; they are slower (~1 s each) but still run by default.

```bash
python -m pytest tests -q -m "not solver"
```

## Provenance

This layer was seeded by porting the bug fixes that had been stranded on the paper branch. Each
test in `l2_components/`, `l4_accounting/` and `regression/` names the defect it pins in its
docstring, and was verified to **fail before** the corresponding fix and **pass after**.

## The in/out convention

Several tests depend on HAMLET's power-flow direction convention, which is the single most
error-prone thing in this codebase. Stated once, here:

- **Retailer input files** (`input_data/retailers/**/*.csv`) name their columns from the
  **retailer's** point of view for `energy` and `balancing`: `energy_price_out` is the price at
  which the retailer sells, i.e. what the **agent pays to buy**.
- **Transaction and forecast columns** inside the executor are from the **agent's** point of
  view: `energy_in` is energy flowing into the agent, i.e. the agent buying.
- Therefore code mapping retailer columns onto transaction columns **must cross in↔out**.

`grid.csv` and `levies.csv` in the shipped example data do *not* follow the retailer convention —
they are written from the agent's point of view (`levies_price_in` is charged on consumption).
The code compensates by not crossing for those two. This inconsistency is a known wart; see the
roadmap item on retailer-input normalisation. Tests here pin the behaviour that matches the
shipped data.
