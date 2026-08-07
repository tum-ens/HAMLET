# Skill: golden-master-failed

**Use when:** `python -m pytest tests -m golden` fails, or you are about to make a change you
expect to move it.

**Read and follow:** `tests/README.md` §"The golden master" and §"The scenario format version".
The bump rule is written out in full at `hamlet.constants.SCENARIO_FORMAT_VERSION`.

**The failure is information, not an obstacle.** Regenerating the reference to make the suite
green is the one move that destroys the only thing this test does. It is the guard rail that makes
executor refactors possible, and it exists because seven defects introduced while porting the
paper fixes were caught by review or ad-hoc measurement and *none* by a passing suite.

## Procedure

1. **Read what moved.** The message names table, column and delta. Do not summarise it from
   memory, and do not count from it — it truncates at `differences[:40]`, which once turned 76
   moved statistics into a recorded "41". Count from the reference diff instead.
2. **Decide whether it is the change you meant.** Structural moves (a table or column appearing
   or disappearing) are a different question from numeric ones. Small numeric moves across many
   columns usually mean a different-but-equally-optimal dispatch; a large move in one column
   usually means physics.
3. **Ask the Creator-vs-Executor question.** If the numbers moved because of a change to *what
   the Creator writes* rather than to *what the Executor computes*, the on-disk scenario format
   changed and `c.SCENARIO_FORMAT_VERSION` must be bumped. This half of the rule is **not
   automated** — a column that keeps its name and changes its meaning is invisible to the shape
   fingerprint, and that is precisely the change that motivated versioning in the first place.
4. **Regenerate only together with the change**, so review sees the numbers move:

   ```bash
   HAMLET_UPDATE_GOLDEN=1 python -m pytest tests -m golden
   ```

5. **Say so in the CHANGELOG**, with the direction and rough size of the move and why it happened.

## Traps

- **The reference is solver-coupled.** Gurobi → HiGHS moved 3 row counts and 76 statistics with
  no structural change at all. Regenerate with the solver the example is configured for (HiGHS),
  never with one you substituted locally. It is *not* platform-coupled: Windows and Linux agree
  under the same solver.
- **Never edit a reference belonging to an older format version.** Those record what that
  version's scenarios looked like. Bumping means a new file, not an edit.
- **A shape-test failure is a different card's problem** —
  `tests/integration/creator/test_scenario_format_shape.py` failing means the Creator's *output
  shape* moved, and its own message lays out revert / regenerate / bump.

**Exit criterion:** `python -m pytest tests -m golden` passes; the regenerated reference is in the
same commit as the change that moved it; the CHANGELOG names the move; and if the Creator's output
changed, `SCENARIO_FORMAT_VERSION` is bumped with its new
`tests/integration/creator/format/scenario_format_v<N>.json` alongside.
