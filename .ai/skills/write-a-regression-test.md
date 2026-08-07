# Skill: write-a-regression-test

**Use when:** fixing a defect, or adding coverage to code that already has some.

**Read and follow:** `tests/README.md` — the layer map, where a test goes, and the provenance
section on what makes a test worth keeping.

**The test must fail against the unfixed code.** A test written after the fix, against the fixed
code, asserts that today equals today. Validate it by mutation: revert the fix, watch the test
fail with a message that names the thing, restore the fix.

## Rules

1. **Mutation-validate, always.** Not "I am confident it would fail" — run it. Both of this
   repository's structural guards were validated this way, and both revealed message quality
   problems that would have made a real failure hard to act on.
2. **A test that exercises no HAMLET code is not a test.** Several were deleted after they turned
   out to demonstrate polars or solver behaviour while reading as regression tests. The check is
   the one from `tests/README.md`: revert one source file and see which tests notice.

   ```bash
   git show origin/develop:PATH > PATH && python -m pytest tests -q; git checkout HEAD -- PATH
   ```

3. **Name the defect in the docstring.** Every test here says what it pins. A future reader
   deciding whether a failing assertion is load-bearing needs to know what it was for.
4. **Put it in the right tier.** Reading or writing files makes it `integration/` regardless of
   how small it is; `unit/` and `integration/` mirror `hamlet/`. Anything that runs the whole
   example is `e2e/` and gets a marker, because the default suite must stay in seconds.
5. **Do not reach for a fixture when the real object will do.** The retailer convention break got
   past 56 passing tests because every one used a hand-written fixture and none built a retailer
   table from a real config.
6. **Check the message, not just the outcome.** A test that fails with `assert False` costs the
   next person an hour. Make the failure name the table, the column and the delta.

**Exit criterion:** the test fails against the pre-fix code with a message that identifies the
defect, passes after, sits in the tier matching its scope, and `python -m pytest tests` is green.
