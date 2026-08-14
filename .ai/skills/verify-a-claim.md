# Skill: verify-a-claim

**Use when:** about to write a factual claim into a tracked file, a commit message, an MR
description, or a CHANGELOG entry. That includes claims about what the code does, what a test
covers, what a measurement showed, and what another branch contains.

**Rule:** check it against the tree you are on, at the commit you are on, and cite `path:line`.
If you cannot check it, leave it out or mark it explicitly as unverified. Nothing else in this
repository has cost as much as documentation that was believed.

## The four ways it has actually gone wrong here

1. **Checked on the wrong branch.** The paper branch carried documents describing a DuckDB
   migration, adaptive chunking and cache eviction as implemented. `grep -rn duckdb hamlet/`
   returns nothing. A backlog item was then written to "delete the fictional docs on `develop`" —
   where none of those files exist either. Two layers of unchecked claims, one on top of the other.
   *A claim about `develop` verified on a paper branch is not verified.*

2. **Quoted from a truncated tool output.** "41 differing statistics" was recorded as a
   re-baseline acceptance threshold. The golden test truncates its failure message at
   `differences[:40]`; the real number was 76. The threshold was 40 lines of output plus the
   `assert` line, counted with `wc -l`.
   *Never quote a count taken from a message that may have been truncated. Count the data.*

3. **Measured with a method that could not have detected the problem.** "Platform effect: none"
   was concluded from a Windows-vs-Linux comparison in which both sides ran a working tree that
   Windows git had materialised and Linux bind-mounted. It varied the kernel, never the checkout.
   The first Linux-native checkout moved 178 golden statistics.
   *Before recording a negative result, state what would have had to be different for the method
   to see the thing you are ruling out.*

4. **Benchmarked a no-op.** An install was timed at "10 s" by warming, uninstalling and re-timing
   inside one container. On that number a change worth 222 s was dismissed as worth 8.
   *One timed run per fresh environment.*

## Rules

- Cite `path:line`, and re-check the line number — they move.
- For "X does not exist", show the command: `git ls-tree`, `grep -rn`, `git ls-files`. Absence is
  the easiest claim to get wrong and the most confidently stated.
- For remote or project state — branch protection, mirrors, MR status, runners — use the
  authenticated `glab` CLI or the API. Do not infer it from git topology.
- A measurement gets the command that produced it, in the same paragraph.
- Prefer deleting a claim to hedging it. `.ai/context.md` is short on purpose.

**Exit criterion:** every factual sentence you wrote has a path, a command, or an API response
behind it that you ran in this session — or is marked as not established. Both are fine; a
confident sentence with neither is not.
