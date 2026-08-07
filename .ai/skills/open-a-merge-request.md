# Skill: open-a-merge-request

**Use when:** work is finished and needs to reach `develop`.

**Read and follow:** `CONTRIBUTING.md` for branch naming (`type-issue-nr-short-description`) and
the workflow. `.ai/context.md` §"Branches and propagation" for what is protected and what is
mirrored.

**`master` and `develop` are protected with push access "No one".** There is no fallback path;
everything lands by merge request. Branch off `origin/develop`, never off a paper branch.

## Rules

1. **No AI attribution trailers.** No `Co-Authored-By: Claude` or equivalent. 28 commits already
   carry one, they will not be rewritten, and no more are to be added.
2. **Never push to `github`.** It is a push mirror with `keep_divergent_refs: false`; a hand push
   is overwritten at the next sync, silently. **Never push to `TUM-Doepfert`** — it is the paper
   fork and holds the only copy of the citable tag.
3. **Check state with `glab`, not with git topology.** Whether an MR is open, merged or superseded
   is not derivable from which commits are reachable from what. A merge that says "replaces !197"
   in its message is the kind of thing that only the API tells you.
4. **CHANGELOG entry, under `[Unreleased]`,** in the section that fits — and a `### Migration`
   note if a user has to do something. Two breaking changes have shipped here that were not
   runtime-detectable; that is why the section exists.
5. **Never commit** `scenarios/`, `results/`, zip archives, or generated figures. Check the diff
   for them rather than trusting `.gitignore`, which has been outrun before — `input_data/*` is
   ignored and 152 files under it are tracked anyway (`git ls-files input_data | wc -l`).
6. **Use a worktree** for anything that takes more than one commit, so the branch you were on
   keeps its working tree. Uncommitted parked work is easy to destroy from an adjacent task.
7. **A pipeline is informational.** The only runner is a laptop, so a missing pipeline is not a
   verdict and durations are not comparable between runs. Read a red one; do not wait on a
   silent one. See `CI_CD_Guide.md`.

## Before opening

```bash
python -m pytest tests -q                          # fast tier; e2e and golden run in CI
git diff --name-only origin/develop...HEAD         # nothing generated, nothing paper-only
git log origin/develop..HEAD --format=%B | grep -i co-authored   # expect no output
```

**Exit criterion:** branch pushed to `origin`, MR opened against `develop` with a description that
states what was verified and how, CHANGELOG entry present, no AI trailers, and the pipeline either
green or explained.
