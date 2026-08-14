# Skill: open-a-merge-request

**Use when:** work is finished and needs to reach `develop`.

**Read and follow:** `CONTRIBUTING.md` for branch naming (`type/short-description`, with the issue
number leading the description when there is one) and the workflow. `.ai/context.md`
§"Branches and propagation" for what is protected and what is mirrored.

**First:** run [`review-a-change`](review-a-change.md). The panel is read-only and fixing
invalidates it, so it belongs before the MR exists rather than after.

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
5. **Put `Closes #<n>` in the MR *description*,** for every issue the change resolves, near the
   top so a reader sees it without expanding. GitLab then closes the issue on merge: `develop`
   **is** the default branch and autoclose is on, so the mechanism works and has simply not been
   used. #195 and #200 each sat open for a week after the work that fixed them had merged, and
   were closed by hand. Note that the keyword in a *commit* message does not do it — GitLab reads
   the MR description. `CONTRIBUTING.md` §5 has said this all along, which is the point: the rule
   existed, this card did not repeat it, and it was skipped every time.
6. **Never commit** `scenarios/`, `results/`, zip archives, or generated figures. Read the diff
   for them rather than trusting `.gitignore` — it has been outrun before. Conversely,
   `input_data/` **is** tracked source and is no longer ignored, so a new input file there is
   supposed to appear in `git status`; commit it rather than assuming it is noise.
7. **Use a worktree** for anything that takes more than one commit, so the branch you were on
   keeps its working tree. Uncommitted parked work is easy to destroy from an adjacent task.
8. **A pipeline is informational.** The only runner is a laptop, so a missing pipeline is not a
   verdict and durations are not comparable between runs. Read a red one; do not wait on a
   silent one. See `CI_CD_Guide.md`.

## Before opening

```bash
python -m pytest tests -q                          # fast tier; e2e and golden run in CI
git diff --name-only origin/develop...HEAD         # nothing generated, nothing paper-only
git log origin/develop..HEAD --format=%B | grep -i co-authored   # expect no output
```

## After opening

```bash
glab api projects/tum-ens%2FHAMLET/merge_requests/<iid>/closes_issues   # expect the issues you named
```

An empty list means the `Closes #<n>` keyword did not take and the issue will still be open
after the merge. This is the one part of rule 5 that can be checked rather than assumed.

**Exit criterion:** branch pushed to `origin`, MR opened against `develop` with a description that
states what was verified and how, every resolved issue named with `Closes #<n>` and confirmed via
`closes_issues`, CHANGELOG entry present, no AI trailers, and the pipeline either green or
explained.
