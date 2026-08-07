# Skill: review-a-change

**Use when:** a branch is ready and before [`open-a-merge-request`](open-a-merge-request.md). Also
when asked to review a diff, a branch or an MR.

**Method:** a panel of **independent** reviewers, one per lens, each blind to the others, followed
by an **adversarial** pass that tries to refute what they found. Independence is the mechanism —
reviewers who can see each other's findings converge on them, and a converged panel is one
reviewer wearing several hats.

**Why a panel here.** The empirical record in this repository is that the defects that mattered
were caught by review and by ad-hoc measurement, and **not once** by the suite going green. Seven
defects were introduced during the paper-fix port; every one was caught by a reader or a
measurement, two of them only after being pushed. The retailer convention break got past 56
passing tests because every test used a hand-written fixture. A panel is how you get several
readers when there is one of you.

## Scale to what the diff touches, not to how big it is

Line count is a bad proxy. A three-line change to an optimisation bound moves every scenario; a
four-hundred-line plotter change cannot.

| The diff touches | Panel |
|---|---|
| Nothing under `hamlet/` — docs, CI, test config | **1 lens: claims.** Verify every factual statement; that is the whole risk surface |
| `hamlet/analyzer/`, plotting, or code no execution path reaches | **2 lenses** (claims, hygiene), single verification pass |
| Anything the Creator or Executor runs — controllers, markets, grids, database, agents | **Full panel**, adversarial verification on every finding |
| Executor core, `Database`, controller components, or anything that moved the golden master | **Full panel + a completeness critic.** The golden master must have been *run*, and a move explained rather than merely regenerated |

## The lenses

Each is a distinct question, and each exists because of a defect that actually shipped or nearly
did. Give each reviewer exactly one.

1. **Numbers.** Can this move results? Was `pytest -m golden` run, and if it moved, is the move
   explained and the Creator-vs-Executor question answered? See
   [`golden-master-failed`](golden-master-failed.md).
2. **Convention and units.** Does anything map a retailer column onto a transaction or forecast
   column without crossing in↔out? Wh where W is expected? A magnitude nobody derived? Precedents:
   fees landing on feed-in, EV sizing 1000× too large, a market bound 4× too tight because a Wh→W
   conversion was dropped.
3. **Determinism.** Any seeded draw over an unordered iteration — `os.listdir`, `set`, dict order.
   Anything whose result could differ between filesystems, platforms or runs. Four unsorted
   `listdir` sites are known and unchecked; `.ai/context.md` names them.
4. **Tests.** Not "are there tests" but **would any of them fail against the unfixed code**. Check
   by mutation, not by reading. A production change no test notices is a coverage gap, and saying
   so is a finding. See [`write-a-regression-test`](write-a-regression-test.md).
5. **Claims.** Commit messages, CHANGELOG, docstrings, comments, MR description — true against
   *this* tree? See [`verify-a-claim`](verify-a-claim.md); the failure modes it lists are all
   things a reviewer can catch cheaply and an author cannot.
6. **Lineage and hygiene.** Paper-only content leaking in; constants tuned to one scenario;
   generated artefacts, `scenarios/`, `results/`, zips; secrets; AI attribution trailers.

## Rules

- **At least one lens must execute something.** A panel that only reads the diff produces
  plausible findings, which is the failure mode of AI review. Lenses 1 and 4 require running.
- **Adversarial verification.** Every finding goes to a reviewer that did not produce it, asked to
  **refute** it, defaulting to refuted when uncertain. A finding with no concrete failing input —
  values in, wrong value or crash out — is a *question*, not a finding. File it as one.
- **Producer is never the judge.** The lens that raised a finding does not decide whether it
  survives, and the author of the code does not either.
- **Tag severity: Blocker / Major / Minor.** A Blocker or Major needs a human to confirm; an
  AI-only pass never closes one out.
- **Read-only.** Fixing invalidates the pass. Fix, then re-review the new tip.
- **Report what you did not cover.** If a lens was skipped or a check truncated, say so. Silence
  reads as "covered", and this repository already has one recorded case of a truncated tool
  message being quoted as a result.

## Mechanics

Where the tool can spawn subagents, run the lenses **in parallel** and give each only the diff and
its own question — not the other lenses' output, and not a running summary. Where it cannot, run
them as separate sequential passes and reset context between them; that is weaker, because
findings bleed forward, so say in the MR that the panel was sequential.

**Exit criterion:** every surviving finding is in the MR description, including the ones accepted
with reasons; questions and out-of-scope items are filed as issues rather than dropped; Blockers
and Majors are confirmed by a human; and the lenses that were skipped are named. Nothing enforces
any of this — there is no review-evidence job here — which is exactly why it is written down.
