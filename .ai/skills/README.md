# Skill cards

One card per recurring task where doing the obvious thing produces a plausible, wrong result.

A card is a **router, not a manual**. It names the situation, points at the canonical procedure,
states only the rules that are costly to rediscover, and ends with an exit criterion you can
actually run. If a card starts explaining how HAMLET works, that belongs in `../context.md` or in
`tests/README.md` instead.

Every rule in a card should be traceable to something that actually went wrong. A card full of
generic good advice trains people to skip cards.

| Card | Use when |
|---|---|
| [`verify-a-claim.md`](verify-a-claim.md) | about to write a factual claim into a tracked file, a commit message or an MR |
| [`golden-master-failed.md`](golden-master-failed.md) | `pytest -m golden` fails, or you expect it to |
| [`write-a-regression-test.md`](write-a-regression-test.md) | fixing a defect, or adding a test to an existing one |
| [`change-a-dependency.md`](change-a-dependency.md) | adding, removing, or moving a pin in `pyproject.toml` |
| [`review-a-change.md`](review-a-change.md) | a branch is ready — run the reviewer panel before opening |
| [`open-a-merge-request.md`](open-a-merge-request.md) | work is finished and needs to reach `develop` |

## Adding a card

Add one when you catch yourself re-deriving a procedure you have already derived once, or when a
review finds the same class of mistake twice. Delete one when the trap it describes is gone —
a card describing infrastructure that no longer exists is the failure mode this whole directory
was built to prevent.

Cards are tool-agnostic markdown. Nothing enforces them; they work because `../context.md` links
them and every tool's pointer file leads there.
