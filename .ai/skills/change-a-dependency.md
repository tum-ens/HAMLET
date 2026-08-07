# Skill: change-a-dependency

**Use when:** adding, removing, upgrading or unpinning anything in `pyproject.toml`, or moving a
package between core, an extra and a dependency group.

**`pyproject.toml` plus `uv.lock` are the only dependency definition.** There is no `env.yml`, no
`requirements.txt`, no `docs/requirements.txt` and no generator script; CI installs from the same
lock a contributor does. Each of those files existed once, and a second list is exactly how
`develop` came to ship an environment that could not `import hamlet`.

**Some pins are load-bearing and do not look it.** `xarray==2024.6.0` is not HAMLET's dependency at
all; it arrives through linopy. linopy 0.3.11 imports `xarray.core.rolling`, removed after
2024.6.0, so unpinning it means `import hamlet` fails outright — with a traceback pointing at
neither package. It is written into `pyproject.toml` rather than left to the lock so that
`uv lock --upgrade` cannot move it and a linopy bump fails resolution loudly. The comment above the
pin says so; do not silence it.

## Rules

1. **Pin exactly.** Everything in `[project.dependencies]` is `==`, matching the environment the
   committed golden master was measured in. This is a deliberate stage, not tidiness: it means a
   change in *results* has one candidate cause. Relaxing the pins to ranges is its own measured
   step, and so is moving off `requires-python = ">=3.11,<3.12"`.
2. **Read the comment above the pin before touching it.** Where a pin has a reason, the reason is
   written there — `xarray` and `psutil` both have one.
3. **Edit `pyproject.toml`, then `uv lock`, and commit the lock in the same commit.** CI runs
   `uv sync --locked` and fails if the two disagree, which is the point; do not work around it with
   `--frozen`.
4. **A transitive dependency that breaks an import belongs in `[project.dependencies]`,** even
   though nothing in `hamlet/` imports it. The lock alone would record the working version without
   preventing the broken one.
5. **Ask whether it is core at all.** If it is imported only inside a function it is an extra
   (`tensorflow`); if only tests or tooling need it, it is a dependency group (`test`, `lint`,
   `docs`). Anything imported at module scope under `hamlet/` is core, including `sktime`.
6. **A wheel must exist for cp311 on Linux and Windows.** `psutil==5.9.0` had none, so installing
   it meant compiling it — which is why it moved to 5.9.4. Check before pinning something old.
7. **Verify in a fresh environment, not in yours.** A long-lived environment has worked by
   historical accident here at least once: `hamlet311` carried a working `xarray` that a fresh
   install did not resolve to.

## Checks

```bash
uv lock                                  # regenerate; commit the diff
uv sync --locked                         # the environment CI will build
uv run python -c "import hamlet"         # the thing that actually broke
uv run python -m pytest -q               # fast tier
uv run python -m pytest -q -m golden     # results unchanged
```

Read `uv.lock`'s diff, not just `pyproject.toml`'s. Bumping one direct pin can move a dozen
transitives, and the golden master is the only thing that will tell you whether that mattered.

**Exit criterion:** all of the above pass in an environment created from the edited files, the
`lint`, `fast`, `e2e` and `golden` CI jobs are green, and any new pin whose reason is not obvious
carries a comment saying what breaks without it.
