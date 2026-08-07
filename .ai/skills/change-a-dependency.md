# Skill: change-a-dependency

**Use when:** adding, removing, upgrading or unpinning anything in `env.yml`.

**`env.yml` is the only dependency definition.** There is no lockfile, no `pyproject.toml` and no
`requirements.txt`. CI does not carry a second list — `ci/requirements_from_env.py` derives its
requirements from the same file a contributor installs from, because a second list is exactly how
`develop` came to ship an `env.yml` that could not `import hamlet`.

**Some pins are load-bearing and do not look it.** `xarray==2024.6.0` is not HAMLET's dependency
at all; it arrives through linopy. linopy 0.3.11 imports `xarray.core.rolling`, removed after
2024.6.0, so unpinning it means `import hamlet` fails outright — with a traceback pointing at
neither package. The comment at `env.yml:37-41` says so; do not silence it by relaxing the pin.

## Rules

1. **Pin exactly.** Everything in `env.yml` is `==`. An unpinned dependency turns an upstream
   release into a red pipeline on a commit that changed nothing.
2. **Read the comment above the pin before touching it.** Where a pin has a reason, the reason is
   written there.
3. **A transitive dependency that breaks an import belongs in `env.yml`,** even though it is not
   a direct dependency. That is a stopgap for the missing lockfile, and the comment should say so.
4. **If you touch the CI exclusion list**, `tensorflow` / `psycopg2` / `jupyter`, note that two of
   the three exclusions are *claims about the source* — that TensorFlow is imported only inside
   functions, and that `psycopg2` has no usages under `hamlet/`. Breaking either claim breaks CI
   correctly. Verify with `--verify`, not by reasoning.
5. **Verify in a fresh environment, not in yours.** The environment that has worked for a year has
   worked by historical accident at least once already: `hamlet311` carried a working `xarray`
   that a fresh `conda env create` did not resolve to.

## Checks

```bash
python ci/requirements_from_env.py env.yml --verify   # the three exclusions still hold
python -c "import hamlet"                             # the thing that actually broke
python -m pytest tests -q                             # fast suite
```

**Exit criterion:** all three pass in an environment created from the edited `env.yml`, the
`dependency-exclusions` and `fast` CI jobs are green, and any new pin whose reason is not obvious
carries a comment saying what breaks without it.
