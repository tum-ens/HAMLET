# CI/CD

What `.gitlab-ci.yml` actually does, and what its results mean. The file itself carries the
reasoning behind each decision inline; this is the overview.

Until 2026-08-07 this document was a generic template describing install / lint / test / build /
scan / deploy / notify stages. HAMLET has never had a build, a deployment or a security scan, so
the template described a pipeline that did not exist. It has been replaced with a description of
the one that does.

## The pipeline

Two stages, four jobs. GitLab CI only; there are no GitHub Actions workflows, and the GitHub
repository is a push mirror rather than a second CI target.

| Stage | Job | What it does |
|---|---|---|
| `lint` | `lint` | `ruff check --select E9,F63,F7,F82` over `hamlet/` and `tests/` |
| `test` | `fast` | unit + integration, no simulation |
| `test` | `e2e` | the shipped example, Creator → Executor → Analyzer |
| `test` | `golden` | the same example against committed reference numbers |

The test tiers are separate jobs on purpose, so a failure says which *kind* of thing broke rather
than just "tests failed". `golden` is the one that catches a change in results rather than in
behaviour someone thought to assert.

`pytest.ini` deselects `e2e` and `golden` by default, which is why `fast` needs no marker and the
other two pass `-m`. See `tests/README.md` for what each tier covers.

**No solver licence is required.** The example runs on HiGHS, which is installed with HAMLET.

### Lint is deliberately narrow

`E9,F63,F7,F82` is syntax errors, undefined names, and mistakes like `assert (x, y)` — things that
are wrong regardless of taste. The repository has no linter configuration and no agreed style, so
a full rule set would report thousands of findings that say nothing about correctness and would
train everyone to ignore the pipeline. `pyproject.toml` now exists, so moving the rules there and
widening them is unblocked — but that changes what the pipeline enforces, which is its own commit.

### Dependencies come from the lockfile

CI runs `uv sync --locked`, installing the exact versions in `uv.lock` — the same ones a
contributor gets from `uv sync`. There is no CI dependency list and no script that generates one:
`env.yml` and `ci/requirements_from_env.py` are both gone, and a second list is precisely how
`develop` once shipped an `env.yml` that could not import hamlet.

`--locked` rather than `--frozen` on purpose. It fails the job when `pyproject.toml` and `uv.lock`
disagree, so a dependency edited without re-locking cannot reach `develop` and surface later on
somebody else's fresh install.

`tensorflow`, `gurobipy` and `jupyter` are extras, and no job installs them. The
`dependency-exclusions` job that used to assert this is gone, because the claim is now structural
rather than asserted: `fast` imports `hamlet` in an environment without them, so a module-scope
`import tensorflow` under `hamlet/` fails on the commit that introduces it, not in review.

### Three environment variables are load-bearing

`PYTHONHASHSEED=0`, because the golden master compares committed numbers and the Creator draws
agent ids and plant sizings from seeded `random`. `MPLBACKEND=Agg`, because a runner has no
display and the analyzer imports matplotlib. `UV_PROJECT_ENVIRONMENT`, because `uv sync` otherwise
creates `.venv` inside the checkout — see the next section for why that costs three minutes.

## The runner is a laptop

There is one runner, registered to this project, with the Docker executor. Its GitLab description
is "My laptop (pipeline only works if it is running)", and that is accurate: LRZ does not give
this project shared runners, and no group runner is registered.

Two consequences worth stating plainly:

1. **Pipelines are informational, not required.** A pipeline that cannot start when the laptop is
   closed must not be a merge blocker, or work stops for reasons unrelated to the work. Read a red
   pipeline; do not treat a missing one as a verdict.
2. **Job durations are not comparable between runs.** Across four consecutive pipelines on
   2026-08-07 the `fast` job took 184 s, 1466 s, 1936 s and 2067 s — the same commit range, the
   same warm cache, an order of magnitude apart. The variance is disk contention on a laptop that
   is also doing other things. Use durations to spot a job that hangs, not to detect a performance
   regression.

## Why installs go through uv

Measured on this runner, same packages, warm cache, one timed install per fresh container:

| | |
|---|---|
| `pip install` | 1005 s |
| `uv pip install --system` | 223 s |
| `uv venv` on `/cache` + install | 1 s |

pip was never network-bound here — every trace showed a full cache hit and zero downloads. The
cost was writing 1.2 GB across ~24,000 files, so the fix is to stop writing them. **The venv's
location is the mechanism, not uv alone**: uv hardlinks out of its cache only when the target is on
the same filesystem, and `--system` writes into the image layer, which is a different one.

This is why `.python` exports `UV_PROJECT_ENVIRONMENT="/cache/venv-$CI_JOB_ID"` before running
`uv sync`. Left to itself, `uv sync` puts the environment at `.venv` in the checkout — a different
filesystem from `UV_CACHE_DIR`, which is exactly the 1 s → 223 s case in the table. Anything that
"simplifies" that export away silently gives back the win. Per job id, so concurrent jobs cannot
race over one venv; the files are hardlinks, so the duplication costs almost no disk.

So `UV_CACHE_DIR` and the venv both live on `/cache`, a volume the runner mounts into every job
container. That is a per-runner cache rather than a GitLab `cache:` entry, which would tar and
untar 1.2 GB per job to protect an install that now takes a second. A fresh runner pays the full
cost once; every job afterwards does not.

Each venv is keyed by `$CI_JOB_ID` so concurrent jobs cannot race, and removed in `after_script`.
This costs almost no disk, because its contents are hardlinks into the cache.

## Requirements

CI/CD must be enabled for the project (Settings → General → Visibility) and a runner with the
Docker executor must be registered. Neither is on by default. Both have been in place since
2026-08-07.
