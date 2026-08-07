# CI/CD

What `.gitlab-ci.yml` actually does, and what its results mean. The file itself carries the
reasoning behind each decision inline; this is the overview.

Until 2026-08-07 this document was a generic template describing install / lint / test / build /
scan / deploy / notify stages. HAMLET has never had a build, a deployment or a security scan, so
the template described a pipeline that did not exist. It has been replaced with a description of
the one that does.

## The pipeline

Two stages, five jobs. GitLab CI only; there are no GitHub Actions workflows, and the GitHub
repository is a push mirror rather than a second CI target.

| Stage | Job | What it does |
|---|---|---|
| `lint` | `lint` | `ruff check --select E9,F63,F7,F82` over `hamlet/`, `tests/`, `ci/` |
| `lint` | `dependency-exclusions` | re-checks the three packages CI omits from `env.yml` |
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
train everyone to ignore the pipeline. Widen it when a `pyproject.toml` exists and the rules can
live in the repository rather than in the CI file.

### Dependencies are derived, never copied

`ci/requirements_from_env.py` generates the CI requirements from `env.yml`. A second dependency
list is precisely how `develop` once shipped an `env.yml` that could not import hamlet. CI omits
`tensorflow`, `psycopg2` and `jupyter`; two of those omissions are claims about the source rather
than about size, so `dependency-exclusions` fails if either stops being true.

### Two environment variables are load-bearing

`PYTHONHASHSEED=0`, because the golden master compares committed numbers and the Creator draws
agent ids and plant sizings from seeded `random`. `MPLBACKEND=Agg`, because a runner has no
display and the analyzer imports matplotlib.

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
