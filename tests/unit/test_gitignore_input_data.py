"""Unit -- `input_data/` is tracked source and must stay visible to `git status`.

`.gitignore` carried `input_data/*` while 152 files under it were tracked anyway. Ignore rules do
not untrack anything, so nothing looked wrong until someone *added* an input: it never appeared in
`git status`, and the omission surfaced only when someone else could not run what needed it. The
benchmark input `energy_da_raw.csv` went that way.

Re-adding the rule looks like tidying, which is why this asserts the behaviour rather than the
absence of a line. `git check-ignore` answers for paths that need not exist, so nothing is written.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files that do not exist. `git check-ignore` matches patterns against the path, so these probe
# the rules without touching the tree. One per subtree that carries tracked inputs today.
PROBES = [
    'input_data/agents/sfh/ev/ev_fulltime_99.csv',
    'input_data/agents/industry/heat/heat_new_profile.csv',
    'input_data/retailers/lem/energy_da_raw.csv',   # the file this actually happened to
    'input_data/general/weather/weather_2024.ft',
    'input_data/markets/lem/some_new_market.json',
]


def ignoring_rule(path):
    """The `.gitignore` rule matching `path`, or None if nothing ignores it.

    `git check-ignore -v` exits 0 with `source:line:pattern\tpath` when a rule matches, and 1
    with no output when none does. Any other exit is a broken invocation, not an answer.
    """
    result = subprocess.run(
        ['git', 'check-ignore', '-v', '--no-index', path],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode == 1:
        return None
    if result.returncode == 0:
        return result.stdout.strip()
    raise RuntimeError(f'git check-ignore failed for {path!r}: {result.stderr.strip()}')


@pytest.fixture(scope='module', autouse=True)
def requires_a_git_checkout():
    """Skip where the question cannot be asked, rather than inventing an answer."""
    if shutil.which('git') is None:
        pytest.skip('git is not on PATH')
    if not (REPO_ROOT / '.git').exists():
        pytest.skip(f'{REPO_ROOT} is not a git checkout (installed package or unpacked sdist)')


@pytest.mark.parametrize('path', PROBES)
def test_a_new_input_file_is_not_silently_ignored(path):
    """Adding an input must show up in `git status`. This is the whole point of the change."""
    rule = ignoring_rule(path)
    assert rule is None, (
        f'`{path}` is ignored by `{rule}`. An ignore rule over `input_data/` does not untrack the '
        f'152 files already committed, it only hides the next one. Ignore the specific generated '
        f'artefact instead.')


def test_the_tracked_input_files_are_still_tracked():
    """Catches the opposite mistake: untracking the inputs instead of dropping the rule."""
    result = subprocess.run(['git', 'ls-files', '--', 'input_data'],
                            cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    tracked = [line for line in result.stdout.splitlines() if line]
    assert len(tracked) > 100, (
        f'only {len(tracked)} files tracked under input_data/; the examples and the golden master '
        f'read these time series. If they were moved out deliberately, update .ai/context.md too.')


def test_generated_output_directories_are_still_ignored():
    """Dropping the input rule must not take the rules that earn their place with it."""
    for path in ('scenarios/some_run/setup.yaml', 'results/some_run/agents.ft'):
        assert ignoring_rule(path) is not None, f'`{path}` is run output and must stay ignored.'


def test_archives_are_ignored_anywhere_in_the_tree():
    """Bulk data is what the blanket `input_data/*` rule was really guarding against."""
    for path in ('V2.zip', 'input_data/agents/sfh/raw.rar', 'docs/big.tar.gz'):
        assert ignoring_rule(path) is not None, (
            f'`{path}` is not ignored. With `input_data/` visible, these rules are what keep a '
            f'multi-GB archive one `git add -A` away from the index.')
