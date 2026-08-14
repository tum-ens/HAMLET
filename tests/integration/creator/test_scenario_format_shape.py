"""Integration — the shape of a generated scenario, pinned per format version.

A version nobody remembers to bump is worse than no version at all, because it gives false
confidence. This test is the thing that remembers.

It generates the shipped example under a fixed seed and fingerprints the *shape* of the result --
which files exist, which columns each table has, which keys each JSON carries -- with the random
agent and plant ids normalised away. The fingerprint is compared against a reference committed
per format version, `format/scenario_format_v<N>.json`. Change what the Creator writes and this
fails, naming what moved and telling you to bump.

**What it does not catch, stated plainly.** A change in what a column *means* while its name stays
the same is invisible here -- and that is exactly the change that motivated the versioning (the
retailer in/out convention). Nothing structural can see it. What sees it is the golden master, by
the numbers moving. So the bump rule documented at `c.SCENARIO_FORMAT_VERSION` has two halves,
and this test automates one of them.

Fingerprinting only the shape is deliberate: the values are the golden master's job, and a test
that fails for two unrelated reasons tells you less than two tests that each fail for one.
"""
import json
import os
import random
import re
import shutil

import polars as pl
import pytest

import hamlet.constants as c
from hamlet.creator.setup import Creator

EXAMPLE = 'simple_scenario'
SEED = 20260804

REFERENCES = os.path.join(os.path.dirname(__file__), 'format')
UPDATE = 'HAMLET_UPDATE_SCENARIO_FORMAT'

# `f.gen_ids` draws 15 characters from ascii letters and digits. Agent and plant ids appear both
# as folder names and as column-name prefixes, and they are random-but-seeded, so they are
# normalised away: the format is "there is a column `<id>_electricity`", not which id drew it.
ID = re.compile(r'^[A-Za-z0-9]{15}$')

# pandapower's grid export is a large machine-written file whose internal keys are pandapower's
# format, not HAMLET's. Recorded as present, not parsed.
OPAQUE = {'grid.json'}


def reference_path(version):
    return os.path.join(REFERENCES, f'scenario_format_v{version}.json')


def normalise(text):
    """Replace seeded-random ids with a placeholder, in a path segment or a column name."""
    return '_'.join('<id>' if ID.match(part) else part for part in text.split('_'))


def key_paths(data, prefix=''):
    """Every key path through a JSON document, with ids normalised and values ignored."""
    if isinstance(data, dict):
        paths = []
        for key, value in data.items():
            here = f'{prefix}.{normalise(str(key))}' if prefix else normalise(str(key))
            paths.extend(key_paths(value, here) or [here])
        return paths
    if isinstance(data, list):
        return [p for item in data for p in key_paths(item, f'{prefix}[]')]
    return []


def fingerprint(scenario):
    """The shape of a scenario folder: files, and the columns or keys inside them."""
    shape = {}
    for root, _, files in os.walk(scenario):
        for name in sorted(files):
            path = os.path.join(root, name)
            relative = os.path.relpath(path, scenario).replace(os.sep, '/')
            key = '/'.join(normalise(part) for part in relative.split('/'))

            entry = shape.setdefault(key, {'files': 0, 'contents': set()})
            entry['files'] += 1

            if name.endswith('.ft'):
                columns = pl.read_ipc(path, memory_map=False).columns
                entry['contents'].update(normalise(column) for column in columns)
            elif name.endswith('.json') and name not in OPAQUE:
                entry['contents'].update(key_paths(json.loads(
                    open(path, encoding='utf-8').read())))

    return {key: {'files': entry['files'], 'contents': sorted(entry['contents'])}
            for key, entry in sorted(shape.items())}


@pytest.fixture(scope='module')
def actual(tmp_path_factory, repo_root):
    """Generate the shipped example once, seeded. Creator only -- no executor, a few seconds."""
    base = tmp_path_factory.mktemp('format')
    config = base / EXAMPLE
    shutil.copytree(repo_root / 'examples' / 'create_simple_scenario' / EXAMPLE, config)
    (base / 'scenarios').mkdir()
    (base / 'results').mkdir()

    setup = config / 'setup.yaml'
    text = setup.read_text(encoding='utf-8')
    for old, new in (('input: ../../input_data', f'input: {(repo_root / "input_data").as_posix()}'),
                     ('scenarios: ../../scenarios', f'scenarios: {(base / "scenarios").as_posix()}'),
                     ('results: ../../results', f'results: {(base / "results").as_posix()}')):
        assert old in text, f'{old!r} not found in setup.yaml'
        text = text.replace(old, new)
    setup.write_text(text, encoding='utf-8')

    # Seed for a reproducible draw of agents and plants, and put the global state back so the
    # rest of the suite is unaffected
    import numpy as np
    state, np_state = random.getstate(), np.random.get_state()
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        Creator(path=str(config)).new_scenario_from_configs()
    finally:
        random.setstate(state)
        np.random.set_state(np_state)

    try:
        yield fingerprint(base / 'scenarios' / EXAMPLE)
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope='module')
def expected(actual):
    """The reference for the *current* format version, regenerated only when asked for."""
    path = reference_path(c.SCENARIO_FORMAT_VERSION)

    if os.environ.get(UPDATE):
        os.makedirs(REFERENCES, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(actual, indent=2, sort_keys=True) + '\n')
        pytest.skip(f'reference regenerated at {path}. If this is a NEW version, commit it with '
                    f'the bump. If it is an existing one, you have just overwritten the record '
                    f'of what that version looked like -- check that is what you meant.')

    assert os.path.exists(path), (
        f'no scenario-format reference for version {c.SCENARIO_FORMAT_VERSION} at {path}. '
        f'If you have just bumped the version, create it with {UPDATE}=1 python -m pytest '
        f'tests/integration/creator/test_scenario_format_shape.py and commit it with the bump.')

    with open(path, encoding='utf-8') as file:
        return json.load(file)


class TestTheStampIsPresentInAGeneratedScenario:

    def test_the_creator_stamps_every_scenario_it_writes(self, actual):
        assert 'general/general.json' in actual
        assert c.K_SCENARIO_FORMAT_VERSION in actual['general/general.json']['contents']


class TestTheShapeMatchesTheVersion:
    """The bump enforcement. Each assertion reports everything at once, because when the format
    genuinely changes it is the pattern across files that tells you whether it was intended."""

    def test_the_same_files_are_written(self, actual, expected):
        added = sorted(set(actual) - set(expected))
        removed = sorted(set(expected) - set(actual))

        assert not (added or removed), (
            f'the scenario folder no longer contains the same files.\n'
            f'  added:   {added}\n'
            f'  removed: {removed}\n\n'
            f'{_remedy()}')

    def test_the_same_number_of_each_file_is_written(self, actual, expected):
        moved = {key: (entry['files'], expected[key]['files'])
                 for key, entry in actual.items()
                 if key in expected and entry['files'] != expected[key]['files']}

        assert not moved, f'file counts moved (actual, expected): {moved}\n\n{_remedy()}'

    def test_the_same_columns_and_keys_are_written(self, actual, expected):
        differences = []
        for key, entry in sorted(actual.items()):
            if key not in expected:
                continue
            added = sorted(set(entry['contents']) - set(expected[key]['contents']))
            removed = sorted(set(expected[key]['contents']) - set(entry['contents']))
            if added:
                differences.append(f'{key}: added {added}')
            if removed:
                differences.append(f'{key}: removed {removed}')

        assert not differences, (
            'the columns and keys inside the scenario have changed:\n  '
            + '\n  '.join(differences) + f'\n\n{_remedy()}')


class TestTheReferencesAreConsistentWithTheConstant:
    """Catches the two ways the bookkeeping can rot without any scenario being generated."""

    def test_there_is_a_reference_for_the_current_version(self):
        assert os.path.exists(reference_path(c.SCENARIO_FORMAT_VERSION))

    def test_no_reference_exists_for_a_version_that_was_never_declared(self):
        """A committed v2 reference with the constant still at 1 means a bump was half-done."""
        committed = {int(re.search(r'_v(\d+)\.json$', name).group(1))
                     for name in os.listdir(REFERENCES)
                     if re.search(r'_v\d+\.json$', name)}

        ahead = sorted(v for v in committed if v > c.SCENARIO_FORMAT_VERSION)

        assert not ahead, (
            f'reference files exist for scenario format version(s) {ahead}, but '
            f'c.SCENARIO_FORMAT_VERSION is still {c.SCENARIO_FORMAT_VERSION}')

    def test_every_version_up_to_the_current_one_has_a_reference(self):
        """The older files record what those scenarios looked like; they are not disposable."""
        missing = [v for v in range(1, c.SCENARIO_FORMAT_VERSION + 1)
                   if not os.path.exists(reference_path(v))]

        assert not missing, f'no reference committed for scenario format version(s) {missing}'


def _remedy():
    return (
        f'This is a change to the on-disk scenario format. Decide which it is:\n\n'
        f'  * Unintended -- revert it.\n'
        f'  * Intended, and a scenario written before it would still be read correctly '
        f'afterwards (a file the executor ignores, say) -- regenerate the reference for '
        f'version {c.SCENARIO_FORMAT_VERSION} with {UPDATE}=1 and commit it with the change.\n'
        f'  * Intended, and it changes how a scenario is read -- bump '
        f'c.SCENARIO_FORMAT_VERSION to {c.SCENARIO_FORMAT_VERSION + 1}, then create '
        f'tests/integration/creator/format/scenario_format_v{c.SCENARIO_FORMAT_VERSION + 1}.json '
        f'with {UPDATE}=1 and commit both. Leave the older reference alone.\n\n'
        f'The rule, and the case this cannot see, are written out at c.SCENARIO_FORMAT_VERSION.')
