"""`create_folder` must sleep only where the sleep is for something.

The function ended with an unconditional `time.sleep(0.01)`, present since the file was written
in 2023 with no comment. The executor's market stage calls it three times per timestep to create
directories that already exist, so on the paper's design 6 it was **0.039 s per timestep** -- 60 %
of that stage, and about 1 % of a whole timestep, spent sleeping.

The plausible reason for it is the one case it is now limited to: on Windows a directory that
`shutil.rmtree` has just removed can still be reported as present, so recreating it immediately
can raise. That path keeps the sleep. The other two -- create-if-missing, and do-nothing-because-
it-exists -- cannot hit that race and no longer pay for it.

The tests assert on whether `time.sleep` is called rather than on elapsed time, because a timing
assertion would be flaky on a loaded machine and would not say *why* it passed.
"""

import os

import pytest

from hamlet import functions as f


@pytest.fixture
def sleeps(monkeypatch):
    """Record every `time.sleep` the function makes, without actually waiting."""
    recorded = []
    monkeypatch.setattr(f.time, 'sleep', lambda seconds: recorded.append(seconds))
    return recorded


def test_creating_a_missing_folder_does_not_sleep(tmp_path, sleeps):
    target = tmp_path / 'fresh'

    f.create_folder(str(target))

    assert target.is_dir()
    assert sleeps == []


def test_an_existing_folder_left_alone_does_not_sleep(tmp_path, sleeps):
    """`delete=False` on an existing folder is the executor's hot path: it does nothing at all."""
    target = tmp_path / 'existing'
    target.mkdir()
    (target / 'keep.txt').write_text('kept')

    f.create_folder(str(target), delete=False)

    assert (target / 'keep.txt').read_text() == 'kept', 'contents must survive'
    assert sleeps == []


def test_recreating_a_folder_still_sleeps(tmp_path, sleeps):
    """The one case the sleep is for keeps it."""
    target = tmp_path / 'replaced'
    target.mkdir()
    (target / 'old.txt').write_text('gone')

    f.create_folder(str(target), delete=True)

    assert target.is_dir()
    assert not (target / 'old.txt').exists(), 'delete=True must empty the folder'
    assert sleeps == [0.01]


def test_nested_paths_are_created(tmp_path, sleeps):
    """`os.makedirs`, not `os.mkdir`: the executor passes several levels at once."""
    target = tmp_path / 'a' / 'b' / 'c'

    f.create_folder(str(target))

    assert target.is_dir()
    assert sleeps == []


def test_the_hot_path_sleeps_no_matter_how_often_it_is_called(tmp_path, sleeps):
    """What the executor actually does: the same folder, over and over, for a whole run."""
    target = tmp_path / 'past_data'

    for _ in range(50):
        f.create_folder(str(target), delete=False)

    assert target.is_dir()
    assert sleeps == [], f'{len(sleeps)} sleeps for 50 calls that had nothing to do'


def test_delete_true_on_a_missing_folder_does_not_sleep(tmp_path, sleeps):
    """There is nothing to race with when there was nothing to remove."""
    target = tmp_path / 'absent'

    f.create_folder(str(target), delete=True)

    assert target.is_dir()
    assert sleeps == []


def test_it_is_still_usable_for_real(tmp_path):
    """One case without the monkeypatch, so the tests cannot all pass against a broken function."""
    target = tmp_path / 'real' / 'nested'

    f.create_folder(str(target), delete=False)
    f.create_folder(str(target), delete=False)

    assert os.path.isdir(target)
