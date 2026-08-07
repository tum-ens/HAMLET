"""Unit — the Creator's input-file selection must not depend on `os.listdir` order.

The Creator picks each agent's load, PV, heat and EV profile by drawing an *index* from a seeded
RNG and using it to subscript `os.listdir(input_path)`. The seed fixes the index; the filesystem
fixes the order. `os.listdir` returns alphabetical order on NTFS and hash order on ext4/overlayfs,
so the same configuration and the same seed produced **different scenarios on Windows and Linux** —
same agent ids and plants, different profiles behind them.

That went unnoticed because every Linux run of the suite until now mounted a checkout that Windows
git had materialised; the first genuinely Linux-native checkout (a GitLab runner) moved 178 golden
statistics, including `forecasts.ft`, which the solver cannot touch.

These tests pin the property directly rather than through the golden master, so the next unsorted
`os.listdir` fails here — naming the function — instead of surfacing as unexplained numbers months
later.
"""
import os
import random

import pytest

from hamlet.creator.agents.agents import Agents

# Deliberately not in alphabetical order, and deliberately including the case-ordering trap: a
# lowercase name sorts after uppercase ones in Python but before them on NTFS.
#
# 3648 and 4352 are equidistant from 4000 on purpose. `_get_closest` is
# `min(enumerate(...), key=...)`, which returns the *first* minimum in iteration order, so a tie is
# the only way `_pick_files_by_values` can notice the order at all -- without one it matches by
# value and looks order-independent even when it is not.
FILENAMES = [
    'hh_3648_0.csv', 'hh_2012_3.csv', 'hh_2532_0.csv', 'hh_2262_0.csv',
    'hh_4711_1.csv', 'hh_2455_0.csv', 'Hh_9000_0.csv', 'hh_4352_0.csv',
]


@pytest.fixture
def input_path(tmp_path):
    """A directory of profile files, created in a deliberately unhelpful order."""
    for name in FILENAMES:
        (tmp_path / name).write_text('timestamp,power\n', encoding='utf-8')
    return str(tmp_path)


def shuffled_listdir(order):
    """A stand-in for `os.listdir` that returns `order`, to imitate a different filesystem."""
    def _listdir(path):
        return list(order)
    return _listdir


def pick_at_random(monkeypatch, input_path, order, seed=1234):
    monkeypatch.setattr(os, 'listdir', shuffled_listdir(order))
    random.seed(seed)
    return Agents._pick_files_at_random(list_owner=[1] * 6, input_path=input_path)


def pick_by_values(monkeypatch, input_path, order):
    monkeypatch.setattr(os, 'listdir', shuffled_listdir(order))
    # 4000 is the tie between 3648 and 4352; the others are unambiguous.
    return Agents._pick_files_by_values(vals=[2500, 4000, 4700], input_path=input_path)


def pick_from_distr(monkeypatch, input_path, order, seed=1234):
    monkeypatch.setattr(os, 'listdir', shuffled_listdir(order))
    random.seed(seed)
    return Agents._pick_files_from_distr(
        list_owner=[1] * 6, distr=[1, 1, 1], vals=[2000, 3000, 4000],
        input_path=input_path, variance=[2000, 2000, 2000])


@pytest.mark.parametrize('pick', [pick_at_random, pick_by_values, pick_from_distr],
                         ids=['at_random', 'by_values', 'from_distr'])
def test_the_same_seed_picks_the_same_files_whatever_order_the_filesystem_returns(
        monkeypatch, input_path, pick):
    """The property the golden master depends on and could not express."""
    forward = pick(monkeypatch, input_path, FILENAMES)
    reversed_ = pick(monkeypatch, input_path, list(reversed(FILENAMES)))
    alphabetical = pick(monkeypatch, input_path, sorted(FILENAMES))

    assert forward == reversed_ == alphabetical, (
        'the files chosen depend on os.listdir order, so the same configuration and seed '
        'generate different scenarios on different filesystems')


def test_the_selection_is_the_one_a_sorted_listing_would_give(monkeypatch, input_path):
    """Pins *which* order, not merely that one exists.

    Without this, sorting the listing in one place and not another would still satisfy the test
    above while quietly changing results.
    """
    monkeypatch.setattr(os, 'listdir', shuffled_listdir(FILENAMES))
    random.seed(1234)
    actual = Agents._pick_files_at_random(list_owner=[1] * 6, input_path=input_path)

    random.seed(1234)
    expected = [random.choice(sorted(FILENAMES)) for _ in range(6)]

    assert actual == expected


def test_get_types_is_ordered(monkeypatch, tmp_path):
    """`__get_types` went through a set as well as a listdir — two unordered steps, not one."""
    for name in ('sfh_1.csv', 'mfh_1.csv', 'ctsp_1.csv', 'industry_1.csv'):
        (tmp_path / name).write_text('', encoding='utf-8')

    # Name-mangled: defined as __get_types on Agents.
    get_types = getattr(Agents, '_Agents__get_types')

    monkeypatch.setattr(os, 'listdir',
                        shuffled_listdir(['sfh_1.csv', 'mfh_1.csv', 'ctsp_1.csv', 'industry_1.csv']))
    forward = get_types(str(tmp_path))
    monkeypatch.setattr(os, 'listdir',
                        shuffled_listdir(['industry_1.csv', 'ctsp_1.csv', 'mfh_1.csv', 'sfh_1.csv']))
    backward = get_types(str(tmp_path))

    assert forward == backward == sorted(forward)
