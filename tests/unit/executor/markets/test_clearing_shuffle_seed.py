"""Unit — the bid/offer shuffle's seed, which is the market half of #216.

`__split_bids_offers` shuffles bids and offers before sorting them on price, so that agents tied
on price do not always clear in the same order. That shuffle was unseeded, and an unseeded
`DataFrame.sample` is not merely arbitrary — polars draws its seed from Python's global `random`,
which HAMLET never seeds, so **every process shuffled differently**. Two identical runs then
cleared tied bids differently.

Measured on the paper's design 6 at 150 steps with the hash seed left random: 45 of 45 run pairs
disagreed, always in the same 6 result files; the ten runs formed ten mutually-distinct groups.
With the seed below, 15 of 15 pairs are identical.

The seed must satisfy two properties that pull against each other, and both are pinned here:

* **stable across processes**, or the run is not reproducible;
* **different for each clearing**, or the shuffle no longer removes the bias it exists to remove.
"""
import json
import os
import subprocess
import sys

import polars as pl
import pytest

import hamlet.constants as c
from hamlet.executor.markets.electricity import ElectricityMarket

# The seed for the fixture below, pinned. This is the cheap half of the cross-process guard: a
# `hash()`-based implementation produces a different value in every interpreter, so it fails this
# in-process, on the first run, without paying for a subprocess.
EXPECTED = {'in': 844758340, 'out': 1477584029}

# One subprocess, not one per seed: `import hamlet` costs ~14 s, and a single interpreter running
# under a different PYTHONHASHSEED is enough to demonstrate the property, given EXPECTED above
# already pins the value.
OTHER_HASH_SEED = '7'

CHILD = """
import json
import hamlet.constants as c
from hamlet.executor.markets.electricity import ElectricityMarket
market = object.__new__(ElectricityMarket)
market.tasks = {c.TC_REGION: 'r1', c.TC_MARKET: 'electricity',
                c.TC_NAME: 'intraday', c.TC_TIMESTEP: '2023-01-01T00:00:00'}
print(json.dumps({'in': market._clearing_seed(c.PF_IN),
                  'out': market._clearing_seed(c.PF_OUT),
                  'builtin_hash': hash('electricity')}))
"""


def make_market(timestep='2023-01-01T00:00:00', name='intraday', region='r1'):
    """A market instance carrying only what `_clearing_seed` reads.

    `__init__` hits the database, so it is bypassed deliberately: the seed is a pure function of
    `tasks`, and this drives the real method on the real class.
    """
    market = object.__new__(ElectricityMarket)
    market.tasks = {c.TC_REGION: region, c.TC_MARKET: 'electricity',
                    c.TC_NAME: name, c.TC_TIMESTEP: timestep}
    return market


@pytest.fixture(scope='module')
def other_process():
    """The same seed computed in a second interpreter under a different hash seed."""
    env = dict(os.environ, PYTHONHASHSEED=OTHER_HASH_SEED)
    completed = subprocess.run([sys.executable, '-c', CHILD],
                               capture_output=True, text=True, env=env, check=True)
    return json.loads(completed.stdout.splitlines()[-1])


def test_the_other_process_really_does_hash_differently(other_process):
    """Instrument check: without it the comparison below could pass while proving nothing.

    If `PYTHONHASHSEED` were not taking effect, a `hash()`-based seed would look stable across
    the two interpreters and this file would licence exactly the defect it exists to prevent.
    """
    assert other_process['builtin_hash'] != hash('electricity'), (
        'the child interpreter hashed strings the same way the parent does, so this file cannot '
        'distinguish a stable digest from a process-local one')


def test_the_seed_is_identical_in_another_process(other_process):
    """The reproducibility property. A `hash()`-based implementation fails this."""
    market = make_market()

    assert other_process['in'] == market._clearing_seed(c.PF_IN)
    assert other_process['out'] == market._clearing_seed(c.PF_OUT)


def test_the_seed_is_the_pinned_value():
    """Catches a `hash()`-based regression in-process, and pins the digest against silent drift."""
    market = make_market()

    assert market._clearing_seed(c.PF_IN) == EXPECTED['in']
    assert market._clearing_seed(c.PF_OUT) == EXPECTED['out']


def test_bids_and_offers_get_different_seeds():
    """Otherwise both sides of the book receive the same permutation."""
    market = make_market()

    assert market._clearing_seed(c.PF_IN) != market._clearing_seed(c.PF_OUT)


@pytest.mark.parametrize('field, other', [
    ('timestep', {'timestep': '2023-06-15T12:00:00'}),
    ('market name', {'name': 'dayahead'}),
    ('region', {'region': 'r2'}),
], ids=['timestep', 'name', 'region'])
def test_the_seed_varies_per_clearing(field, other):
    """The anti-bias property: a constant seed would be reproducible and useless.

    With one fixed permutation for the whole run, agents tied on price would clear in the same
    order at every timestep — which is the bias the shuffle was added to remove.
    """
    base = make_market()
    changed = make_market(**other)

    assert base._clearing_seed(c.PF_IN) != changed._clearing_seed(c.PF_IN), (
        f'the seed does not depend on the {field}, so every clearing shares one permutation')


def test_the_seed_is_in_range_for_polars():
    """polars rejects a seed outside u64; `digest_size=4` keeps it comfortably inside."""
    seed = make_market()._clearing_seed(c.PF_IN)

    assert isinstance(seed, int) and 0 <= seed < 2 ** 32
    pl.DataFrame({'a': [1, 2, 3]}).sample(fraction=1, shuffle=True, seed=seed)


def test_the_same_seed_reproduces_the_same_shuffle():
    """The end of the chain: equal seeds must give equal orders, or none of the above matters."""
    frame = pl.DataFrame({'agent': [f'a{i}' for i in range(50)], 'price': [100] * 50})
    seed = make_market()._clearing_seed(c.PF_IN)

    first = frame.sample(fraction=1, shuffle=True, seed=seed)['agent'].to_list()
    second = frame.sample(fraction=1, shuffle=True, seed=seed)['agent'].to_list()

    assert first == second
    assert first != frame['agent'].to_list(), 'the shuffle did not actually reorder anything'


def test_the_shuffle_actually_receives_the_seed(monkeypatch):
    """Everything above tests the seed; this tests that the shuffle *uses* it.

    Without this, deleting `seed=` from the two `sample` calls while leaving `_clearing_seed`
    in place reintroduces the whole defect with every other check in this file still green —
    verified by mutation, which is why it is here.

    It drives the real `__split_bids_offers` and records what `sample` was actually called with.
    """
    calls = []
    original = pl.DataFrame.sample

    def recording_sample(self, *args, **kwargs):
        calls.append(kwargs)
        kwargs.setdefault('seed', 0)      # keep the call deterministic for the test itself
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, 'sample', recording_sample)

    market = make_market()
    empty = pl.DataFrame(schema={c.TC_ID_AGENT_IN: pl.Utf8})
    market.bids_cleared = empty
    market.offers_cleared = pl.DataFrame(schema={c.TC_ID_AGENT_OUT: pl.Utf8})
    bids_offers = pl.DataFrame({
        c.TC_ID_AGENT: ['a1', 'a2', 'a3'],
        c.TC_ENERGY_IN: [10, 0, 5],
        c.TC_ENERGY_OUT: [0, 7, 0],
        c.TC_PRICE_PU_IN: [100, 0, 120],
        c.TC_PRICE_PU_OUT: [0, 90, 0],
    })

    market._ElectricityMarket__split_bids_offers(bids_offers, add_cumsum=False)

    assert len(calls) == 2, f'expected one shuffle for bids and one for offers, got {len(calls)}'
    seeds = [call.get('seed') for call in calls]
    assert seeds == [market._clearing_seed(c.PF_IN), market._clearing_seed(c.PF_OUT)], (
        f'the shuffle was not given the per-clearing seed: got {seeds}. An unseeded `sample` '
        f'draws from Python global `random`, so every process would shuffle differently (#216)')
