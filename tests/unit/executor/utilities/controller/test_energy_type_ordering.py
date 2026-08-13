"""Unit — the energy-type ordering that fixes every agent model's row and column order.

`derive_energy_types` is what both controller bases use to enumerate an agent's energy types, and
all four backends iterate that result in `add_balance_constraints` to add one balance constraint
and one `{energy_type}_{direction}_slack` variable pair each. So its order *is* the optimisation
model's row and column order.

It used to be a `set`. Python randomises string hashing per process, so the same agent's model was
handed to the solver with its rows and columns permuted from run to run, `update_socs` carried the
difference into the next timestep, and two identical runs of the same scenario produced different
results (issue #216). The usual explanation — the permuted model is degenerate enough that the
solver returns a different equally-optimal vertex — is interpretation; what was measured is that
sorting this one set removes the divergence.

**The ordering checks have to run across processes.** Hash randomisation is fixed for the life of
a process, so a same-process test cannot see the defect at all — every call within one interpreter
agrees with every other, set or not. The three ordering checks therefore drive the real function
in subprocesses under several `PYTHONHASHSEED` values; the two that only concern the return value
(`test_it_returns_a_sequence_rather_than_a_set`, `test_an_empty_mapping_is_not_an_error`) are
ordinary in-process checks, because nothing about them depends on the seed.
"""
import json
import os
import subprocess
import sys

import pytest

# More than one energy type, or there is nothing to permute -- a one-element set has one ordering,
# so only multi-type scenarios were ever exposed. (The shipped example is not one of the immune
# ones: it declares `hp` and `heat-storage`, so it reaches electricity *and* heat. It escaped
# because the golden master pins PYTHONHASHSEED=0, under which that two-element set already
# iterates in sorted order.)
MAPPING = {
    'heat_pump': {'heat': 'generation', 'electricity': 'consumption'},
    'pv': {'electricity': 'generation'},
    'heat_storage': {'heat': 'storage'},
    'electrolyser': {'hydrogen': 'generation', 'electricity': 'consumption'},
    'chiller': {'cold': 'generation', 'electricity': 'consumption'},
}
EXPECTED = ['cold', 'electricity', 'heat', 'hydrogen']

# Chosen because they make the raw set iterate in more than one order; the test asserts that
# rather than trusting it, so a Python whose hashing changed cannot quietly make this vacuous.
HASH_SEEDS = ['1', '2', '3', '4', '5', '6', '7', '8']

# Runs in the child. Reports both the function's answer and the raw set's iteration order, so the
# test can tell "the ordering is stable" apart from "nothing varied and I proved nothing".
#
# `controller_base.py` is loaded straight from its path rather than imported as
# `hamlet.executor...`, because `import hamlet` pulls in matplotlib, pandapower and the rest of the
# Creator -- ~18 s per child, and there is one child per hash seed. It is the same file the runtime
# compiles: the parent takes the path from the real imported module, and the venv is an editable
# install. If this module ever grows a *relative* import the children fail loudly; an *absolute*
# one would still pass, but ~18x slower (measured: 12 s -> 213 s), so a sudden slowdown here is
# the symptom to look for rather than a failure.
#
# KNOWN GAP, so nobody reads more into this file than it proves: it pins the ordering that
# `derive_energy_types` returns and that both bases store, but nothing here asserts the order of
# the *model* the backends build from it. Re-wrapping the sequence in the backends -- e.g.
# `for energy_type in set(self.energy_types):` in any of the four `add_balance_constraints` --
# reintroduces the defect one level down with this file still green.
CHILD = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('controller_base', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mapping = json.loads(sys.argv[2])
raw = {energy_type for component in mapping.values() for energy_type in component}
print(json.dumps({'derived': list(module.derive_energy_types(mapping)),
                  'raw_set_order': list(raw)}))
"""


def _module_path():
    from hamlet.executor.utilities.controller import controller_base
    return controller_base.__file__


def _run_under_seed(seed, mapping=None):
    env = dict(os.environ, PYTHONHASHSEED=seed)
    completed = subprocess.run(
        [sys.executable, '-c', CHILD, _module_path(), json.dumps(MAPPING if mapping is None else mapping)],
        capture_output=True, text=True, env=env, check=True)
    return json.loads(completed.stdout.splitlines()[-1])


@pytest.fixture(scope='module')
def real_results():
    """The same sweep over the mapping production actually uses.

    `MAPPING` above is synthetic: it invents `hydrogen` and `cold` through an `electrolyser` and a
    `chiller`, components `c.COMP_MAP` does not contain. The only call that ever happens at
    runtime is `derive_energy_types(c.COMP_MAP)` — both call sites pass that constant itself
    (`rtc.py:51`, `fbc.py:49`) — and it yields exactly **two** energy types.

    That difference is load-bearing: an implementation that sorted only when it had more than two
    types would be a complete no-op in production while every synthetic-mapping check above stayed
    green. The mapping is passed to the children as data, so this stays as cheap as the rest.
    """
    import hamlet.constants as c

    return [_run_under_seed(seed, c.COMP_MAP) for seed in HASH_SEEDS]


def test_the_real_mapping_yields_a_set_that_also_permutes(real_results):
    """Instrument check for the production input, which the synthetic one does not cover.

    The 4-element set permuting says nothing about the 2-element set production uses.
    """
    raw_orders = {tuple(result['raw_set_order']) for result in real_results}

    assert len(raw_orders) > 1, (
        f'the real mapping\'s energy types iterated identically under every seed ({raw_orders}), '
        f'so the check below cannot tell a sorted list from a set for the input that matters')


def test_the_real_mapping_derives_identically_in_every_process(real_results):
    """The property that actually ships: two energy types, same order, whatever the process."""
    derived_orders = {tuple(result['derived']) for result in real_results}

    assert derived_orders == {('electricity', 'heat')}, (
        f'the production mapping does not derive a stable sorted order across processes: '
        f'{derived_orders}')


@pytest.fixture(scope='module')
def results():
    """One subprocess per hash seed, reused by every check below."""
    return [_run_under_seed(seed) for seed in HASH_SEEDS]


def test_the_hash_seed_really_does_permute_the_raw_set(results):
    """The instrument check: without this the rest of the file could pass while testing nothing.

    If the chosen seeds all happened to iterate the raw set identically, every assertion below
    would hold just as well on the defective `set` implementation. This asserts the defect would
    have been visible to this test, before asserting that it is absent.
    """
    raw_orders = {tuple(result['raw_set_order']) for result in results}

    assert len(raw_orders) > 1, (
        f'the {len(HASH_SEEDS)} hash seeds all iterated the raw set in the same order '
        f'({raw_orders}), so this file cannot distinguish a sorted list from a set -- pick '
        f'different seeds or a larger mapping')


def test_the_derived_order_is_identical_in_every_process(results):
    """The property #216 is about: same input, same order, whatever the process."""
    derived_orders = {tuple(result['derived']) for result in results}

    assert len(derived_orders) == 1, (
        f'energy type order varies across processes: {derived_orders}. Every agent model built '
        f'in one process then has different rows and columns from the same model built in '
        f'another, and the solver may return a different equally-optimal vertex (#216)')


def test_the_derived_order_is_sorted(results):
    """Pins *which* stable order, so the ordering cannot drift silently between versions.

    Stability across processes is what #216 needs, but an insertion-ordered answer would also be
    stable *and* would depend on the mapping's construction order, which is not something callers
    should have to reason about.
    """
    for result in results:
        assert result['derived'] == EXPECTED, (
            f"expected {EXPECTED}, got {result['derived']}")


def test_it_returns_a_sequence_rather_than_a_set(results):
    """A set is unordered by definition, so `list(...)` of one is not a fixed order.

    Guards the specific regression: swapping the sorted list back for a set would still satisfy
    a naive "same elements" check.
    """
    from hamlet.executor.utilities.controller.controller_base import derive_energy_types

    derived = derive_energy_types({'pv': {'electricity': 'generation'}})

    assert isinstance(derived, list), (
        f'derive_energy_types returns {type(derived).__name__}, which carries no order')


def test_an_empty_mapping_is_not_an_error():
    """An agent with no plants has no energy types; that is a valid, if idle, agent."""
    from hamlet.executor.utilities.controller.controller_base import derive_energy_types

    assert derive_energy_types({}) == []


@pytest.mark.parametrize('base_import', [
    'hamlet.executor.utilities.controller.rtc.rtc_base:RtcBase',
    'hamlet.executor.utilities.controller.fbc.fbc_base:FbcBase',
], ids=['rtc', 'fbc'])
def test_the_bases_actually_derive_their_energy_types_this_way(base_import):
    """Without this, re-inlining a raw `set()` in either base passes every check above.

    The checks above all exercise `derive_energy_types` directly, so they guard the helper and not
    its use. This one runs the real `__init__` of each base: the constructor is driven on an
    uninitialised instance with only `mapping` supplied, so it assigns `mapping` and
    `energy_types` and then raises on the next line, which needs a timetable this test has no
    reason to build. What it asserts is the attribute the constructor actually left behind.
    """
    import importlib

    import hamlet.constants as c
    from hamlet.executor.utilities.controller.controller_base import derive_energy_types

    module_name, class_name = base_import.split(':')
    base = getattr(importlib.import_module(module_name), class_name)

    instance = object.__new__(base)          # no __init__ -- we are about to drive it ourselves
    with pytest.raises(Exception):           # the timetable lookup, several lines later
        base.__init__(instance, mapping=c.COMP_MAP)

    assert hasattr(instance, 'energy_types'), (
        f'{class_name}.__init__ did not set energy_types before it failed -- this test drives the '
        f'constructor further than it used to reach, so its stub needs updating')
    assert instance.energy_types == derive_energy_types(c.COMP_MAP), (
        f'{class_name} does not derive its energy types through derive_energy_types; it got '
        f'{instance.energy_types!r}')
    assert isinstance(instance.energy_types, list), (
        f'{class_name}.energy_types is {type(instance.energy_types).__name__}, which carries no '
        f'order')
