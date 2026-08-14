"""Unit -- §14a's shortest paths are computed once per run, and the cache is not a lie.

`EnWG14a.__calculate_shortest_path_to_trafo` ran on every timestep. Its answer depends only on the
topology, and a timestep writes `p_mw` and `q_mvar` onto loads and sgens -- never a bus, a line or
a transformer. So it was repeated work: measured on design 6 (104 agents, live grid), **0.609 s of
a 7.8 s timestep**.

The cache lives on the `GridDB` rather than on `EnWG14a`, because `Electricity.execute` builds a
new `EnWG14a` every timestep. Caching on the instance would have cached nothing while looking
correct, which is the failure this file exists to catch.

The invariant it rests on -- topology is static within a run -- is not enforced anywhere, so the
last test states it as a test rather than as a comment.
"""
import pandapower as pp
import pytest

from hamlet.executor.utilities.grid_restrictions.enwg_14a import EnWG14a


def network():
    """Two feeders under one transformer, so paths of different lengths exist."""
    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=20.0, zone='region')
    lv = pp.create_bus(net, vn_kv=0.4, zone='region')
    pp.create_ext_grid(net, bus=hv)
    pp.create_transformer(net, hv_bus=hv, lv_bus=lv, std_type='0.25 MVA 20/0.4 kV')

    previous = lv
    for _ in range(3):
        bus = pp.create_bus(net, vn_kv=0.4, zone='region')
        pp.create_line(net, from_bus=previous, to_bus=bus, length_km=0.05,
                       std_type='NAYY 4x50 SE')
        previous = bus

    branch = pp.create_bus(net, vn_kv=0.4, zone='region')
    pp.create_line(net, from_bus=lv, to_bus=branch, length_km=0.05, std_type='NAYY 4x50 SE')
    pp.create_load(net, bus=previous, p_mw=0.001)
    return net, lv


class Db:
    """Stands in for the GridDB: the only thing the cache needs is somewhere to live."""


@pytest.fixture
def restriction():
    instance = EnWG14a.__new__(EnWG14a)
    instance.grid_db = Db()
    return instance


def paths(restriction, net, lv):
    return restriction._EnWG14a__calculate_shortest_path_to_trafo(
        grid=net, trafo_lv_bus_index=lv)


def computed(net, lv):
    return EnWG14a._EnWG14a__compute_shortest_path_to_trafo(net, lv)


def test_the_first_call_matches_a_direct_computation(restriction):
    net, lv = network()

    assert paths(restriction, net, lv) == computed(net, lv)


def test_paths_are_actually_found(restriction):
    """Guard against the whole thing being vacuously empty."""
    net, lv = network()

    result = paths(restriction, net, lv)

    assert result, 'no paths at all, so equality with the oracle would prove nothing'
    assert max(len(path) for path in result.values()) >= 2, 'expected a multi-line path'


def test_the_second_call_is_served_from_the_cache(restriction):
    net, lv = network()
    first = paths(restriction, net, lv)

    assert paths(restriction, net, lv) is first, 'recomputed, so the cache is doing nothing'


def test_the_cache_lives_on_the_grid_db_not_the_restriction(restriction):
    """`EnWG14a` is rebuilt every timestep, so a cache on it would never be hit twice."""
    net, lv = network()
    paths(restriction, net, lv)

    assert getattr(restriction.grid_db, '_shortest_path_cache', None), 'cache is not on the GridDB'

    later = EnWG14a.__new__(EnWG14a)          # what the next timestep constructs
    later.grid_db = restriction.grid_db       # the GridDB is what survives

    assert paths(later, net, lv) is paths(restriction, net, lv)


def test_writing_timestep_parameters_does_not_change_the_answer(restriction):
    """The invariant the cache rests on, stated as a test.

    A timestep writes power onto loads and sgens. If that could change the paths, caching across
    timesteps would be wrong -- so this asserts it cannot.
    """
    net, lv = network()
    before = computed(net, lv)

    net.load['p_mw'] = 0.42
    net.load['q_mvar'] = 0.1
    if len(net.sgen):
        net.sgen['p_mw'] = 0.3

    assert computed(net, lv) == before
