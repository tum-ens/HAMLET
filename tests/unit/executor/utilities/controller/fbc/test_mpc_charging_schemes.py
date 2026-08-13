"""Unit — every `charging_scheme` method must build its constraints, on both backends.

`charging_scheme.method: full` could not work for **any** agent type on the default backend (#220).
PyOptInterface's `__constraint_cs_full` handed `add_linear_constraint` the whole variable array and
a whole `Series`, where it takes one expression and a scalar rhs:

    TypeError: add_linear_constraint(): incompatible function arguments.
    Invoked with types: Model, ndarray, ConstraintSense, pandas.core.series.Series

The sibling `__constraint_cs_min_soc` already looped `for timestep in self.timesteps:`, and the
linopy implementation takes arrays natively, so this was POI-port fallout rather than a modelling
disagreement.

**It was latent, not live** — no shipped scenario asked for `full` until `ctsp_industry` turned its
EV share on, which is the same change that needed #219. Worth saying plainly, because `min_soc` is
the only method any test or scenario had ever exercised, on either backend.

`tests/unit/.../test_mpc_poi_components.py` deliberately needs no solver and so only *constructs*
the components. That is exactly why this defect survived: it lives in `define_constraints`. These
tests build a real model, so they skip where HiGHS cannot be loaded.
"""
import datetime

import pytest

import hamlet.constants as c
from hamlet.executor.utilities.controller.fbc.mpc.linopy import components as linopy_components
from hamlet.executor.utilities.controller.fbc.mpc.poi import components as poi_components
from tests.poi_support import available_backend

EV = 'ev1'
CAPACITY = 50_000  # Wh
CHARGING_POWER = 11_000  # W

#: The four methods `__constraint_charging_scheme` dispatches on, and what each needs in the block.
#: Derived from the `match` in both backends rather than listed by hand -- see
#: `test_the_matrix_covers_every_method_the_backends_dispatch_on`.
SCHEMES = {
    'full': {'method': 'full'},
    'price_sensitive': {'method': 'price_sensitive', 'price_sensitive': {'threshold': 0.05}},
    'min_soc': {'method': 'min_soc', 'min_soc': {'val': 0.8}},
    'min_soc_time': {'method': 'min_soc_time', 'min_soc_time': {'val': 0.8, 'time': 8.5}},
}

#: Methods neither backend implements yet. They must raise `NotImplementedError` -- a *named*
#: refusal -- rather than the `TypeError` that #220 was.
NOT_IMPLEMENTED = ('price_sensitive', 'min_soc_time')

SIZING = {'capacity': CAPACITY, 'charging_home': CHARGING_POWER, 'charging_AC': CHARGING_POWER,
          'charging_DC': CHARGING_POWER, 'charging_efficiency': 1.0, 'v2g': 0}


def make_ev(module, scheme, timesteps):
    """An EV that drives in one timestep, so the SoC trajectory is not flat."""
    return module.Ev(
        EV,
        forecasts={f'{EV}_availability': [1, 1, 1, 1],
                   f'{EV}_energy_consumed': [0, 10_000, 0, 0]},
        timesteps=timesteps, delta=datetime.timedelta(hours=1), socs={EV: [30_000]},
        charging_scheme=scheme, sizing=SIZING)


def build_poi(scheme):
    """Define the EV's variables and constraints on a real PyOptInterface model."""
    from hamlet.executor.utilities.controller.poi_solver import create_model

    ev = make_ev(poi_components, scheme, timesteps=[0, 1, 2, 3])
    model, variables = create_model('highs'), {}
    ev.define_variables(model, variables, comp_type=c.P_EV)
    ev.define_constraints(model, variables)
    return model


def build_linopy(scheme, timesteps):
    from linopy import Model

    ev = make_ev(linopy_components, scheme, timesteps=timesteps)
    model = ev.define_variables(Model(force_dim_names=True), comp_type=c.P_EV)
    return ev.define_constraints(model)


@pytest.fixture(scope='module')
def poi_available():
    if available_backend() is None:
        pytest.skip('no PyOptInterface solver can be loaded here, so no model can be built')


IMPLEMENTED = [name for name in SCHEMES if name not in NOT_IMPLEMENTED]


@pytest.mark.parametrize('method', IMPLEMENTED)
def test_poi_builds_the_constraints_for_every_implemented_scheme(method, poi_available):
    """Regression for #220. `full` raised `TypeError`; `min_soc` was the only method ever built."""
    build_poi(SCHEMES[method])


@pytest.mark.parametrize('method', IMPLEMENTED)
def test_linopy_builds_the_constraints_for_every_implemented_scheme(method, timesteps):
    """The reference implementation, so a POI failure cannot be blamed on the scheme itself."""
    build_linopy(SCHEMES[method], timesteps)


@pytest.mark.parametrize('method', NOT_IMPLEMENTED)
def test_an_unimplemented_scheme_refuses_by_name_on_poi(method, poi_available):
    """A scheme HAMLET does not model must say so, not fail on an argument type.

    Pinned because it is what tells a config error apart from #220: before the fix, `full` and
    these two were indistinguishable from the outside -- every one of them was simply "it crashed".
    """
    with pytest.raises(NotImplementedError):
        build_poi(SCHEMES[method])


@pytest.mark.parametrize('method', NOT_IMPLEMENTED)
def test_an_unimplemented_scheme_refuses_by_name_on_linopy(method, timesteps):
    with pytest.raises(NotImplementedError):
        build_linopy(SCHEMES[method], timesteps)


def test_an_unknown_scheme_is_rejected(poi_available):
    """The `case _` arm. Without it a typo in `charging_scheme.method` charges however it likes."""
    with pytest.raises(ValueError):
        build_poi({'method': 'charge_a_lot'})


def test_the_full_scheme_constrains_every_timestep(poi_available):
    """The fix is a per-timestep loop, so the horizon must be constrained across its whole length.

    Counting constraints rather than reading the source: a loop written `for timestep in [0]` would
    pass `test_poi_builds_the_constraints_for_every_implemented_scheme` and leave three quarters of
    the horizon free. Compared against `min_soc`, the sibling that was already correct, so the
    number comes from the code that works rather than from a constant typed in here.
    """
    import pyoptinterface as poi

    full = build_poi(SCHEMES['full']).number_of_constraints(poi.ConstraintType.Linear)
    min_soc = build_poi(SCHEMES['min_soc']).number_of_constraints(poi.ConstraintType.Linear)

    assert full == min_soc, (
        f'the `full` scheme added {full} linear constraints where `min_soc` -- the sibling that '
        f'was already correct -- adds {min_soc} over the same four-timestep horizon')


@pytest.mark.parametrize('module, name',
                         [(linopy_components, 'linopy'), (poi_components, 'poi')],
                         ids=['linopy', 'poi'])
def test_the_matrix_covers_every_method_the_backends_dispatch_on(module, name):
    """`SCHEMES` is derived from the dispatch, not maintained beside it.

    A hand-kept list of methods is a test that passes by omission: adding a fifth scheme to the
    `match` would leave it uncovered and everything here green. The `case` labels are read out of
    the dispatch so a new one fails this instead.
    """
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(module.Ev))
    match = next(node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Match) and any(
                     isinstance(case.pattern, ast.MatchValue) for case in node.cases))
    dispatched = {case.pattern.value.value for case in match.cases
                  if isinstance(case.pattern, ast.MatchValue)}

    assert dispatched == set(SCHEMES), (
        f'{name} dispatches on {sorted(dispatched)} but this file covers {sorted(SCHEMES)}; '
        f'add the missing scheme here rather than leaving it untested (#220)')
