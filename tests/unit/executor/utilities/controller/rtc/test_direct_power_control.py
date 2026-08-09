"""L2 — direct power control (§14a EnWG) reaches the model, in both RTC backends.

When the grid stage finds an overload it issues a per-plant power cap, which the real-time
controller must impose on the optimisation. `framework` is a per-agent option, so an agent on
`poi` must honour a cap exactly as one on `linopy` does -- a backend that ignores it does not
fail, it silently runs an uncapped grid.

This is tested at the controller method rather than end to end because no shipped example
exercises §14a at all: the two scenarios that set `restrictions.apply: ['enwg_14a']` have
`electricity.active: False`, and the two with a live grid have an empty restriction list. See
#198.

`apply_grid_commands` only needs a handful of attributes off the controller, so it is called
against a bare instance carrying exactly those -- `__new__` without `__init__`. That keeps the
test on the behaviour under test rather than on the scaffolding a real controller construction
needs (an agent, a timetable, markets, a forecaster). A real instance rather than a stand-in
object, because the method dispatches to name-mangled private helpers that only resolve on the
class itself.
"""
import types

import pytest

import hamlet.constants as c
from hamlet.executor.utilities.controller.rtc.optim.linopy.optim_linopy import Linopy
from hamlet.executor.utilities.controller.rtc.optim.optim_base import OptimBase
from hamlet.executor.utilities.controller.rtc.optim.poi.optim_poi import POI
from tests.poi_support import available_backend, skip_on_windows

AGENT = 'agent1'
PLANT = 'hp1'
POWER = f'{PLANT}_{c.P_HP}_{c.ET_ELECTRICITY}'
TARGET = f'{PLANT}_{c.P_HP}_target'

# A heat pump able to draw up to 5 kW, currently targeted at 3 kW, capped by the grid at 2 kW.
# Negative because a load is modelled as negative in HAMLET's sign convention.
FULL_POWER, TARGET_POWER, CAP = -5000, -3000, -2000


def commands(cap=CAP, plant=PLANT):
    return {c.G_ELECTRICITY: {'current_direct_power_control': {AGENT: {plant: cap}}}}


def test_the_base_controller_ignores_commands_by_default():
    """Pins why a missing override is dangerous: the base is a silent no-op, not an error.

    `OptimBase.apply_grid_commands` is called unconditionally during model construction, so a
    backend that does not override it discards every cap without raising.
    """
    assert OptimBase.apply_grid_commands.__doc__
    stand_in = types.SimpleNamespace()

    assert OptimBase.apply_grid_commands(stand_in) is None


@pytest.mark.parametrize('backend', [Linopy, POI], ids=['linopy', 'poi'])
def test_both_backends_implement_direct_power_control(backend):
    """Neither backend may inherit the base no-op.

    A cheap structural guard that holds even where no solver library is available, so the parity
    itself is pinned on every platform rather than only where the model can be built.
    """
    assert backend.apply_grid_commands is not OptimBase.apply_grid_commands, (
        f'{backend.__module__} inherits the base no-op, so it silently discards grid commands')


class TestLinopy:
    """The reference implementation, and the behaviour the POI backend has to match."""

    @staticmethod
    def build():
        from linopy import Model

        model = Model(force_dim_names=True)
        model.add_variables(lower=FULL_POWER, upper=0, name=POWER)
        model.add_variables(lower=TARGET_POWER, upper=TARGET_POWER, name=TARGET)
        return model

    @staticmethod
    def controller(model, grid_commands):
        """A Linopy controller carrying only what `apply_grid_commands` reads."""
        instance = Linopy.__new__(Linopy)
        instance.model = model
        instance.grid_commands = grid_commands
        instance.agent = types.SimpleNamespace(agent_id=AGENT)
        instance.plants = {PLANT: {'type': c.P_HP}}
        return instance

    def test_a_cap_tightens_the_plant_power_bound(self):
        model = self.build()

        self.controller(model, commands()).apply_grid_commands()

        assert float(model.variables[POWER].lower) == CAP

    def test_a_cap_also_moves_the_target(self):
        """The target must follow the cap, or the deviation term prices a now-illegal setpoint."""
        model = self.build()

        self.controller(model, commands()).apply_grid_commands()

        assert float(model.variables[TARGET].lower) == CAP
        assert float(model.variables[TARGET].upper) == CAP

    def test_no_command_for_this_agent_leaves_the_model_alone(self):
        model = self.build()
        other = {c.G_ELECTRICITY: {'current_direct_power_control': {'someone_else': {PLANT: CAP}}}}

        self.controller(model, other).apply_grid_commands()

        assert float(model.variables[POWER].lower) == FULL_POWER

    def test_an_empty_command_set_leaves_the_model_alone(self):
        model = self.build()

        self.controller(model, {}).apply_grid_commands()

        assert float(model.variables[POWER].lower) == FULL_POWER


@skip_on_windows
class TestPoi:
    """The same behaviour, expressed against PyOptInterface's model API."""

    @staticmethod
    def build():
        import pyoptinterface as poi

        from hamlet.executor.utilities.controller.poi_solver import create_model

        model = create_model('highs')
        variables = {
            POWER: model.add_variable(lb=FULL_POWER, ub=0, name=POWER),
            TARGET: model.add_variable(lb=TARGET_POWER, ub=TARGET_POWER, name=TARGET),
        }
        return model, variables, poi

    @staticmethod
    def controller(model, variables, grid_commands):
        """A POI controller carrying only what `apply_grid_commands` reads."""
        instance = POI.__new__(POI)
        instance.model = model
        instance.variables = variables
        instance.grid_commands = grid_commands
        instance.agent = types.SimpleNamespace(agent_id=AGENT)
        instance.plants = {PLANT: {'type': c.P_HP}}
        return instance

    @staticmethod
    def bounds(model, variables, poi, name):
        return (model.get_variable_attribute(variables[name], poi.VariableAttribute.LowerBound),
                model.get_variable_attribute(variables[name], poi.VariableAttribute.UpperBound))

    def setup_method(self):
        if available_backend() is None:
            pytest.skip('no PyOptInterface solver library is loadable')

    @pytest.mark.solver
    def test_a_cap_tightens_the_plant_power_bound(self):
        model, variables, poi = self.build()

        self.controller(model, variables, commands()).apply_grid_commands()

        assert self.bounds(model, variables, poi, POWER)[0] == CAP

    @pytest.mark.solver
    def test_a_cap_also_moves_the_target(self):
        model, variables, poi = self.build()

        self.controller(model, variables, commands()).apply_grid_commands()

        assert self.bounds(model, variables, poi, TARGET) == (CAP, CAP)

    @pytest.mark.solver
    def test_no_command_for_this_agent_leaves_the_model_alone(self):
        model, variables, poi = self.build()
        other = {c.G_ELECTRICITY: {'current_direct_power_control': {'someone_else': {PLANT: CAP}}}}

        self.controller(model, variables, other).apply_grid_commands()

        assert self.bounds(model, variables, poi, POWER)[0] == FULL_POWER

    @pytest.mark.solver
    def test_an_empty_command_set_leaves_the_model_alone(self):
        model, variables, poi = self.build()

        self.controller(model, variables, {}).apply_grid_commands()

        assert self.bounds(model, variables, poi, POWER)[0] == FULL_POWER
