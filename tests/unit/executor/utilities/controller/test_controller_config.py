"""Unit — controller configuration becoming model behaviour.

This is the seam where a user's `agents.xlsx` / `agents.yaml` turns into optimisation bounds and
slack settings. Both controller bases resolve it through these two functions, so a change that
stopped honouring configuration entirely would fail here rather than pass everything.
"""
import pytest

import hamlet.constants as c


class TestResolveLimits:
    """`limits:` under a controller, merged onto the defaults."""

    def test_no_configuration_gives_the_defaults(self):
        assert c.resolve_limits({}) == c.RTC_DEFAULT_LIMITS

    @pytest.mark.parametrize('ems', [{}, None, {c.K_LIMITS: None}, {c.K_LIMITS: {}}])
    def test_absent_or_empty_configuration_changes_nothing(self, ems):
        assert c.resolve_limits(ems) == c.RTC_DEFAULT_LIMITS

    def test_a_configured_bound_reaches_the_result(self):
        resolved = c.resolve_limits({c.K_LIMITS: {'market_power': 4_000_000}})

        assert resolved['market_power'] == 4_000_000

    def test_the_other_bounds_keep_their_defaults(self):
        resolved = c.resolve_limits({c.K_LIMITS: {'market_power': 4_000_000}})

        assert resolved['balancing_power'] == c.RTC_DEFAULT_LIMITS['balancing_power']
        assert resolved['hp_power_heat'] == c.RTC_DEFAULT_LIMITS['hp_power_heat']

    def test_the_defaults_are_not_mutated(self):
        """A merge that wrote into the module-level dict would leak between agents."""
        before = dict(c.RTC_DEFAULT_LIMITS)
        c.resolve_limits({c.K_LIMITS: {'market_power': 1}})

        assert c.RTC_DEFAULT_LIMITS == before

    def test_an_unknown_bound_is_carried_through(self):
        """A bound added to a component before the defaults table must still reach it."""
        resolved = c.resolve_limits({c.K_LIMITS: {'some_new_bound': 42}})

        assert resolved['some_new_bound'] == 42


class TestResolveSlack:
    """`slack:` and `penalties:` under a controller."""

    def test_slack_is_on_by_default(self):
        enabled, _ = c.resolve_slack({}, c.FBC_DEFAULT_SLACK_PENALTY)

        assert enabled is True

    def test_slack_can_be_switched_off_per_agent(self):
        """The documented way to reproduce runs made before the slacks existed."""
        enabled, _ = c.resolve_slack({c.K_SLACK: False}, c.FBC_DEFAULT_SLACK_PENALTY)

        assert enabled is False

    def test_the_default_penalty_is_the_one_passed_in(self):
        """The two controllers have different objectives and so different penalty scales."""
        assert c.resolve_slack({}, c.FBC_DEFAULT_SLACK_PENALTY)[1] == c.FBC_DEFAULT_SLACK_PENALTY
        assert c.resolve_slack({}, c.RTC_DEFAULT_SLACK_PENALTY)[1] == c.RTC_DEFAULT_SLACK_PENALTY

    def test_a_configured_penalty_overrides_it(self):
        _, penalty = c.resolve_slack({c.K_PENALTIES: {'slack': 7}}, c.FBC_DEFAULT_SLACK_PENALTY)

        assert penalty == 7

    @pytest.mark.parametrize('ems', [{}, None, {c.K_PENALTIES: None}])
    def test_absent_or_empty_configuration_is_tolerated(self, ems):
        enabled, penalty = c.resolve_slack(ems, c.FBC_DEFAULT_SLACK_PENALTY)

        assert enabled is c.DEFAULT_SLACK_ENABLED
        assert penalty == c.FBC_DEFAULT_SLACK_PENALTY


class TestSlackPenaltyMagnitude:
    """The penalty has to dominate the price of serving the load, or shedding becomes a trade."""

    def test_the_feedback_penalty_outprices_any_realistic_tariff(self):
        """Prices are integers in 0.01 ct/kWh; the shipped data's most expensive stack is 3700.

        A penalty below that would let the optimiser shed load to save money.
        """
        most_expensive_shipped_stack = 1500 + 400 + 1800  # energy + grid fee + levy

        assert c.FBC_DEFAULT_SLACK_PENALTY > 10 * most_expensive_shipped_stack

    def test_the_realtime_penalty_outranks_every_deviation_weight(self):
        """The real-time objective is a weighted sum of deviations, not a cost.

        The weights are read from the controller itself rather than restated here, so raising
        one above the slack penalty fails this test instead of silently inverting the priority.
        """
        from hamlet.executor.utilities.controller.rtc.optim.linopy import optim_linopy
        import inspect
        import re

        source = inspect.getsource(optim_linopy.Linopy.define_objective)
        weights = [int(value) for value in re.findall(r':\s*(\d+),?\s*#\s*weight', source)]

        assert weights, 'could not read the deviation weights from the controller'
        assert c.RTC_DEFAULT_SLACK_PENALTY > max(weights)
