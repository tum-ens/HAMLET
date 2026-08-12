"""Unit — `import hamlet` must not touch the process's warning filters, and what it does
suppress during a run must be exactly the enumerated list.

Issue #199: `hamlet/executor/setup.py` called `warnings.filterwarnings("ignore")` at module
scope and `hamlet/__init__.py` imports it, so importing the package silenced every warning in the
process -- HAMLET's own, and every dependency's. A second blanket filter lived at the top of
`hamlet/creator/agents/agents.py`.

Three separate claims are tested here, because they fail for different reasons:

1. **Nothing at import.** A subprocess check, since the filter list is process-global and pytest
   installs its own filters -- comparing inside this process would compare against pytest's.
2. **The policy is narrow.** A warning that is not on the list survives `quiet_known_noise`,
   including a `DeprecationWarning` with an unlisted message. This is what tells a blanket filter
   apart from an enumerated one.
3. **The list is not stale.** Every pattern is matched against the message it was added for.
   If polars rewords one, the pattern silently stops matching and the entry becomes decoration;
   that is the safe direction to fail in only if somebody is told.
"""
import re
import subprocess
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest
from polars.exceptions import MapWithoutReturnDtypeWarning

from hamlet.warning_policy import SUPPRESSED, quiet, quiet_known_noise

#: One real (category, message) pair per `SUPPRESSED` entry, recorded from runs of
#: `examples/create_simple_scenario` and `tests/e2e/scenarios/grid_golden` with the filters
#: lifted after import.
#:
#: The **category is recorded here too, not read from `SUPPRESSED`**. An earlier version of this
#: table held messages only and took the category from the entry under test, which made the
#: checks below self-consistent: give an entry the wrong category and it would warn with that
#: same wrong category, match its own filter, and pass. Mutation testing found it -- reading
#: could not have. The observed data and the thing being tested have to be independent.
RECORDED = (
    (DeprecationWarning, '`cumsum` is deprecated. It has been renamed to `cum_sum`.'),
    (DeprecationWarning, '`groupby` is deprecated. It has been renamed to `group_by`.'),
    (DeprecationWarning, '`apply` is deprecated. It has been renamed to `map_elements`.'),
    (DeprecationWarning,
     'The `axis` parameter for `DataFrame.sum` is deprecated. Use `DataFrame.sum_horizontal()` '
     'to perform horizontal aggregation.'),
    (DeprecationWarning, "Use of `how='outer'` should be replaced with `how='full'`."),
    (DeprecationWarning,
     'The default coalesce behavior of left join will change to `False` in the next breaking '
     'release. Pass `coalesce=True` to keep the current behavior and silence this warning.'),
    (MapWithoutReturnDtypeWarning,
     'Calling `map_elements` without specifying `return_dtype` can lead to unpredictable '
     'results. Specify `return_dtype` to silence this warning.'),
)

#: Run in a subprocess, because the question is what `import hamlet` does to a *pristine*
#: interpreter and pytest has already installed filters of its own in this one.
#:
#: Stated behaviourally rather than as "the filter list is unchanged". Importing HAMLET pulls in
#: numpy, scipy, urllib3, ruamel, deepdiff and requests, and between them those add a dozen or so
#: filters at import -- every one naming a specific category or message. There is nothing in a
#: filter tuple that says who installed it, so "nothing was added" is not a property HAMLET can
#: have. "Nothing HAMLET added swallows an ordinary warning" is, and it is the property #199 is
#: actually about.
IMPORT_PROBE = """
import json, sys, warnings


# The categories a blanket filter in this package would plausibly name. Restricted to the
# built-ins on purpose: a dependency that unconditionally ignores a category it defines itself
# (urllib3 does exactly that with `DependencyWarning`) is silencing only its own noise and is
# none of HAMLET's business.
BROAD = (Warning, DeprecationWarning, PendingDeprecationWarning, FutureWarning, UserWarning,
         RuntimeWarning)


def unconditional_ignores():
    # Python's own defaults already include unconditional ignores for DeprecationWarning,
    # PendingDeprecationWarning, ImportWarning and ResourceWarning, so this is only meaningful
    # as a before/after difference.
    return [str(entry) for entry in warnings.filters
            if entry[0] == 'ignore' and entry[1] is None and entry[3] is None
            and entry[2] in BROAD]


before = unconditional_ignores()

import hamlet

with warnings.catch_warnings(record=True) as caught:
    warnings.warn('a plain user warning', UserWarning)
    warnings.warn('a future warning', FutureWarning)
    warnings.warn('a runtime warning', RuntimeWarning)
    warnings.warn('a deprecation warning', DeprecationWarning)

json.dump({'escaped': sorted(record.category.__name__ for record in caught),
           'blanket': [entry for entry in unconditional_ignores() if entry not in before]},
          sys.stdout)
"""


def test_importing_hamlet_does_not_silence_ordinary_warnings():
    """The headline of #199.

    Both original filters fail this: the blanket `ignore` swallows all four categories, and
    `agents.py`'s `simplefilter(ignore, FutureWarning)` swallows one. Checked by raising
    warnings rather than by grepping the source, so it also catches the filter reappearing
    somewhere new -- which matters, because there turned out to be two of them in different
    files and the issue only named one.

    `DeprecationWarning` counts here because the probe runs as `__main__`, where Python's own
    default is to show it; a blanket filter is inserted at position 0 and would win.
    """
    import json

    completed = subprocess.run([sys.executable, '-c', IMPORT_PROBE], capture_output=True,
                               text=True, encoding='utf-8', timeout=300)
    assert completed.returncode == 0, completed.stderr[-3000:]
    result = json.loads(completed.stdout)

    assert result['escaped'] == ['DeprecationWarning', 'FutureWarning', 'RuntimeWarning',
                                 'UserWarning'], (
        f"after `import hamlet`, only {result['escaped']} of the four categories raised reached "
        f'the caller. A library must not edit the warning configuration of whatever imports it; '
        f'if a run needs warnings quietened, that belongs in `quiet_known_noise`, which is '
        f'entered around the run and left again afterwards.')
    assert not result['blanket'], (
        f"`import hamlet` leaves an unconditional ignore filter in place: {result['blanket']}")


class TestThePolicyIsNarrow:
    """A blanket filter and an enumerated one both silence the noise. Only one of them lets the
    next real warning through, and that is the whole point of #199."""

    def test_a_listed_warning_is_suppressed(self):
        """Otherwise the policy is decoration and the console floods."""
        category, message = RECORDED[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with quiet_known_noise():
                warnings.warn(message, category)

        assert not caught, f'expected the listed message to be hidden, got {caught}'

    @pytest.mark.parametrize('category', [UserWarning, RuntimeWarning, FutureWarning])
    def test_an_unlisted_category_still_reaches_the_user(self, category):
        """The `UserWarning` case is live, not hypothetical: `enwg_14a` raises 64 per run of
        pandas' 'Boolean Series key will be reindexed' (#210), and the blanket filter is why
        nobody had seen them."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with quiet_known_noise():
                warnings.warn('something the policy has never heard of', category)

        assert [record.category for record in caught] == [category]

    def test_an_unlisted_deprecation_still_reaches_the_user(self):
        """The sharp case. Suppressing `DeprecationWarning` wholesale would pass every other test
        in this class while re-creating the defect for the one category the noise came from."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with quiet_known_noise():
                warnings.warn('`some_future_polars_api` is deprecated', DeprecationWarning)

        assert len(caught) == 1, (
            'a DeprecationWarning whose message is not in SUPPRESSED was hidden, so the policy '
            'is filtering by category rather than by the enumerated messages')

    def test_the_filters_are_gone_again_afterwards(self):
        """A context manager that leaks is a blanket filter with extra steps.

        `resetwarnings()` first, and this is load-bearing rather than tidiness. `conftest.py`
        registers `SUPPRESSED` with pytest, which re-applies it before every test, so the
        policy's filters are *already installed* when the test starts -- and
        `warnings.filterwarnings` removes an identical entry before re-inserting it, so a
        leaking context manager would leave the list byte-identical and this test would pass
        against it. It did: mutation testing caught the vacuity, reading it did not.

        The inner assertion is the other half. Without it the test would also pass against a
        `quiet_known_noise` that installs nothing at all.
        """
        with warnings.catch_warnings():
            warnings.resetwarnings()
            before = list(warnings.filters)

            with quiet_known_noise():
                assert list(warnings.filters) != before, 'no filters were installed at all'

            assert list(warnings.filters) == before

    def test_the_filters_are_restored_when_the_block_raises(self):
        """The failure mode of the `sys.stdout = open(os.devnull)` hack this replaces.

        Same `resetwarnings()` reasoning as above.
        """
        with warnings.catch_warnings():
            warnings.resetwarnings()
            before = list(warnings.filters)

            with pytest.raises(ValueError):
                with quiet_known_noise():
                    assert list(warnings.filters) != before, 'no filters were installed at all'
                    raise ValueError('boom')

            assert list(warnings.filters) == before


class TestTheListIsNotStale:
    """Patterns are matched against messages produced by somebody else's library."""

    def test_the_recorded_messages_line_up_with_the_entries(self):
        """Guards the parametrisation below against silently covering fewer entries."""
        assert len(RECORDED) == len(SUPPRESSED)

    @pytest.mark.parametrize('index', range(len(SUPPRESSED)))
    def test_every_entry_names_the_category_the_warning_is_actually_raised_with(self, index):
        """A filter with the right message and the wrong category hides nothing.

        Checked against the recorded category rather than by re-raising with the entry's own,
        which is the trap the module docstring for `RECORDED` describes.
        """
        listed, _, reason = SUPPRESSED[index]
        observed, _ = RECORDED[index]

        assert listed is observed, (
            f'SUPPRESSED[{index}] ({reason}) filters {listed.__name__}, but the warning is '
            f'raised as {observed.__name__}, so the entry suppresses nothing')

    @pytest.mark.parametrize('index', range(len(SUPPRESSED)))
    def test_every_pattern_still_matches_a_real_message(self, index):
        """A pattern that matches nothing suppresses nothing, and reads as though it does."""
        _, pattern, reason = SUPPRESSED[index]
        _, message = RECORDED[index]

        assert re.match(pattern, message), (
            f'SUPPRESSED[{index}] ({reason}) no longer matches the message it was added for:\n'
            f'  pattern: {pattern}\n  message: {message}\n'
            f'Either the dependency reworded it -- in which case the warning is now visible '
            f'again and the entry needs updating -- or the pattern was edited by mistake.')

    @pytest.mark.parametrize('index', range(len(SUPPRESSED)))
    def test_every_entry_actually_hides_its_message(self, index):
        """The end-to-end version of the above: pattern plus category, through the real filter.

        Matching the regex is necessary but not sufficient -- `filterwarnings` anchors at the
        start of the message and pairs the pattern with a category, so an entry can have a
        correct regex and the wrong category and still hide nothing.
        """
        category, message = RECORDED[index]
        _, _, reason = SUPPRESSED[index]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with quiet_known_noise():
                warnings.warn(message, category)

        assert not caught, f'SUPPRESSED[{index}] ({reason}) did not hide its own message'


class TestTheEntryPointsUseIt:
    """The policy is only worth anything where it is applied.

    Everything here is behavioural. An earlier version asserted `'quiet_known_noise()' in
    inspect.getsource(Executor.run)`, and a review panel defeated it by deleting the `with` block
    and leaving a comment mentioning the name -- the source-text check passed with the policy
    entirely absent. Source-text assertions are satisfiable by comments; this repo has been bitten
    by that before.
    """

    def test_quiet_actually_suppresses(self):
        """What `@quiet` is *for*, as opposed to the fact that it is a decorator.

        A panel gutted `quiet` to a bare `functools.wraps` pass-through and every test in this
        class stayed green, because they all checked decorator metadata. This one goes red.
        """
        category, message = RECORDED[0]

        @quiet
        def noisy():
            warnings.warn(message, category)
            warnings.warn('not on the list', UserWarning)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            noisy()

        assert [record.category for record in caught] == [UserWarning], (
            '`@quiet` did not apply the policy: expected the listed message hidden and the '
            f'unlisted one through, got {[record.category.__name__ for record in caught]}')

    def test_the_creator_entry_points_use_this_decorator(self):
        """All three, not just the sink they share: which one a user calls is their choice.

        Identified by where the wrapper's code was compiled, not by `__wrapped__` being set --
        `functools.wraps` copies `__module__` and `__qualname__` off the wrapped function, so a
        locally defined, unrelated `@wraps` decorator is indistinguishable from this one by every
        attribute except this. A panel substituted exactly that and the old check passed.
        """
        from hamlet.creator.setup import Creator

        for name in ('new_scenario_from_configs', 'new_scenario_from_grids',
                     'new_scenario_from_files'):
            method = getattr(Creator, name)
            assert getattr(method, '__wrapped__', None) is not None, (
                f'Creator.{name} is not wrapped at all, so a scenario created through it prints '
                f'the polars deprecation noise')
            assert Path(method.__code__.co_filename).name == 'warning_policy.py', (
                f'Creator.{name} is wrapped by a decorator defined in '
                f'{method.__code__.co_filename}, not by `warning_policy.quiet`')

    def test_the_executor_enters_the_policy_around_every_stage(self, monkeypatch):
        """`Executor.run` uses the block form, so there is no decorator to inspect.

        Driven rather than read: a recording context manager replaces `quiet_known_noise` and the
        three stages are stubbed, so the assertion is about the order things actually happened in
        -- the policy is entered before `setup` and left after `cleanup`, not wrapped around one
        stage or dropped entirely.
        """
        from hamlet.executor import setup as executor_setup

        events = []

        @contextmanager
        def recording():
            events.append('enter')
            try:
                yield
            finally:
                events.append('exit')

        monkeypatch.setattr(executor_setup, 'quiet_known_noise', recording)

        class Probe(executor_setup.Executor):
            def __init__(self):  # noqa: D107 -- deliberately skips the real constructor
                pass

            def setup(self):
                events.append('setup')

            def execute(self):
                events.append('execute')

            def cleanup(self):
                events.append('cleanup')

        Probe().run()

        assert events == ['enter', 'setup', 'execute', 'cleanup', 'exit'], (
            f'the executor did not run its stages inside the warning policy: {events}')

    def test_the_suite_registers_the_same_list(self, pytestconfig):
        """`pytest.ini` used to say `ignore::DeprecationWarning`, which is the same blanket
        suppression one layer out -- it hid every deprecation raised anywhere in the suite.

        `tests/conftest.py:pytest_configure` now appends `SUPPRESSED` instead. Checked against
        the live ini value rather than the file, so writing the entries back into `pytest.ini` by
        hand would still satisfy it -- what must not happen is the suite filtering by something
        other than this list.
        """
        registered = pytestconfig.getini('filterwarnings')

        assert 'ignore::DeprecationWarning' not in registered, (
            'the suite ignores every DeprecationWarning again, so a new polars or pandas '
            'deprecation would be invisible in CI')
        for category, message, reason in SUPPRESSED:
            expected = f'ignore:{message}:{category.__module__}.{category.__qualname__}'
            assert expected in registered, (
                f'SUPPRESSED entry ({reason}) is not registered with pytest, so the suite and '
                f'the runtime disagree about what is noise')

        # Registered is not the same claim as applied: the ini value is text that pytest has to
        # compile and install per test item. Checked here from inside one, so a pytest change
        # that stopped honouring `addinivalue_line` would fail rather than pass on the string.
        live = {(entry[0], entry[2]) for entry in warnings.filters}
        for category, _, reason in SUPPRESSED:
            assert ('ignore', category) in live, (
                f'SUPPRESSED entry ({reason}) is in the ini value but not in the filters actually '
                f'in force inside a test')

    def test_quiet_preserves_the_wrapped_signature(self):
        """`functools.wraps`, so `delete=` and friends stay introspectable and documented."""
        import inspect

        def sample(self, delete: bool = True) -> None:
            """Docstring."""

        wrapped = quiet(sample)

        assert wrapped.__doc__ == 'Docstring.'
        assert list(inspect.signature(wrapped).parameters) == ['self', 'delete']
