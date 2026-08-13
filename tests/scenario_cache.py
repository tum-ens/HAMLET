"""One scenario run per distinct request, shared between test modules for the session.

Every e2e fixture is `scope='module'`, so two files asking for the *same* run each pay for one.
Across the whole suite that is one duplicated pair -- `test_grid_restrictions` and
`test_golden_master[grid_golden]` make byte-identical requests -- and this is what lets them
share it.

**Worth 70-125 s, and quote that band rather than a level.** Measured over the two modules alone
(`pytest tests/e2e/test_golden_master.py tests/e2e/test_grid_restrictions.py -m "e2e or golden"`),
three pairs with the arm order alternated: 181 -> 110, 187 -> 115, 270 -> 146 s. The two
opposite-order pairs agree to 1 s, so the ordering bias cancels; the third ran on a busier machine
and its delta grew with it, which is what removing a fixed unit of work looks like. The same base
arm spanned 181-270 s on identical work, so the *level* says nothing.

**It saves nothing in CI, and that is not a defect in the cache.** `e2e` and `golden` are
separate jobs running separate pytest processes (`.gitlab-ci.yml`), and the duplicated pair
straddles that boundary: the two runs never co-exist in one session, so there is nothing to
share. The saving is real only for a local `pytest tests -m "e2e or golden"`. Merging the two
jobs would realise it in CI and was considered and rejected -- they currently run in parallel, so
merging trades pipeline wall-clock for a smaller amount of runner compute, and costs the
"which kind of thing broke" separation the CI file argues for.

Sharing a run is exactly how a test stops exercising what it believes it does: ask for `linopy`,
be handed a cached `poi`, and agree with yourself. That is !212 and #206 one layer down. Two
things stop it, and both are load-bearing:

**The key is the whole request, derived mechanically.** `request_key` binds the arguments against
`run_example`'s own signature, so every parameter is part of the key unless it is named in
`NOT_PART_OF_THE_REQUEST`. A parameter added to `run_example` later is therefore keyed *by
default*, with nothing to remember. This is the complement of an allowlist on purpose: three
enumerated constants in this repository (`ROUNDING`, `KEYS`, `AGENT_TABLES`) have each passed by
omission, and the failure mode of this list is the safe one -- naming too little over-keys, which
costs a run, where naming too much would silently merge two different requests.

**Every consumer is checked against the entry it is handed, on a cache hit as much as on a miss.**
The key deciding two requests are the same is not evidence that they are; asserting it against the
key would only ask the key to vouch for itself. So `ScenarioRuns.run` performs two checks that do
not go through the key:

1. the **full request**, compared argument by argument against the one that produced the entry.
   This covers every keyed argument, so a key that lost a field fails at the first consumer it
   mis-serves. It compares two argument sets, never two keys, which is what makes it independent.
2. the **receipt** `BACKEND_PROBE` wrote, for callers that named a backend. Weaker in reach and
   stronger in kind: it is the only check that reports what the run *did* rather than what it was
   asked for, and so the only one that could catch `run_example` failing to apply a switch (#206).

`tests/unit/test_scenario_cache_key.py` breaks the key on purpose and shows both firing.
"""
import inspect
from pathlib import Path

from tests.scenario_run import assert_backend_honoured, run_example

#: Arguments naming *where* a run puts its output rather than *what* it runs. Everything else in
#: `run_example`'s signature is part of the key. See the module docstring for why this is an
#: exclusion list rather than an allowlist.
NOT_PART_OF_THE_REQUEST = frozenset({'base', 'record_backends'})

_SIGNATURE = inspect.signature(run_example)

#: A rename of an excluded parameter must not silently start including it -- that direction is
#: harmless (an extra key field only costs a run) but it would make this list a lie, and the next
#: reader would trust it. Checked at import so it fails at collection rather than mid-suite.
_unknown = NOT_PART_OF_THE_REQUEST - set(_SIGNATURE.parameters)
assert not _unknown, (
    f'NOT_PART_OF_THE_REQUEST names {sorted(_unknown)}, which are not parameters of run_example '
    f'({sorted(_SIGNATURE.parameters)}). Either the parameter was renamed -- update this set -- '
    f'or the name is a typo that has been excluding nothing')


def _hashable(value):
    """A stable, order-significant, hashable rendering of an argument value.

    Order is preserved rather than normalised away: `edits` is applied as a sequence of string
    replacements, so two orderings are two different requests and must not share a run.
    """
    if isinstance(value, Path):
        return f'path:{value.as_posix()}'
    if isinstance(value, dict):
        return ('dict',) + tuple((key, _hashable(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ('seq',) + tuple(_hashable(item) for item in value)
    return value


def request_key(*args, **kwargs):
    """The full request to `run_example`, minus where it writes, as a hashable key.

    Takes the arguments `ScenarioRuns.run` takes -- that is, `run_example`'s without `base`.
    """
    bound = _SIGNATURE.bind(None, *args, **kwargs)
    bound.apply_defaults()
    return tuple(sorted((name, _hashable(value)) for name, value in bound.arguments.items()
                        if name not in NOT_PART_OF_THE_REQUEST))


class Entry:
    """One completed run, and the artefacts a consumer reads it through.

    `request` is the full argument set that produced it, kept verbatim. It is what a later
    consumer is compared against, and keeping it is the reason a lossy *key* cannot go unnoticed:
    the comparison is between two argument sets, never between two keys.
    """

    def __init__(self, base, request, fingerprint, record):
        self.base = base
        self.request = request
        self.fingerprint = fingerprint
        self.record = record

    @property
    def scenario_name(self):
        return self.request['scenario_name']

    @property
    def results(self):
        """The run's results directory, which several tests read instead of the fingerprint."""
        return self.base / 'results' / self.scenario_name


class ScenarioRuns:
    """Session-scoped store of runs, keyed by the request that produced them.

    `runner` and `key` are injected so `tests/unit/test_scenario_cache_key.py` can substitute a
    deliberately broken key and a stub runner, and show a consumer going red rather than
    agreeing with itself. Production callers take the defaults.
    """

    def __init__(self, tmp_path_factory, runner=run_example, key=request_key):
        self._tmp_path_factory = tmp_path_factory
        self._runner = runner
        self._key = key
        self._entries = {}
        #: Every (key, scenario) actually run, in order — a registry of what this session really
        #: executed. Read by `tests/unit/test_scenario_cache_key.py` to prove a collision merged
        #: two requests into one run, which is the premise its mutation test rests on.
        #:
        #: It is *not* read by `test_solver_backend_smoke`'s deferral guard, though it looks like
        #: the right evidence for it: that guard is in the fast tier, where no example runs, so it
        #: would read an empty log and pass. See the docstring there.
        self.log = []

    def run(self, *args, needs_receipt=False, **kwargs):
        """Return the run for this request, running it only if no identical one exists.

        Takes exactly `run_example`'s arguments except `base` and `record_backends`, both of which
        this owns, plus `needs_receipt`. Returns an `Entry`.

        `needs_receipt` asks for `BACKEND_PROBE` even though this caller names no backend --
        `test_grid_examples` needs one to prove the shipped examples run without a commercial
        licence. It is **part of the key**, because turning the probe on changes what the run does:
        it monkeypatches `create_model` and `linopy.Model.solve`. Leaving it out of the key would
        let a probe-off run be served to a caller that needs the receipt, or the reverse -- and the
        reverse is the one that matters, because it would quietly put the probe into the golden
        master's run.
        """
        bound = _SIGNATURE.bind(None, *args, **kwargs)
        bound.apply_defaults()

        # Refused rather than ignored. A shared run owns one receipt, so this is set below; a
        # caller passing its own path would get a file that never appears and no explanation,
        # which is the silent no-op this whole module is built to avoid.
        assert bound.arguments['record_backends'] is None, (
            'record_backends is owned by the run cache, because a shared entry needs one receipt '
            'every consumer reads; ask for `needs_receipt=True` and take `entry.record` instead')

        request = dict(bound.arguments)
        request.pop('base')
        request.pop('record_backends')
        request['needs_receipt'] = needs_receipt

        # `run_example` writes a receipt whenever a backend was named; this adds the opt-in. The
        # runs that ask for neither stay probe-free, which is what keeps the golden master's run
        # byte-for-byte the one that produced the committed reference.
        wants_record = (needs_receipt or request['framework'] is not None
                        or request['solver'] is not None)

        key = (self._key(*args, **kwargs), needs_receipt)
        entry = self._entries.get(key)

        if entry is None:
            base = Path(self._tmp_path_factory.mktemp(f'run_{request["scenario_name"]}_'))
            record = base / 'backends_used.json' if wants_record else None
            call = dict(bound.arguments)
            call.pop('base')
            call['record_backends'] = record
            fingerprint = self._runner(base, **call)
            entry = Entry(base, request, fingerprint, record)
            self._entries[key] = entry
            self.log.append((key, request['scenario_name']))

        self._check_the_entry_matches_this_request(entry, request)
        return entry

    @staticmethod
    def _check_the_entry_matches_this_request(entry, request):
        """What this consumer asked for, checked against the entry it was handed.

        Runs on every call, cache hit included -- that is the whole point.

        Two independent checks, and the difference between them matters:

        - **the full request**, compared argument by argument. This is not the key and does not go
          through it, so a key that lost a field -- the failure this module exists to prevent --
          is caught here on the first consumer it mis-serves, on *every* keyed argument rather
          than the three a receipt can speak to.
        - **the receipt**, which is the stronger evidence where it exists, because it reports what
          the run *did* rather than what it was asked to do. It is the only one of the two that
          could catch `run_example` itself failing to apply a backend (#206). It can only speak
          for a caller that named one.
        """
        if entry.request != request:
            differing = sorted(name for name in request
                               if entry.request.get(name) != request.get(name))
            raise AssertionError(
                f'the run cache served a run whose request differs from this one in {differing}: '
                f'asked for { {name: request[name] for name in differing} }, got '
                f'{ {name: entry.request.get(name) for name in differing} }. Two requests that '
                f'are not the same were given the same key, so one of these tests is not running '
                f'what it believes it is')

        assert entry.results.is_dir(), (
            f'the run of {entry.scenario_name!r} completed but wrote no results directory at '
            f'{entry.results}, so there is nothing for this test to read')

        if request['framework'] is not None or request['solver'] is not None:
            assert_backend_honoured(entry.record, request['framework'], request['solver'])
