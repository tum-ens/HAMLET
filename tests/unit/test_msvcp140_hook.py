"""Unit -- the `.pth` startup hook that lets `framework: poi` survive a pandas-first import (#202).

This is the delicate part of making PyOptInterface the default, and not because the mechanism is
hard: a `.pth` executes in **every** Python process in the environment, including ones that will
never touch HAMLET. So the tests here are weighted towards what it must *not* do -- raise, cost
much, or do anything at all off Windows -- rather than towards the one thing it must.

`packaging/hamlet_msvcp140_hook.py` is installed at the top level of site-packages rather than
inside the package, so it is imported by its installed name. That import failing is itself a
result: it means the wheel/editable `force-include` in `pyproject.toml` has been lost, and with it
the fix, silently.
"""
import os
import subprocess
import sys
import textwrap

import pytest

hook = pytest.importorskip(
    'hamlet_msvcp140_hook',
    reason='the startup hook is not installed; re-run `uv sync` to pick up the force-include')

windows_only = pytest.mark.skipif(sys.platform != 'win32',
                                  reason='the MSVCP140 loader race is a Windows-only problem')


def run(script, env=None):
    """A fresh interpreter, because the loader decision is made once per process."""
    done = subprocess.run([sys.executable, '-c', textwrap.dedent(script)],
                          capture_output=True, text=True, timeout=300,
                          env={**os.environ, **(env or {})})
    assert done.returncode == 0, (
        f'exit {done.returncode} -- for these tests that most likely means the access violation '
        f'of #202:\n{done.stderr[-2000:]}')
    return done.stdout.strip()


def test_the_pth_is_installed_next_to_the_hook():
    """Both halves ship, or neither works.

    The `.pth` is what makes the module run; the module is what the `.pth` calls. `pyproject.toml`
    force-includes them into the wheel *and* the editable target, and it is the editable one that
    every contributor and CI job actually exercises -- so losing it would go unnoticed here while
    breaking real installs.
    """
    directories = {os.path.dirname(hook.__file__)}
    assert any(os.path.isfile(os.path.join(d, 'hamlet-msvcp140.pth')) for d in directories), (
        f'hamlet_msvcp140_hook.py is installed in {directories} but hamlet-msvcp140.pth is not; '
        f'check [tool.hatch.build.targets.*.force-include] in pyproject.toml')


def test_the_hook_is_inert_off_windows():
    if sys.platform == 'win32':
        pytest.skip('this asserts the non-Windows branch')
    assert hook.claim() is None


def test_the_opt_out_is_honoured():
    """Without this the suite could not provoke the bad ordering at all -- see
    `tests/unit/test_msvc_runtime.py::test_a_stale_runtime_is_refused_rather_than_solved_against`,
    which depends on being able to turn the hook off."""
    previous = os.environ.get(hook.OPT_OUT)
    os.environ[hook.OPT_OUT] = '1'
    try:
        assert hook.claim() is None
    finally:
        if previous is None:
            del os.environ[hook.OPT_OUT]
        else:
            os.environ[hook.OPT_OUT] = previous


@windows_only
def test_it_wins_the_race_when_pandas_is_imported_first():
    """The residual #202 left behind, and the reason this module exists.

    `import pandas` with no HAMLET import anywhere: before the hook this handed the process
    `pyarrow`'s 14.28 copy and made `framework: poi` unusable, which was tolerable only while
    `poi` was opt-in.
    """
    output = run("""
        import pandas  # noqa: F401  -- the ordering that used to lose
        import hamlet_msvcp140_hook
        runtime = hamlet_msvcp140_hook._load_msvc_runtime()
        path, version = runtime['loaded_msvcp140']()
        print(version[0], version[1], runtime['describe_unsupported_msvcp140']() is None, path)
    """)
    major, minor, supported, path = output.split(maxsplit=3)
    minimum = hook._load_msvc_runtime()['MINIMUM_MSVCP140']
    assert (int(major), int(minor)) >= minimum, f'pandas won the race: {path}'
    assert supported == 'True', path


@windows_only
@pytest.mark.solver
def test_a_pandas_first_process_can_actually_solve():
    """The end-to-end statement, which the DLL-identity check above does not quite make.

    A supported runtime being loaded is necessary, not sufficient -- the failure this prevents is
    a corrupted process during a solve, so the test solves.
    """
    output = run("""
        import pandas  # noqa: F401
        import pyoptinterface as poi
        from hamlet.executor.utilities.controller.poi_solver import create_model

        model = create_model('highs')
        x = model.add_variable(lb=0, ub=3)
        model.set_objective(1.0 * x, poi.ObjectiveSense.Maximize)
        model.optimize()
        print(model.get_value(x))
    """)
    assert float(output) == pytest.approx(3.0)


@windows_only
def test_the_same_process_loses_the_race_without_the_hook():
    """The control: the two tests above pass *because of* the hook.

    Without it, this machine must actually be capable of loading the bad runtime -- otherwise
    those tests would pass on a machine where #202 could never happen, and would keep passing if
    the hook were deleted. Deliberately narrow: that a stale runtime is then *refused* rather than
    solved against is `poi_solver`'s guarantee, and is asserted in `test_msvc_runtime.py` rather
    than restated here.
    """
    output = run("""
        import pandas  # noqa: F401
        import hamlet_msvcp140_hook
        runtime = hamlet_msvcp140_hook._load_msvc_runtime()
        path, version = runtime['loaded_msvcp140']()
        print(version[0], version[1], path)
    """, env={hook.OPT_OUT: '1'})
    major, minor, path = output.split(maxsplit=2)
    minimum = hook._load_msvc_runtime()['MINIMUM_MSVCP140']
    if (int(major), int(minor)) >= minimum:
        pytest.skip(f'nothing here ships an MSVCP140 older than {minimum} to lose the race to')
    assert 'pyarrow' in path.lower() or 'sklearn' in path.lower() or 'learn' in path.lower(), (
        f'lost the race to an unexpected module -- worth knowing about: {path}')


def test_it_survives_hamlet_being_unimportable():
    """A half-installed HAMLET must cost a no-op, not a traceback on every interpreter start.

    `site.addpackage` catches whatever a `.pth` raises and keeps going -- but it prints the
    traceback first, to stderr, for every Python process in the environment. That is the failure
    mode worth guarding: noisy and global, in processes with no connection to HAMLET.
    """
    output = run("""
        import sys
        sys.path[:] = [p for p in sys.path if 'HAMLET' not in p and 'site-packages' not in p]
        sys.path.insert(0, %r)
        import hamlet_msvcp140_hook
        print(hamlet_msvcp140_hook._load_msvc_runtime(), hamlet_msvcp140_hook.claim())
    """ % os.path.dirname(hook.__file__))
    assert output == 'None None', output


def test_a_clean_interpreter_start_is_silent():
    """Nothing on stderr, in the configuration every process in the environment gets.

    Worth its own test because `site` degrades a broken `.pth` to a *printed* traceback rather
    than a failure, so a regression here would never turn anything red -- it would just make every
    command in the environment noisy.
    """
    done = subprocess.run([sys.executable, '-c', 'pass'], capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stderr == '', f'the startup hook printed to stderr:\n{done.stderr[-2000:]}'
