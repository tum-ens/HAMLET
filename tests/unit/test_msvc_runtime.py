"""Unit -- the Windows C++ runtime guard that makes `framework: poi` usable there (#202).

The bug these pin is not a wrong number, it is a process that dies with an access violation at a
location that moves between runs. So the assertions are about *which DLL serves the process* and
*whether HAMLET refuses to solve against a bad one* -- the two things that were unmeasured while
the issue was open.

The interesting failure mode can only be provoked in a fresh interpreter, because the loader
decision is made once per process and cannot be undone. `test_a_stale_runtime_is_refused` therefore
runs a subprocess; everything else can be checked in-process.
"""
import subprocess
import sys
import textwrap

import pytest

from hamlet import msvc_runtime

windows_only = pytest.mark.skipif(sys.platform != 'win32',
                                  reason='the MSVCP140 loader race is a Windows-only problem')


@windows_only
def test_importing_hamlet_claims_the_msvcp140_name_for_a_supported_runtime():
    """The whole fix in one assertion: after `import hamlet`, the loaded CRT is new enough.

    `import hamlet` has already happened by the time this module is collected, so this checks the
    state the fix is responsible for rather than calling the function again -- calling it again
    would pass even if `hamlet/__init__.py` stopped calling it.
    """
    loaded = msvc_runtime.loaded_msvcp140()
    assert loaded is not None, 'no MSVCP140.dll is loaded at all'
    path, version = loaded
    assert version is not None, f'{path} has no version resource'
    assert version[:2] >= msvc_runtime.MINIMUM_MSVCP140, (
        f'{path} is {version}, older than the {msvc_runtime.MINIMUM_MSVCP140} that '
        f"highsbox' HiGHS needs; see hamlet/msvc_runtime.py")


@windows_only
def test_the_guard_passes_once_a_supported_runtime_is_loaded():
    assert msvc_runtime.describe_unsupported_msvcp140() is None


def test_the_module_is_inert_off_windows():
    """Every entry point must be a no-op elsewhere, so importing HAMLET stays portable."""
    if sys.platform == 'win32':
        pytest.skip('this asserts the non-Windows branch')
    assert msvc_runtime.ensure_supported_msvcp140() is None
    assert msvc_runtime.loaded_msvcp140() is None
    assert msvc_runtime.describe_unsupported_msvcp140() is None


@windows_only
def test_hamlet_wins_the_race_against_pandas():
    """`import hamlet` must beat `pandas` to the name, in a fresh interpreter.

    Ordering inside `hamlet/__init__.py` is the entire mechanism, and it is one edit away from
    being lost -- moving the call below the `Creator` import silently reintroduces #202. Checked in
    a subprocess because this one has already been decided in the parent.
    """
    script = textwrap.dedent("""
        import hamlet
        from hamlet import msvc_runtime
        import pandas  # noqa: F401  -- would claim the name for pyarrow's copy if it got there first
        path, version = msvc_runtime.loaded_msvcp140()
        print(version[0], version[1], path)
    """)
    done = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    major, minor, path = done.stdout.split(maxsplit=2)
    assert (int(major), int(minor)) >= msvc_runtime.MINIMUM_MSVCP140, (
        f'pandas won the race: {path.strip()}')


@windows_only
@pytest.mark.solver
def test_a_stale_runtime_is_refused_rather_than_solved_against():
    """Losing the race must raise, not corrupt the process.

    This is the assertion that would have turned #202 from a moving access violation into a bug
    report. It deliberately provokes the bad ordering -- `pandas` first -- and requires a clean
    `RuntimeError` naming the offending DLL. A subprocess, because a process that has bound the
    wrong CRT is not safe to keep using.
    """
    script = textwrap.dedent("""
        import pandas  # noqa: F401  -- claims MSVCP140 for pyarrow's old private copy
        from hamlet import msvc_runtime
        from hamlet.executor.utilities.controller import poi_solver

        if msvc_runtime.describe_unsupported_msvcp140() is None:
            print('NO_STALE_RUNTIME_AVAILABLE')
        else:
            try:
                poi_solver.get_solver_module('highs')
            except RuntimeError as exc:
                print('REFUSED', 'msvcp140' in str(exc).lower())
            else:
                print('NOT_REFUSED')
    """)
    done = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, (
        f'the subprocess did not exit cleanly -- exit {done.returncode}, which for this test most '
        f'likely means the access violation of #202:\n{done.stderr[-2000:]}')
    output = done.stdout.strip()
    if output == 'NO_STALE_RUNTIME_AVAILABLE':
        pytest.skip('this environment has no MSVCP140 older than the minimum to provoke it with')
    assert output == 'REFUSED True', output
