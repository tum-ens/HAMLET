"""Windows only: claim the `MSVCP140.dll` name at interpreter startup, before any user import.

Installed at the top level of `site-packages` alongside `hamlet-msvcp140.pth`, which is what
invokes `claim()`. It is not part of the `hamlet` package and must not become one: the whole
point is to run *without* importing `hamlet`, whose `__init__` reaches `pandas` and would cost
every Python process in the environment HAMLET's full import time.

### Why a startup hook rather than the `import hamlet` call

`hamlet/__init__.py` already calls `ensure_supported_msvcp140()` on its first line, and that is
enough for anyone who imports HAMLET first. It cannot help a script that imports `pandas` first,
because by then `pyarrow`'s private 14.28 copy owns the name and the loader will not rebind it.
That residual was acceptable while `framework: poi` was opt-in and is not now that it is the
default -- see `hamlet/msvc_runtime.py` for the measurements, and issue #202.

A `.pth` runs inside `site` before any user code, which is the only place early enough.

### What this is not

**It is not the safety mechanism.** `poi_solver` still refuses to solve against a stale runtime
and names the offending DLL, and that check is what makes a lost race a clear error instead of a
moving access violation. This hook only makes the race easier to win. So every failure here is
swallowed: a hook that cannot run degrades to the pre-existing "import `hamlet` first" rule,
whereas a hook that raises would print a traceback on *every* interpreter start in the
environment, including ones that will never touch HAMLET.

Set `HAMLET_NO_MSVCP140_HOOK=1` to disable it. The test suite uses that to provoke the bad
ordering deliberately; a user would only need it if loading the system C++ runtime ever conflicts
with something else in their environment.

### What it costs, measured

`python -X importtime -c pass`, cumulative for `site`, on Windows:

    hook absent      8.2 ms
    hook opted out  10.5 ms     (+2.4 -- this module and nothing else)
    hook active     ~48 ms      (+40, of which ~26 is `ctypes`)

Off Windows the `.pth` short-circuits on `sys.platform` before touching the disk, so the cost
there is one string comparison.

The ~26 ms of `ctypes` is a floor, not slack: there is no way to ask which `MSVCP140.dll` a
process holds, or to load one, without it. What was removed is the rest -- see `_load_msvc_runtime`.

**A deferred variant was considered and rejected.** A `sys.meta_path` observer that claimed the
name on first sight of `pandas`/`pyarrow`/`scikit-learn` would cost +2.4 ms instead of +40, and
charge the `ctypes` only to processes that import those. It was not taken because the trigger list
is a blocklist: it would have to enumerate every wheel shipping an unmangled `msvcp140.dll`, which
is exactly the set #202 had to discover empirically and which the next release of any dependency
can extend. Being unconditionally first needs no such list. 40 ms in a virtual environment whose
purpose is multi-hour simulations is the cheaper side of that trade.
"""
import os
import sys

OPT_OUT = 'HAMLET_NO_MSVCP140_HOOK'

#: `__name__` for the executed copy. Deliberately not `hamlet.msvc_runtime`: nothing is inserted
#: into `sys.modules`, so the real import later gets a clean package rather than finding a
#: half-built one someone else put there.
_MODULE_NAME = 'hamlet_msvcp140_hook._msvc_runtime'

_SOURCE = os.path.join('hamlet', 'msvc_runtime.py')


def _load_msvc_runtime():
    """`hamlet/msvc_runtime.py`, executed without executing `hamlet/__init__.py`.

    Returns the resulting namespace, or None if HAMLET is not on `sys.path` as a directory.

    Executed by path rather than imported so that the version threshold and the loader logic keep
    a single home in the package. The obvious spelling of that is `importlib.util.find_spec` plus
    `spec_from_file_location`, and it was measured and rejected: importing `importlib.util` costs
    **25.8 ms** at interpreter start (`python -X importtime`), more than the `ctypes` this hook
    genuinely needs, and it is paid by every Python process in the environment rather than only by
    HAMLET's. A `sys.path` walk needs nothing that is not already imported at startup.

    Both install layouts put the package on `sys.path` as a plain directory -- site-packages for a
    wheel, the repository root for the editable install this repository uses -- and `site` appends
    a site directory before it processes the `.pth` files in it, so the entry is present by the
    time this runs. Anything more exotic (a zipimport, a frozen interpreter) simply finds nothing
    and the hook no-ops, which is the documented degradation.
    """
    for entry in sys.path:
        if not entry:
            continue
        path = os.path.join(entry, _SOURCE)
        if not os.path.isfile(path):
            continue
        with open(path, 'rb') as handle:
            source = handle.read()
        namespace = {'__name__': _MODULE_NAME, '__file__': path}
        exec(compile(source, path, 'exec'), namespace)
        return namespace
    return None


def claim():
    """Load a supported system `MSVCP140.dll` if nothing holds the name yet.

    Returns `(path, version)` for whichever module holds the name afterwards, or None when the
    hook did not apply -- off Windows, opted out, HAMLET not locatable, or no C++ redistributable
    installed. Never raises; see the module docstring for why that is not laziness.
    """
    if sys.platform != 'win32' or os.environ.get(OPT_OUT):
        return None

    try:
        namespace = _load_msvc_runtime()
        if namespace is None:
            return None
        return namespace['ensure_supported_msvcp140']()
    except Exception:
        return None
