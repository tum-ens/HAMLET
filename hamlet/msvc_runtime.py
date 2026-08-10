"""Windows only: make a supported `MSVCP140.dll` win the loader race, before anything else can.

### Why this module exists

`highsbox` ships `highs.dll` built with the MSVC 14.43 toolset, and that binary needs a
`MSVCP140.dll` of **14.38 or newer**. It does not vendor one, so it takes whatever the process
already has. Meanwhile `pyarrow` and `scikit-learn` each ship an *unmangled* `msvcp140.dll` of
their own -- 14.28 and 14.32 respectively -- and `import pandas` pulls `pyarrow` in.

The Windows loader resolves an import by **base name against the modules already loaded**. So
whichever `msvcp140.dll` enters the process first serves everybody, `highs.dll` included, and
there is no way to rebind it afterwards. Load `pandas` before HiGHS and `highs.dll` runs against a
C++ runtime five toolsets older than the one it was compiled with; the process then corrupts
memory and dies with an access violation at a location that moves between runs. That is issue
#202, and it is why `framework: poi` could not be used on Windows at all.

The only lever is to be first, which is why this runs at `import hamlet` and not at solver-load
time -- by then `pandas` is long since imported and the race is over. `hamlet/__init__.py` calls
`ensure_supported_msvcp140()` on its first line for that reason; keep it there.

Loading the system C++ runtime ahead of an older bundled one is the supported direction: the
`140` runtime is binary-backward-compatible by design, which is the whole point of a shared
redistributable. It is also measured rather than assumed -- the full suite and the shipped example
run clean with `pyarrow`, `pandas` and `scikit-learn` all bound to the system copy.

### What was measured (Windows 11, Python 3.11.12, highsbox 1.10.0)

`highs.dll` preceded by one `msvcp140.dll`, then a single HiGHS solve, 20 runs each:

    14.28 (pyarrow)   20/20 crash        14.38   0/20
    14.32 (sklearn)   20/20 crash        14.40   0/20
    14.36             19/20 crash        14.44   0/20 (System32)

Controls: a solve with no `msvcp140` preloaded never crashes; preloading a *copy of the system
DLL from a foreign path* never crashes, so the path is irrelevant and the version is the cause;
and importing `pyarrow` **after** the solve never crashes, which is the load-order half.

Only `highs.dll` is exposed. `highspy/_core.pyd` and `pyoptinterface`'s `highs_model_ext.pyd` are
both built with 14.29 and are content with any of these, which is why the linopy path has always
worked on Windows and only the POI path breaks.

**Do not try to derive the threshold from a toolset version.** That reasoning looks right and is
wrong: `highsbox` 1.12.0 is linked with a *newer* toolset (14.44 against 1.10.0's 14.43) and runs
happily against 14.28. Holding PyOptInterface at 0.2.8 and moving only `highsbox` 1.10.0 -> 1.12.0
takes the same experiment from 15/15 crashes to 0/15, and moving only PyOptInterface 0.2.8 -> 0.6.1
leaves it at 15/15 -- so whatever changed is inside the `highsbox` build, and it is not the
compiler version. The pinned 1.10.0 stays because it is the HiGHS `highspy` ships and 3-5x faster
on these models than 1.12+; see issue #202.
"""
import ctypes
import ctypes.wintypes as wt
import os
import sys

# The oldest MSVCP140 that `highsbox`' HiGHS survives, measured on the ladder in the module
# docstring rather than derived from the toolset version. If `highsbox` is ever rebuilt this
# number can move in either direction -- re-measure, do not reason about it.
MINIMUM_MSVCP140 = (14, 38)

_LIBRARY = 'msvcp140.dll'


def _file_version(path):
    """The four-part file version of `path`, or None if it has no version resource."""
    try:
        size = ctypes.windll.version.GetFileVersionInfoSizeW(path, None)
        if not size:
            return None
        buffer = ctypes.create_string_buffer(size)
        ctypes.windll.version.GetFileVersionInfoW(path, 0, size, buffer)
        block = ctypes.c_void_p()
        length = ctypes.c_uint()
        if not ctypes.windll.version.VerQueryValueW(buffer, '\\', ctypes.byref(block),
                                                    ctypes.byref(length)):
            return None
        info = ctypes.cast(block, ctypes.POINTER(ctypes.c_uint32 * 13)).contents
        most, least = info[2], info[3]
        return (most >> 16, most & 0xFFFF, least >> 16, least & 0xFFFF)
    except OSError:
        return None


def loaded_msvcp140():
    """The `MSVCP140.dll` currently serving this process, as `(path, version)`, or None.

    `GetModuleHandleW` is asked by *base name* deliberately: that is the same lookup the loader
    performs when it resolves `highs.dll`'s import, so this reports the module that will actually
    be used rather than the one a path search would find.
    """
    if sys.platform != 'win32':
        return None

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wt.HMODULE
    kernel32.GetModuleFileNameW.argtypes = [wt.HMODULE, wt.LPWSTR, wt.DWORD]

    handle = kernel32.GetModuleHandleW(_LIBRARY)
    if not handle:
        return None
    buffer = ctypes.create_unicode_buffer(1024)
    if not kernel32.GetModuleFileNameW(handle, buffer, 1024):
        return None
    return buffer.value, _file_version(buffer.value)


def ensure_supported_msvcp140():
    """Load the system `MSVCP140.dll` if nothing has claimed the name yet. Never raises.

    Returns `(path, version)` for whichever module holds the name afterwards, or None on a
    non-Windows platform or when no `MSVCP140.dll` could be loaded at all. Callers that care
    whether it is *good enough* compare against `MINIMUM_MSVCP140`; this function deliberately
    does not, because `import hamlet` must not fail over a solver that may never be used.
    """
    if sys.platform != 'win32':
        return None

    already = loaded_msvcp140()
    if already is not None:
        return already

    # System32 explicitly rather than the default search order, which would also consider the
    # interpreter's own directory -- some Python distributions ship an old MSVCP140 there, and
    # picking it up would reintroduce exactly the bug this module exists to prevent.
    directory = ctypes.create_unicode_buffer(wt.MAX_PATH)
    if not ctypes.windll.kernel32.GetSystemDirectoryW(directory, wt.MAX_PATH):
        return None
    path = os.path.join(directory.value, _LIBRARY)

    try:
        ctypes.WinDLL(path)
    except OSError:
        # No Visual C++ redistributable installed. Nothing to do here; `poi_solver` turns this
        # into an actionable error if and when a HiGHS solve is actually requested.
        return None
    return loaded_msvcp140()


def describe_unsupported_msvcp140():
    """A ready-to-raise explanation if the loaded MSVCP140 is too old, else None.

    Kept next to the measurements rather than in `poi_solver` so that the version threshold and
    the text that quotes it cannot drift apart.
    """
    if sys.platform != 'win32':
        return None

    loaded = loaded_msvcp140()
    if loaded is None:
        return (
            f"No {_LIBRARY} is loaded, so the HiGHS shared library cannot run. Install the "
            f"Microsoft Visual C++ Redistributable (version "
            f"{'.'.join(map(str, MINIMUM_MSVCP140))} or newer).")

    path, version = loaded
    if version is None or version[:2] >= MINIMUM_MSVCP140:
        return None

    return (
        f"The C++ runtime serving this process is too old for the HiGHS shared library, so "
        f"solving would corrupt memory rather than fail cleanly.\n"
        f"  loaded:   {'.'.join(map(str, version))} from {path}\n"
        f"  required: {'.'.join(map(str, MINIMUM_MSVCP140))} or newer\n"
        f"Windows resolves this DLL by base name against whatever loaded first, and `pyarrow` "
        f"and `scikit-learn` each ship an old private copy, so importing `pandas` before HAMLET "
        f"hands the process one of those. Fix it by importing `hamlet` first -- it claims the "
        f"name for the system runtime on the first line of `hamlet/__init__.py` -- or use "
        f"`framework: linopy`, which is unaffected. See issue #202.")
