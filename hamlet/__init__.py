# This must stay the first statement in the package. On Windows it claims the `MSVCP140.dll`
# name for the system C++ runtime before `pandas` -- and through it `pyarrow` -- can hand the
# process an older private copy that `highsbox`' HiGHS cannot run against. The imports below
# reach `pandas`, so anything placed above this line loses the race. See `hamlet/msvc_runtime.py`
# and issue #202. No-op off Windows, and never raises.
from hamlet.msvc_runtime import ensure_supported_msvcp140

ensure_supported_msvcp140()

from hamlet.creator.setup import Creator
from hamlet.executor.setup import Executor
from hamlet.analyzer.setup import Analyzer
