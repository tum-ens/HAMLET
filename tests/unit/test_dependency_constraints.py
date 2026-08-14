"""Unit -- the dependency constraints that are load-bearing rather than tidy.

`import hamlet` has failed outright, twice, because of one transitive package. linopy 0.3.11
imports `xarray.core.rolling`, which xarray removed after 2024.6.0; xarray is not HAMLET's
dependency, so nothing declared it, and both times a resolver simply took the newest one.

`pyproject.toml` now pins `xarray==2024.6.0`, which stops `uv lock --upgrade` moving it. That is
not by itself enough, and it is worth being precise about why: **linopy declares only a floor.**
Every release from 0.3.13 to 0.9.0 requires `xarray>=2024.2.0`, never an upper bound. So raising
the linopy pin does not conflict with ours and does not fail resolution -- the ceiling is one
HAMLET carries on linopy's behalf, and a resolver will never enforce it for us.

These tests are that enforcement. They fail on the commit that relaxes the pin, rather than on
whoever next creates a fresh environment.
"""
import re
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[2] / 'pyproject.toml'


def requirement(name):
    """The version specifier `pyproject.toml` declares for `name`, as written."""
    text = PYPROJECT.read_text(encoding='utf-8')
    match = re.search(rf'^\s*"{re.escape(name)}(==[^"]+)"', text, re.MULTILINE)
    assert match, f'{name} is not an exactly-pinned dependency in pyproject.toml'
    return match.group(1)


def test_linopy_can_import_the_xarray_submodule_it_needs():
    """The failure itself, asserted directly rather than via a version number.

    This is what actually broke: linopy's import of `xarray.core.rolling`. Asserting the module
    resolves means the test still means something if the pins are ever restructured, and it names
    the right two packages when it fails.
    """
    pytest.importorskip('xarray')
    import importlib

    try:
        importlib.import_module('xarray.core.rolling')
    except ModuleNotFoundError as exc:  # pragma: no cover - the whole point is that it does not
        import xarray
        pytest.fail(
            f'xarray {xarray.__version__} has no `xarray.core.rolling` ({exc}). linopy 0.3.11 '
            f'imports it at module scope, so `import hamlet` fails outright with a traceback '
            f'naming neither package. xarray removed it after 2024.6.0; see the pin in '
            f'pyproject.toml.')


def test_xarray_stays_pinned_at_the_last_version_linopy_can_use():
    """linopy declares no upper bound on xarray, so this pin is the only thing holding it."""
    assert requirement('xarray') == '==2024.6.0', (
        'xarray must stay pinned at 2024.6.0 while linopy 0.3.11 is in use: linopy imports '
        '`xarray.core.rolling`, removed after that release. linopy declares only `xarray>=...`, '
        'so no resolver will catch this for you.')


def test_igraph_is_a_runtime_dependency_and_not_a_development_one():
    """The Analyzer's grid topology plot needs it, so a user must get it (#227).

    It is asserted against `[project.dependencies]` specifically, because moving it back into a
    dependency group is invisible to every other test: the test environment installs the groups,
    so the suite stays green while `plot_electricity_grid_topology` raises `ImportError` for
    everyone who installed HAMLET. That is the state this test exists to prevent returning to --
    it was undeclared entirely until #222, and nothing noticed because `PlotterBase.plot_all`
    swallows the exception (#228).
    """
    import tomllib

    pyproject = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))
    runtime = {name.split('==')[0].split('[')[0].strip()
               for name in pyproject['project']['dependencies']}
    grouped = {name.split('==')[0].strip()
               for group in pyproject.get('dependency-groups', {}).values()
               for name in group if isinstance(name, str)}

    assert 'igraph' in runtime, (
        'igraph must be a runtime dependency: hamlet/analyzer/grids/grid_data_processor.py calls '
        'pandapower create_generic_coordinates(library="igraph"), which is shipped Analyzer code '
        f'on a shipped scenario. Found instead in dependency groups: {sorted(grouped)}')
    assert 'igraph' not in grouped, 'igraph is declared twice; the runtime entry is the real one'


def test_the_analyzer_can_import_what_its_topology_plot_needs():
    """The dependency, asserted as the capability rather than as a line in a file.

    Complements the declaration check above: that one fails on the commit that moves the pin, this
    one on an environment where the pin did not take.
    """
    import importlib

    importlib.import_module('igraph')

    from pandapower.plotting import generic_geodata

    assert getattr(generic_geodata, 'IGRAPH_INSTALLED', True), (
        'pandapower cannot see igraph, so create_generic_coordinates(library="igraph") will raise '
        'and the Analyzer grid topology plot is unavailable')


def test_linopy_stays_pinned_so_the_xarray_ceiling_is_reconsidered_deliberately():
    """The pins are a pair. Moving one without the other is the mistake this guards."""
    assert requirement('linopy') == '==0.3.11', (
        'linopy is pinned, and the xarray ceiling above exists because of this exact version. If '
        'you are upgrading linopy, check whether the new release still imports '
        '`xarray.core.rolling` -- if it does not, raise both pins together and update this test '
        'and the comment in pyproject.toml.')
