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


#: pandapower treats these as optional, and the Analyzer's grid topology plot calls both --
#: `igraph` to lay the buses out, `plotly` to draw the figure. Neither was declared anywhere, so
#: that plot raised `ImportError` for every user of every version (#227).
PANDAPOWER_SOFT_DEPENDENCIES = ('igraph', 'plotly')


def declared():
    """`(runtime, grouped)` package names, from pyproject rather than from the environment."""
    import tomllib

    pyproject = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))
    runtime = {name.split('==')[0].split('[')[0].strip()
               for name in pyproject['project']['dependencies']}
    grouped = {name.split('==')[0].strip()
               for group in pyproject.get('dependency-groups', {}).values()
               for name in group if isinstance(name, str)}
    return runtime, grouped


@pytest.mark.parametrize('package', PANDAPOWER_SOFT_DEPENDENCIES)
def test_the_analyzers_plotting_dependencies_are_runtime_not_development(package):
    """Asserted against `[project.dependencies]` specifically, and that is the whole point.

    Moving one of these into a dependency *group* is invisible to every other test, because the
    test environment installs the groups: the suite stays green while
    `plot_electricity_grid_topology` raises for everyone who installed HAMLET. That is the state
    this test exists to prevent returning to -- both were undeclared entirely until #227, and
    nothing noticed because `PlotterBase.plot_all` printed the exception instead of raising it
    (#228).
    """
    runtime, grouped = declared()

    assert package in runtime, (
        f'{package} must be a runtime dependency: hamlet/analyzer/grids/ reaches it through '
        f'pandapower on a shipped scenario, so it is not something a user opts into. Found '
        f'instead in dependency groups: {sorted(grouped)}')
    assert package not in grouped, (
        f'{package} is declared twice; the runtime entry is the real one')


def test_matplotlib_stays_below_the_version_pandapower_cannot_use():
    """A ceiling HAMLET carries on pandapower's behalf, like the xarray one above.

    pandapower 2.14.8 calls `matplotlib.cm.get_cmap`, removed in matplotlib 3.9, so on 3.9.0 the
    grid topology plot raised `AttributeError` for every user. pandapower declares only a floor on
    matplotlib, so no resolver enforces this.
    """
    assert requirement('matplotlib') == '==3.8.4', (
        'matplotlib must stay below 3.9 while pandapower 2.14.8 is in use: pandapower calls '
        'matplotlib.cm.get_cmap, which 3.9 removed. Raise it together with pandapower, and re-run '
        'the golden master when you do -- pandapower is the power flow engine.')


def test_the_analyzer_can_actually_draw_a_grid_topology():
    """The capability, not the declaration -- these fail in different situations.

    The tests above fail on the commit that moves a pin; this one fails in an environment where a
    pin did not take, and it is deliberately expressed as the three things the plot needs rather
    than as three imports, because two of the three failures were not import errors.
    """
    import importlib

    for package in PANDAPOWER_SOFT_DEPENDENCIES:
        importlib.import_module(package)

    from matplotlib import cm
    from pandapower.plotting import generic_geodata

    assert getattr(generic_geodata, 'IGRAPH_INSTALLED', True), (
        'pandapower cannot see igraph, so create_generic_coordinates(library="igraph") raises and '
        'the grid topology plot is unavailable')
    assert hasattr(cm, 'get_cmap'), (
        'matplotlib.cm.get_cmap is gone, which pandapower 2.14.8 calls while colouring the '
        'topology figure')


def test_linopy_stays_pinned_so_the_xarray_ceiling_is_reconsidered_deliberately():
    """The pins are a pair. Moving one without the other is the mistake this guards."""
    assert requirement('linopy') == '==0.3.11', (
        'linopy is pinned, and the xarray ceiling above exists because of this exact version. If '
        'you are upgrading linopy, check whether the new release still imports '
        '`xarray.core.rolling` -- if it does not, raise both pins together and update this test '
        'and the comment in pyproject.toml.')
