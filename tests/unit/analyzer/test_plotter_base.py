"""Unit -- `PlotterBase.plot_all` reports what failed instead of printing it (#228).

The old implementation caught every exception and printed it, so `Analyzer.plot_all()` could not
fail whatever happened underneath. That is how `plot_electricity_grid_topology` accumulated five
independent breakages without a single test noticing: two undeclared dependencies, a hardcoded
grid filename, a results path missing a directory component, and a pandapower call into a
matplotlib function that no longer exists.

Both halves of the fix are pinned here, because each without the other is a different defect:
every plot must still be *attempted*, and the failures must still be *raised*.
"""
import pytest

from hamlet.analyzer.plotter_base import PlotterBase, decorator_plot_function


class Plotter(PlotterBase):
    """A plotter whose plots do what the test tells them to."""

    def __init__(self, failing=()):
        super().__init__(path={'run': 'nowhere'}, config={}, name_subdirectory='x',
                         data_processor=None)
        self.failing = set(failing)
        self.ran = []

    def _record(self, name):
        self.ran.append(name)
        if name in self.failing:
            raise RuntimeError(f'{name} is broken')

    @decorator_plot_function
    def plot_first(self, **kwargs):
        self._record('plot_first')

    @decorator_plot_function
    def plot_second(self, **kwargs):
        self._record('plot_second')

    @decorator_plot_function
    def plot_third(self, **kwargs):
        self._record('plot_third')

    def helper(self):
        """Not decorated, so `plot_all` must not call it despite the name matching nothing."""
        self.ran.append('helper')


def test_all_plots_run_when_none_fail():
    plotter = Plotter()

    plotter.plot_all(save_path=None)

    assert plotter.ran == ['plot_first', 'plot_second', 'plot_third']


def test_a_failing_plot_is_raised_rather_than_printed():
    """The half that makes the failure visible at all."""
    plotter = Plotter(failing={'plot_second'})

    with pytest.raises(ExceptionGroup) as error:
        plotter.plot_all(save_path=None)

    assert 'plot_second' in error.value.message
    assert [type(sub) for sub in error.value.exceptions] == [RuntimeError]


def test_the_other_plots_still_run_when_one_fails():
    """The half that keeps one broken plot from costing the rest.

    Raising at the first failure instead of collecting would be a different regression, and it is
    the one `GridPlotter.plot_all` had -- it called its plots directly, so the transformer loading
    plot was skipped whenever the topology plot failed, which was always.
    """
    plotter = Plotter(failing={'plot_first'})

    with pytest.raises(ExceptionGroup):
        plotter.plot_all(save_path=None)

    assert plotter.ran == ['plot_first', 'plot_second', 'plot_third']


def test_every_failure_is_reported_not_just_the_first():
    plotter = Plotter(failing={'plot_first', 'plot_third'})

    with pytest.raises(ExceptionGroup) as error:
        plotter.plot_all(save_path=None)

    assert len(error.value.exceptions) == 2
    assert 'plot_first' in error.value.message and 'plot_third' in error.value.message


def test_only_decorated_plots_are_run():
    """`plot_all` selects on the decorator, not on the name, so a helper is not a plot."""
    plotter = Plotter()

    plotter.plot_all(save_path=None)

    assert 'helper' not in plotter.ran
