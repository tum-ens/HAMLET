"""Reducing the Analyzer's data processors to a comparable fingerprint.

Used by `tests/e2e/test_analyzer_processors.py`, which explains what is pinned and why.

`MPLBACKEND` is set before anything here imports the analyzer, because importing
`grid_data_processor` pulls in `pandapower.plotting.plotly` and therefore `matplotlib.pyplot`.
Setting it inside a fixture would be too late -- collection has already imported this module.

**Nothing is enumerated.** `processor_classes` walks `hamlet.analyzer` for `DataProcessorBase`
subclasses and `processor_methods` reads `process_*` off each, so a new processor -- or a new
processor *class* -- is pinned by default. Three enumerated constants in this repository
(`ROUNDING`, `KEYS`, `AGENT_TABLES`) have each passed by omission, and an earlier draft of this
module reintroduced the shape one level up by listing the three classes.

Three properties of the reduction are load-bearing. Each is pinned by
`tests/unit/test_analyzer_outputs.py` rather than only exercised inside a minutes-long job:

1. **Every recorded statistic is order-independent**, because the row order is not reproducible:
   two of the six group by Categorical columns, whose category order depends on the order values
   were first seen in the process. Filed as #229, not pinned. Index summaries are taken after
   casting every level to `str` and sorting, for the same reason.
2. **A column holding numbers as `object` is still numeric.** `process_total_balancing` builds its
   frame with `pd.DataFrame(index=..., columns=['cost', 'revenue'])`, so both are `object` dtype
   and `select_dtypes('number')` skips them. Columns are classified by whether `pd.to_numeric`
   succeeds.
3. **An empty return is a failure, not a pass** -- see `emptiness_complaints`.
"""
import hashlib
import importlib
import inspect
import os
import pkgutil
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import hamlet.analyzer
from hamlet.analyzer.data_processor_base import DataProcessorBase

#: Above this many distinct values a text column is recorded as a digest rather than a list. The
#: list is worth having where it fits -- `agent_description` swapping with `plant_description` is
#: invisible to any count -- and unreadable where it does not.
MAX_LISTED_VALUES = 30


def processor_classes():
    """Every `DataProcessorBase` subclass under `hamlet.analyzer`, by class name.

    Discovered by walking the package rather than listed, so a fourth data processor is pinned
    without anyone remembering to add it here.
    """
    found = {}
    for module in pkgutil.walk_packages(hamlet.analyzer.__path__, 'hamlet.analyzer.'):
        for _, obj in inspect.getmembers(importlib.import_module(module.name), inspect.isclass):
            if issubclass(obj, DataProcessorBase) and obj is not DataProcessorBase:
                found[obj.__name__] = obj
    return found


def processor_methods(cls):
    """Every `process_*` method on a data-processor class, discovered rather than listed."""
    return sorted(name for name in dir(cls)
                  if name.startswith('process_') and callable(getattr(cls, name)))


def processor_names():
    """Every processor as a flat `<class>.<method>` set, which is how references key them."""
    return {f'{name}.{method}'
            for name, cls in processor_classes().items() for method in processor_methods(cls)}


def _as_numeric(series):
    """The column as numbers, or None if it does not hold numbers. See the module docstring."""
    import pandas as pd

    if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype.kind in 'Mm':
        return None
    try:
        converted = pd.to_numeric(series, errors='raise')
    except (ValueError, TypeError):
        return None
    if converted.dtype == bool:
        converted = converted.astype(int)
    return converted if converted.dtype.kind in 'iuf' else None


def _finite(value):
    """A JSON-safe float. NaN and infinity are recorded as null rather than crashing json.dump."""
    import math

    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _index_order(index):
    """Positions that put the rows in stringified-index order, or None if labels repeat.

    This is the sort key that makes an order-sensitive statistic *reproducible*: it is derived
    from the index's string form, never from a Categorical's local integer encoding (#229). None
    when a label occurs twice, because the order within that group would then be the source order,
    which is exactly the thing that is not reproducible.
    """
    labels = _index_labels(index)
    if len(set(labels)) != len(labels):
        return None
    return [position for _, position in sorted(zip(labels, range(len(labels))))]


def _value_types(series):
    """The Python types an `object` column actually holds, sorted. See `describe_column`."""
    return sorted({type(value).__name__ for value in series.tolist()})


def _index_labels(index):
    return ['|'.join(str(part) for part in (row if isinstance(row, tuple) else (row,)))
            for row in index.tolist()]


def describe_column(series, order=None):
    """One column, reduced to statistics that are reproducible across runs.

    `min`/`max` are None exactly when the column holds no value at all, which is what
    `emptiness_complaints` reads. `sum` cannot serve that purpose: pandas sums an all-NaN column
    to `0.0`, so a column of nothing is indistinguishable from a column of zeros by its total.

    **`ordered` is what makes a value/index misalignment visible.** Sum, min, max and a distinct
    count are all invariant under any permutation of values against the index, so without it the
    whole family of "right numbers, wrong row" defects -- positional indexing written back onto a
    sorted frame, an off-by-one interval convention, a bad join -- is unassertable. It is a
    position-weighted total taken in stringified-index order: reproducible in the face of #229
    (the weights come from the index's string form, not a Categorical encoding) and compared with
    the same relative tolerance as every other float, rather than being a digest that would flake
    on a last-place difference between platforms. Text columns take a digest instead, where exact
    comparison is safe.

    `dtype` is the numpy kind character rather than the dtype name, so it is stable across
    platforms while still catching a numeric result column silently becoming strings -- which
    matplotlib would plot as a categorical axis. For an `object` column the kind is `O` either
    way, so `value_types` records the Python types actually held: the topology processor fills
    `power_description` with floats into a column initialised with a string, and that is exactly
    the shape where the kind cannot see the change.
    """
    numeric = _as_numeric(series)
    if numeric is not None:
        present = numeric.dropna()
        values = numeric.fillna(0.0).tolist()
        ordered = None
        if order is not None:
            ordered = _finite(sum((rank + 1) * float(values[position])
                                  for rank, position in enumerate(order)))
        described = {
            'kind': 'numeric',
            'dtype': series.dtype.kind,
            'nulls': int(numeric.isna().sum()),
            'sum': _finite(numeric.sum()),
            'min': _finite(present.min()) if len(present) else None,
            'max': _finite(present.max()) if len(present) else None,
            'ordered': ordered,
        }
        if series.dtype.kind == 'O':
            described['value_types'] = _value_types(series)
        return described

    raw = ['' if value is None else str(value) for value in series.tolist()]
    values = sorted(set(raw))
    described = {'kind': 'text', 'dtype': series.dtype.kind, 'distinct': len(values)}
    if series.dtype.kind == 'O':
        described['value_types'] = _value_types(series)
    if len(values) <= MAX_LISTED_VALUES:
        described['values'] = values
    else:
        described['digest'] = hashlib.sha256('\x00'.join(values).encode('utf-8')).hexdigest()[:16]
    if order is not None:
        described['ordered_digest'] = hashlib.sha256(
            '\x00'.join(raw[position] for position in order).encode('utf-8')).hexdigest()[:16]
    return described


def describe_index(index):
    """The index, summarised after casting every level to `str` and sorting.

    The index is the figure's x-axis (or, for the grouped market frames, its categories), so a
    processor that loses a timestep or an agent has to be visible here even though no column
    changed. `nlevels` is recorded alongside, so two different index shapes cannot flatten to the
    same summary.
    """
    labels = sorted(_index_labels(index))
    return {
        'nlevels': int(index.nlevels),
        'length': int(len(index)),
        'distinct': len(set(labels)),
        'min': labels[0] if labels else None,
        'max': labels[-1] if labels else None,
    }


def describe_frame(frame):
    """One DataFrame or Series, as row count, index summary and per-column statistics.

    The index order is computed once and handed to every column, so all of them are weighted by
    the same ordering -- otherwise a column could not be compared against its neighbours.
    """
    import pandas as pd

    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=frame.name if frame.name is not None else 'value')

    order = _index_order(frame.index)
    return {
        'rows': int(len(frame)),
        'index': describe_index(frame.index),
        'columns': {str(name): describe_column(frame[name], order) for name in frame.columns},
    }


def _normalise_key(key, results_root):
    """A dict key as a stable label, with the run's temp path reduced to a relative one.

    `MarketDataProcessor._get_market_transactions_for_scenario` keys its result by the absolute
    directory it found `market_transactions.ft` in, so without this every reference would record
    the `tmp_path` of the run that produced it and match nothing afterwards.
    """
    label = str(key)
    try:
        relative = Path(label).resolve().relative_to(Path(results_root).resolve())
    except (ValueError, OSError):
        return label
    return relative.as_posix() or '.'


def _record(flattened, label, description):
    """Store one leaf, refusing to overwrite another. A collision would lose coverage silently."""
    assert label not in flattened, (
        f'two outputs reduced to the same label {label!r}, so one would silently replace the '
        f'other and the reference would compare fewer outputs than the processor returned')
    flattened[label] = description


def reduce_output(value, results_root, prefix=''):
    """Flatten one processor's return value into `{label: description}`.

    Handles the four shapes the six processors return: a nested dict, a DataFrame, a Series, and a
    `pandapowerNet`. **The net is tested for first and that ordering is load-bearing** --
    `pandapowerNet` subclasses `dict`, so the dict branch would otherwise walk it as a mapping and
    reduce every private cache and empty `res_*` table alongside the tables the processor actually
    filled in.
    """
    import pandas as pd
    import pandapower as pp

    flattened = {}

    if isinstance(value, pp.auxiliary.pandapowerNet):
        for name in sorted(value.keys()):
            table = value[name]
            if isinstance(table, pd.DataFrame) and not name.startswith('_') and not table.empty:
                _record(flattened, f'{prefix}/{name}', describe_frame(table))
        return flattened

    if isinstance(value, dict):
        for key, item in value.items():
            label = _normalise_key(key, results_root)
            for nested, description in reduce_output(item, results_root,
                                                     f'{prefix}/{label}').items():
                _record(flattened, nested, description)
        return flattened

    if isinstance(value, (pd.DataFrame, pd.Series)):
        _record(flattened, prefix, describe_frame(value))
        return flattened

    if value is None:
        return flattened

    _record(flattened, prefix, {'rows': 0, 'index': None, 'columns': {},
                                'repr': type(value).__name__})
    return flattened


def emptiness_complaints(fingerprint):
    """Every reason `fingerprint` is not evidence that its processors produced anything.

    Returned as a list rather than raised, so one call reports all of them and so the *same*
    predicate can be applied to a live run and to the committed reference. Both matter and neither
    substitutes for the other: checking only the live output lets a reference regenerated from a
    broken run pass forever, and checking only the reference lets an empty run be compared against
    nothing and agree.

    Four conditions, because each can hold while the next fails. The last is the subtle one: a
    numeric column that is entirely null still has a `sum` -- pandas totals all-NaN to `0.0` -- so
    presence of a total says nothing. `min` is None exactly when there was no value, and both
    committed references contain such columns (`bus_geodata.coords`), so this is a live
    distinction rather than a hypothetical one.
    """
    complaints = []
    for processor in sorted(processor_names()):
        outputs = fingerprint.get(processor)
        if not outputs:
            complaints.append(f'{processor} returned nothing at all')
            continue
        if not any(entry.get('rows', 0) > 0 for entry in outputs.values()):
            complaints.append(
                f'{processor} returned {len(outputs)} output(s) and every one of them is empty, '
                f'so any assertion over its rows passes by vacuity')
            continue
        numeric = [statistics for entry in outputs.values()
                   for statistics in entry.get('columns', {}).values()
                   if statistics.get('kind') == 'numeric']
        if not numeric:
            complaints.append(
                f'{processor} produced rows but not one numeric column, so nothing it returns '
                f'can be compared as a number')
        elif not any(statistics.get('min') is not None for statistics in numeric):
            complaints.append(
                f'{processor} produced {len(numeric)} numeric column(s) and every one of them is '
                f'entirely null, so there is no value to compare')
    return complaints


def run_processors(analyzer, results_root):
    """Every discovered processor, called against `analyzer`, reduced to a comparable fingerprint.

    Returns `{'<class>.<method>': {label: description}}`. The processors are found on the
    `analyzer` itself -- every attribute carrying a `data_processor` -- and checked against the
    classes discovered in the package, so a processor class that exists but is never wired into
    `Analyzer` fails here rather than being quietly pinned as absent.

    Exceptions are deliberately not caught: `PlotterBase.plot_all` already swallows every one of
    them (#228), which is how two of these could rot to the point of raising on every machine
    without a test noticing. Here a raise is a test failure.
    """
    wired = {}
    for plotter in vars(analyzer).values():
        processor = getattr(plotter, 'data_processor', None)
        if isinstance(processor, DataProcessorBase):
            wired[type(processor).__name__] = processor

    discovered = set(processor_classes())
    assert set(wired) == discovered, (
        f'the Analyzer wires up {sorted(wired)} but hamlet.analyzer defines {sorted(discovered)}; '
        f'unwired: {sorted(discovered - set(wired))}, unknown: {sorted(set(wired) - discovered)}')

    fingerprint = {}
    for name, processor in sorted(wired.items()):
        for method in processor_methods(type(processor)):
            fingerprint[f'{name}.{method}'] = reduce_output(
                getattr(processor, method)(), results_root)
    return fingerprint
