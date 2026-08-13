"""Unit -- the reduction the Analyzer's reference is built from.

`tests/analyzer_outputs.py` only runs inside an `e2e` job, so without this file every property it
relies on would be checked minutes at a time and never in the fast tier. Each test here is a trap
that has already cost this repository, or the Analyzer, a real defect.
"""
import pandas as pd
import pytest

from tests.analyzer_outputs import (describe_column, describe_frame, describe_index,
                                    emptiness_complaints, processor_classes, processor_methods,
                                    processor_names, reduce_output)


class TestTheProcessorSetIsDiscovered:
    """Nothing is enumerated -- the package is the list, and the classes are the list of methods."""

    def test_all_six_processors_are_found(self):
        assert processor_names() == {
            'AgentDataProcessor.process_all_meters_data',
            'GridDataProcessor.process_electricity_grid_topology',
            'GridDataProcessor.process_electricity_transformer_loading',
            'MarketDataProcessor.process_agent_balancing',
            'MarketDataProcessor.process_average_pricing',
            'MarketDataProcessor.process_total_balancing',
        }

    def test_the_classes_are_found_by_walking_the_package(self):
        """A fourth data processor is discovered without anyone adding it to a list."""
        assert set(processor_classes()) == {
            'AgentDataProcessor', 'GridDataProcessor', 'MarketDataProcessor'}

    def test_a_new_processor_method_would_be_discovered(self):
        """The discovery is by prefix on the class, so adding one cannot be missed.

        Asserted against a stand-in, so the test does not depend on the Analyzer growing a seventh
        method -- what is checked is that `processor_methods` reads the class, not a constant.
        """
        class Grown:
            def process_all_meters_data(self):
                ...

            def process_something_new(self):
                ...

            def helper(self):
                ...

        assert processor_methods(Grown) == ['process_all_meters_data', 'process_something_new']

    def test_every_class_contributes_at_least_one(self):
        """A class whose methods were renamed out of the convention would pin nothing."""
        for name, cls in processor_classes().items():
            assert processor_methods(cls), f'{name} contributes no process_* method'


class TestNumbersHeldAsObjectsAreStillNumbers:
    """`process_total_balancing` builds an object-dtype frame; `select_dtypes` skips it."""

    def test_an_object_column_of_numbers_is_numeric(self):
        frame = pd.DataFrame(index=['grid', 'levies'], columns=['cost', 'revenue'])
        frame.loc['grid'] = [-3.0, 1.0]
        frame.loc['levies'] = [-1.0, 0.5]

        described = describe_frame(frame)

        assert described['columns']['cost']['kind'] == 'numeric', (
            'the cost column of process_total_balancing is object dtype; treating it as text '
            'would leave the Analyzer output that most directly becomes a figure unasserted')
        assert described['columns']['cost']['sum'] == pytest.approx(-4.0)
        assert described['columns']['revenue']['max'] == pytest.approx(1.0)

    def test_a_genuine_text_column_is_not_coerced(self):
        frame = pd.DataFrame({'agent_description': ['a (sfh)', 'b (sfh)']})

        described = describe_frame(frame)

        statistics = described['columns']['agent_description']
        assert statistics['kind'] == 'text'
        assert statistics['distinct'] == 2
        assert statistics['values'] == ['a (sfh)', 'b (sfh)']

    def test_a_categorical_of_digits_is_not_silently_summed(self):
        """Categoricals are the trap this module exists for; never treat one as a number."""
        series = pd.Series(pd.Categorical(['1', '2', '1']))

        assert describe_column(series)['kind'] == 'text'

    def test_a_timestamp_column_is_not_summed_as_nanoseconds(self):
        """`to_numeric` turns datetimes into ~1e18 ns, where a 1e-6 tolerance is ~27 minutes.

        No processor returns a timestamp *column* today -- timestamps are indexes here -- so this
        pins the classification before one does rather than after.
        """
        series = pd.Series(pd.to_datetime(['2021-03-23 23:00', '2021-03-24 00:00'], utc=True))

        assert describe_column(series)['kind'] == 'text'


class TestTheCategoricalOrderingTrap:
    """Two runs return the same rows in different orders; the reduction must not care."""

    def build(self, categories):
        """The same three rows, with the category order a different process would have seen."""
        index = pd.MultiIndex.from_arrays([
            pd.Categorical(['b', 'a', 'a'], categories=categories),
            pd.Categorical(['grid', 'retail', 'grid'],
                           categories=['retail', 'grid'] if categories[0] == 'b'
                           else ['grid', 'retail']),
        ])
        return pd.DataFrame({'price_in': [3.0, 1.0, 2.0]}, index=index)

    def test_the_index_summary_is_the_same_under_either_category_order(self):
        first = self.build(['a', 'b'])
        second = self.build(['b', 'a']).sort_index()

        assert describe_index(first.index) == describe_index(second.index), (
            'the index summary changed with the categorical encoding order, which differs '
            'between processes -- the reference would never match twice')

    def test_the_statistics_are_the_same_under_either_row_order(self):
        """Reordering whole rows -- index and values together -- must not move anything."""
        frame = self.build(['a', 'b'])

        assert describe_frame(frame)['columns'] == describe_frame(frame[::-1])['columns']


class TestAValueIndexMisalignmentIsVisible:
    """Sum, min, max and a distinct count are all invariant under a permutation of values.

    Without `ordered`, the entire family of "right numbers, wrong row" defects is unassertable --
    positional indexing written back onto a sorted frame, an off-by-one interval convention, a
    price series sorted into a duration curve. A review panel broke all six processors this way
    with every assertion green, which is what these tests exist to stop happening again.
    """

    def frame(self, values):
        index = pd.to_datetime(['2021-03-23 23:00', '2021-03-24 00:00', '2021-03-24 01:00'],
                               utc=True)
        return pd.DataFrame({'total': values}, index=index)

    def test_rolling_the_values_by_one_moves_the_ordered_total(self):
        """The time-shift defect: every profile plotted an hour out, with wraparound."""
        correct = describe_frame(self.frame([1.0, 2.0, 3.0]))['columns']['total']
        rolled = describe_frame(self.frame([2.0, 3.0, 1.0]))['columns']['total']

        assert correct['sum'] == rolled['sum'], 'the premise: the totals are identical'
        assert correct['min'] == rolled['min'] and correct['max'] == rolled['max']
        assert correct['ordered'] != rolled['ordered'], (
            'a whole-series roll left every recorded statistic unchanged, so a plot drawn an hour '
            'out of position would be pinned as correct')

    def test_sorting_the_values_against_the_index_moves_the_ordered_total(self):
        """The price-duration-curve defect: values sorted and written back positionally."""
        correct = describe_frame(self.frame([3.0, 1.0, 2.0]))['columns']['total']
        sorted_back = describe_frame(self.frame([1.0, 2.0, 3.0]))['columns']['total']

        assert correct['sum'] == sorted_back['sum']
        assert correct['ordered'] != sorted_back['ordered']

    def test_a_text_column_permuted_against_the_index_moves_its_digest(self):
        """The mislabelled-bus defect: descriptions rolled onto the wrong rows.

        `values` is a distinct *set*, so it is unchanged when every row is relabelled with another
        row's text -- a column can change on all but one row and keep an identical `distinct`.
        """
        index = pd.Index([0, 1, 2], name='bus')
        correct = describe_frame(pd.DataFrame(
            {'agent_description': ['no agents at bus', 'a (sfh)', 'b (sfh)']}, index=index))
        rolled = describe_frame(pd.DataFrame(
            {'agent_description': ['a (sfh)', 'b (sfh)', 'no agents at bus']}, index=index))

        assert (correct['columns']['agent_description']['values']
                == rolled['columns']['agent_description']['values']), 'the premise: same set'
        assert (correct['columns']['agent_description']['ordered_digest']
                != rolled['columns']['agent_description']['ordered_digest'])

    def test_the_ordered_total_survives_the_categorical_encoding_order(self):
        """It must catch misalignment without reintroducing the #229 flake it sits next to."""
        def build(categories):
            index = pd.CategoricalIndex(['b', 'a', 'c'], categories=categories)
            return pd.DataFrame({'price_in': [3.0, 1.0, 2.0]}, index=index)

        first = describe_frame(build(['a', 'b', 'c']))['columns']['price_in']
        second = describe_frame(build(['c', 'b', 'a']).sort_index())['columns']['price_in']

        assert first['ordered'] == second['ordered'], (
            'the ordered total changed with the categorical encoding order, which differs between '
            'processes -- it would flake rather than pin anything')

    def test_a_duplicated_index_label_records_no_ordered_total(self):
        """Ambiguous order is recorded as absent rather than as a value that cannot reproduce."""
        frame = pd.DataFrame({'v': [1.0, 2.0]}, index=pd.Index(['a', 'a']))

        assert describe_frame(frame)['columns']['v']['ordered'] is None


class TestADtypeChangeIsVisible:

    def test_a_numeric_column_becoming_strings_is_caught(self):
        """`to_numeric` accepts '60.448', so without dtype the change is invisible.

        matplotlib plots a string column on a categorical axis, so this is a real figure change.
        """
        numbers = describe_frame(pd.DataFrame({'power': [60.448, 62.605]}))['columns']['power']
        strings = describe_frame(pd.DataFrame({'power': ['60.448', '62.605']}))['columns']['power']

        assert numbers['kind'] == strings['kind'] == 'numeric', 'the premise: both read as numeric'
        assert numbers['sum'] == pytest.approx(strings['sum'])
        assert numbers['dtype'] != strings['dtype']

    def test_an_object_column_of_numbers_becoming_strings_is_caught(self):
        """The real shape: `power_description` is already `object`, so the kind cannot move.

        The topology processor initialises the column with a string and fills it with floats, so
        both the correct and the broken version have `dtype == 'O'` and identical stringified
        values. Only the types actually held tell them apart.
        """
        numbers = pd.Series(['no power at bus', 60.448], dtype=object)
        strings = numbers.astype(str)

        before, after = describe_column(numbers), describe_column(strings)

        assert before['dtype'] == after['dtype'] == 'O', 'the premise: the kind does not move'
        assert before['value_types'] == ['float', 'str']
        assert after['value_types'] == ['str']


class TestAbsolutePathsAreNormalised:
    """The market processors key their output by the directory they read it from."""

    def test_a_key_under_the_results_root_becomes_relative(self, tmp_path):
        market = tmp_path / 'markets' / 'electricity' / 'continuous'
        market.mkdir(parents=True)
        output = {str(market): pd.DataFrame({'price_in': [1.0]})}

        reduced = reduce_output(output, str(tmp_path))

        assert list(reduced) == ['/markets/electricity/continuous'], (
            'an un-normalised key records the tmp_path of the run that produced the reference, '
            'so the reference matches nothing on the next run')

    def test_a_key_that_is_not_a_path_is_left_alone(self, tmp_path):
        reduced = reduce_output({'electricity': pd.DataFrame({'x': [1.0]})}, str(tmp_path))

        assert list(reduced) == ['/electricity']


class TestThePandapowerNetIsWalkedAsANetwork:
    """`pandapowerNet` subclasses `dict`, so the isinstance order in `reduce_output` decides this."""

    def test_only_the_filled_element_tables_are_reduced(self, tmp_path):
        import pandapower as pp

        assert issubclass(pp.auxiliary.pandapowerNet, dict), (
            'the premise of this test has changed; the dict branch of reduce_output can no longer '
            'shadow the network branch')

        net = pp.create_empty_network()
        pp.create_bus(net, vn_kv=0.4)
        pp.create_bus(net, vn_kv=0.4)

        reduced = reduce_output({'run': net}, str(tmp_path))

        assert list(reduced) == ['/run/bus'], (
            'a pandapowerNet must be walked as its non-empty element tables. Reaching the dict '
            'branch instead reduces every private cache and empty res_* table, which is dozens of '
            'labels of noise and would bury the tables the processor actually filled in')
        assert reduced['/run/bus']['rows'] == 2


class TestALabelCollisionIsRefused:
    """Two outputs reducing to one label would silently compare fewer outputs than were produced."""

    def test_a_collision_raises_rather_than_overwriting(self, tmp_path):
        colliding = {'a': {'b': pd.DataFrame({'x': [1.0]})},
                     'a/b': pd.DataFrame({'x': [2.0]})}

        with pytest.raises(AssertionError, match='same label'):
            reduce_output(colliding, str(tmp_path))


class TestAnEmptyReturnIsNotEvidence:
    """The vacuity family: empty parametrize, empty SCENARIOS, empty allowlist, and this."""

    def full(self):
        """A fingerprint in which every processor produced one row and one number."""
        return {name: {'/x': {'rows': 1, 'index': None,
                              'columns': {'v': {'kind': 'numeric', 'sum': 1.0,
                                                'min': 1.0, 'max': 1.0, 'nulls': 0}}}}
                for name in processor_names()}

    def test_a_complete_fingerprint_has_no_complaints(self):
        assert emptiness_complaints(self.full()) == []

    def test_a_processor_returning_nothing_is_reported(self):
        fingerprint = self.full()
        fingerprint['GridDataProcessor.process_electricity_transformer_loading'] = {}

        complaints = emptiness_complaints(fingerprint)

        assert len(complaints) == 1
        assert 'process_electricity_transformer_loading' in complaints[0]
        assert 'nothing at all' in complaints[0]

    def test_a_processor_missing_entirely_is_reported(self):
        fingerprint = self.full()
        del fingerprint['MarketDataProcessor.process_agent_balancing']

        assert any('process_agent_balancing' in complaint
                   for complaint in emptiness_complaints(fingerprint))

    def test_a_processor_returning_only_empty_frames_is_reported(self):
        """`{'scenario': {}}` is not `{}`, and an assertion over its rows still passes vacuously."""
        fingerprint = self.full()
        fingerprint['AgentDataProcessor.process_all_meters_data'] = {
            '/scenario/electricity': {'rows': 0, 'index': None, 'columns': {}}}

        complaints = emptiness_complaints(fingerprint)

        assert len(complaints) == 1
        assert 'vacuity' in complaints[0]

    def test_rows_without_a_single_number_are_reported(self):
        """A frame of labels and no numbers cannot be compared as a result."""
        fingerprint = self.full()
        fingerprint['MarketDataProcessor.process_total_balancing'] = {
            '/x': {'rows': 4, 'index': None,
                   'columns': {'label': {'kind': 'text', 'distinct': 4}}}}

        complaints = emptiness_complaints(fingerprint)

        assert len(complaints) == 1
        assert 'not one numeric column' in complaints[0]

    def test_an_entirely_null_numeric_column_does_not_count_as_a_number(self):
        """The subtle one, and it is not hypothetical -- both references contain such a column.

        pandas totals an all-NaN column to `0.0`, so a guard reading `sum is not None` accepts a
        column holding no value at all. `bus_geodata.coords` is exactly that shape. The reference
        statistics are built by `describe_column` here rather than hand-written, because an
        earlier version of this test asserted against a `sum: None` shape the reducer never
        produces -- so it passed while the case it named went unguarded.
        """
        import numpy as np

        statistics = describe_column(pd.Series([np.nan, np.nan, np.nan, np.nan]))
        assert statistics['sum'] == 0.0 and statistics['min'] is None, (
            'the premise of this test has changed; describe_column no longer renders an all-NaN '
            'column as a zero sum with no minimum')

        fingerprint = self.full()
        fingerprint['MarketDataProcessor.process_average_pricing'] = {
            '/x': {'rows': 4, 'index': None, 'columns': {'average_price': statistics}}}

        complaints = emptiness_complaints(fingerprint)

        assert len(complaints) == 1
        assert 'entirely null' in complaints[0]


class TestNonFiniteValuesSurviveJson:
    """A reference that cannot be serialised is a test that cannot be regenerated."""

    def test_nan_and_infinity_are_recorded_as_null(self):
        import json
        import numpy as np

        frame = pd.DataFrame({'v': [np.nan, np.inf, -np.inf]})

        described = describe_frame(frame)

        json.dumps(described)  # would raise on a bare float('nan')
        assert described['columns']['v']['sum'] is None
