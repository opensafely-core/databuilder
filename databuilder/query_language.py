import dataclasses
import datetime

from databuilder import query_model as qm
from databuilder.codes import BaseCode, Codelist
from databuilder.population_validation import validate_population_definition
from databuilder.query_model import get_series_type, has_one_row_per_patient

# This gets populated by the `__init_subclass__` methods of EventSeries and
# PatientSeries. Its structure is:
#
#   (<type>, <is_patient_level>): <SeriesClass>
#
# For example:
#
#   (bool, False): BoolEventSeries,
#   (bool, True): BoolPatientSeries,
#
REGISTERED_TYPES = {}


# Because ehrQL classes override `__eq__` we can't use them as dictionary keys. So where
# the query model expects dicts we represent them as lists of pairs, which the
# `_apply()` function can convert to dicts when it passes them to the query model.
class _DictArg(list):
    "Internal class for passing around dictionary arguments"


class Dataset:
    def set_population(self, population):
        validate_population_definition(population.qm_node)
        object.__setattr__(self, "population", population)

    def __setattr__(self, name, value):
        # TODO raise proper errors here
        assert name != "population"
        assert qm.has_one_row_per_patient(value.qm_node), value.qm_node
        super().__setattr__(name, value)


def compile(dataset):  # noqa A003
    return {k: v.qm_node for k, v in vars(dataset).items() if isinstance(v, Series)}


# BASIC SERIES TYPES
#


@dataclasses.dataclass(frozen=True)
class Series:
    qm_node: qm.Node

    def __hash__(self):
        # The issue here is not mutability but the fact that we overload `__eq__` for
        # syntatic sugar, which makes these types spectacularly ill-behaved as dict keys
        raise TypeError(f"unhashable type: {self.__class__.__name__!r}")

    # These are the basic operations that apply to any series regardless of type or
    # dimension
    def __eq__(self, other):
        return _apply(qm.Function.EQ, self, other)

    def __ne__(self, other):
        return _apply(qm.Function.NE, self, other)

    def is_null(self):
        return _apply(qm.Function.IsNull, self)

    def is_in(self, other):
        # The query model requires an immutable Set type for containment queries, but
        # that's a bit user-unfriendly so we accept other types here and convert them
        if isinstance(other, (tuple, list, set)):
            other = frozenset(other)
        return _apply(qm.Function.In, self, other)

    def map_values(self, mapping):
        """
        Accepts a dictionary mapping one set of values to another and applies that
        mapping to the series
        """
        cases = _DictArg(
            (self == from_value, to_value) for from_value, to_value in mapping.items()
        )
        return _apply(qm.Case, cases)


class EventSeries(Series):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register the series using its `_type` attribute
        REGISTERED_TYPES[cls._type, False] = cls

    # If we end up with any type-agnostic aggregations (count non-null, maybe?) then
    # they would be defined here as well


class PatientSeries(Series):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register the series using its `_type` attribute
        REGISTERED_TYPES[cls._type, True] = cls


# BOOLEAN SERIES
#


class BoolFunctions:
    def __invert__(self):
        return _apply(qm.Function.Not, self)

    def __and__(self, other):
        return _apply(qm.Function.And, self, other)

    def __or__(self, other):
        return _apply(qm.Function.Or, self, other)


class BoolEventSeries(BoolFunctions, EventSeries):
    _type = bool


class BoolPatientSeries(BoolFunctions, PatientSeries):
    _type = bool


# STRING SERIES
#


class ComparableFunctions:
    def __lt__(self, other):
        return _apply(qm.Function.LT, self, other)

    def __le__(self, other):
        return _apply(qm.Function.LE, self, other)

    def __ge__(self, other):
        return _apply(qm.Function.GE, self, other)

    def __gt__(self, other):
        return _apply(qm.Function.GT, self, other)


class ComparableAggregations:
    def minimum_for_patient(self):
        return _apply(qm.AggregateByPatient.Min, self)

    def maximum_for_patient(self):
        return _apply(qm.AggregateByPatient.Max, self)


class StrEventSeries(ComparableFunctions, ComparableAggregations, EventSeries):
    _type = str


class StrPatientSeries(ComparableFunctions, PatientSeries):
    _type = str


# NUMERIC SERIES
#


class NumericFunctions(ComparableFunctions):
    def __neg__(self):
        return _apply(qm.Function.Negate, self)

    def __add__(self, other):
        return _apply(qm.Function.Add, self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return _apply(qm.Function.Subtract, self, other)

    def __rsub__(self, other):
        return other + -self


class NumericAggregations(ComparableAggregations):
    def sum_for_patient(self):
        return _apply(qm.AggregateByPatient.Sum, self)


class IntEventSeries(NumericFunctions, NumericAggregations, EventSeries):
    _type = int


class IntPatientSeries(NumericFunctions, PatientSeries):
    _type = int


class FloatEventSeries(NumericFunctions, NumericAggregations, EventSeries):
    _type = float


class FloatPatientSeries(NumericFunctions, PatientSeries):
    _type = float


# DATE SERIES
#


def parse_date_if_str(value):
    if isinstance(value, str):
        return datetime.date.fromisoformat(value)
    else:
        return value


class DateFunctions(ComparableFunctions):
    @property
    def year(self):
        return _apply(qm.Function.YearFromDate, self)

    @property
    def month(self):
        return _apply(qm.Function.MonthFromDate, self)

    @property
    def day(self):
        return _apply(qm.Function.DayFromDate, self)

    def difference_in_years(self, other):
        other = parse_date_if_str(other)
        return _apply(qm.Function.DateDifferenceInYears, self, other)

    def add_days(self, other):
        return _apply(qm.Function.DateAddDays, self, other)

    def subtract_days(self, other):
        return _apply(qm.Function.DateSubtractDays, self, other)

    def is_before(self, other):
        other = parse_date_if_str(other)
        return self < other

    def is_on_or_before(self, other):
        other = parse_date_if_str(other)
        return self <= other

    def is_after(self, other):
        other = parse_date_if_str(other)
        return self > other

    def is_on_or_after(self, other):
        other = parse_date_if_str(other)
        return self >= other

    def to_first_of_month(self):
        return _apply(qm.Function.ToFirstOfMonth, self)


class DateAggregations(ComparableAggregations):
    "Empty for now"


class DateEventSeries(DateFunctions, DateAggregations, EventSeries):
    _type = datetime.date


class DatePatientSeries(DateFunctions, PatientSeries):
    _type = datetime.date


# CODE SERIES
#


class CodeFunctions:
    def to_category(self, categorisation):
        return self.map_values(categorisation)


class CodeEventSeries(CodeFunctions, EventSeries):
    _type = BaseCode


class CodePatientSeries(CodeFunctions, PatientSeries):
    _type = BaseCode


# CONVERT QUERY MODEL SERIES TO EHRQL SERIES
#


def _wrap(qm_node):
    """
    Wrap a query model series in the ehrQL series class appropriate for its type and
    dimension
    """
    type_ = get_series_type(qm_node)
    is_patient_level = has_one_row_per_patient(qm_node)
    try:
        cls = REGISTERED_TYPES[type_, is_patient_level]
    except KeyError:
        # If we don't have a match for exactly this type then we should have one for a
        # superclass
        matches = [
            cls
            for ((target_type, target_dimension), cls) in REGISTERED_TYPES.items()
            if issubclass(type_, target_type) and is_patient_level == target_dimension
        ]
        assert len(matches) == 1
        cls = matches[0]
    return cls(qm_node)


def _apply(qm_cls, *args):
    """
    Applies a query model operation `qm_cls` to its arguments which can be either ehrQL
    series or static values, returns an ehrQL series
    """
    # Convert all arguments into query model nodes
    qm_args = map(_convert, args)
    qm_node = qm_cls(*qm_args)
    # Wrap the resulting node back up in an ehrQL series
    return _wrap(qm_node)


def _convert(arg):
    # Unpack dictionary arguments
    if isinstance(arg, _DictArg):
        return {_convert(key): _convert(value) for key, value in arg}
    # If it's an ehrQL series then get the wrapped query model node
    elif isinstance(arg, Series):
        return arg.qm_node
    # If it's a Codelist extract the set of codes and put it in a Value wrapper
    elif isinstance(arg, Codelist):
        return qm.Value(frozenset(arg.codes))
    # Otherwise it's a static value and needs to be put in a query model Value wrapper
    else:
        return qm.Value(arg)


# FRAME TYPES
#


class Frame:
    def __init__(self, qm_node):
        self.qm_node = qm_node

    def __getattr__(self, name):
        return _wrap(qm.SelectColumn(source=self.qm_node, name=name))

    def exists_for_patient(self):
        return _wrap(qm.AggregateByPatient.Exists(source=self.qm_node))

    def count_for_patient(self):
        return _wrap(qm.AggregateByPatient.Count(source=self.qm_node))


class PatientFrame(Frame):
    pass


class EventFrame(Frame):
    def take(self, series):
        return EventFrame(
            qm.Filter(
                source=self.qm_node,
                condition=series.qm_node,
            )
        )

    def drop(self, series):
        return EventFrame(
            qm.Filter(
                source=self.qm_node,
                condition=qm.Function.Or(
                    lhs=qm.Function.Not(series.qm_node),
                    rhs=qm.Function.IsNull(series.qm_node),
                ),
            )
        )

    def sort_by(self, *order_series):
        qm_node = self.qm_node
        # We expect series to be supplied highest priority first and, as the most
        # recently applied Sort operation has the highest priority, we need to apply
        # them in reverse order
        for series in reversed(order_series):
            qm_node = qm.Sort(
                source=qm_node,
                sort_by=series.qm_node,
            )
        return SortedEventFrame(qm_node)


class SortedEventFrame(Frame):
    def first_for_patient(self):
        return PatientFrame(
            qm.PickOneRowPerPatient(
                position=qm.Position.FIRST,
                source=self.qm_node,
            )
        )

    def last_for_patient(self):
        return PatientFrame(
            qm.PickOneRowPerPatient(
                position=qm.Position.LAST,
                source=self.qm_node,
            )
        )


# FRAME CONSTRUCTOR ENTRYPOINTS
#


def build_patient_table(name, schema, contract=None):
    if contract is not None:
        contract.validate_schema(schema)
    return PatientFrame(
        qm.SelectPatientTable(name, schema=qm.TableSchema(schema)),
    )


def build_event_table(name, schema, contract=None):
    if contract is not None:  # pragma: no cover
        contract.validate_schema(schema)
    return EventFrame(
        qm.SelectTable(name, schema=qm.TableSchema(schema)),
    )


# CASE EXPRESSION FUNCTIONS
#


# TODO: There's no explicit error handling on using this wrong e.g. not calling `then()`
# or passing the wrong sort of thing as `condition`. The query model will prevent any
# invalid queries being created, but we should invest time in making the errors as
# immediate and as friendly as possible.
class when:
    def __init__(self, condition):
        self._condition = condition

    def then(self, value):
        new = self.__class__(self._condition)
        new._value = value
        return new


def case(*when_thens, default=None):
    cases = _DictArg((case._condition, case._value) for case in when_thens)
    # If we don't want a default then we shouldn't supply an argument, or else it will
    # get converted into `Value(None)` which is not what we want
    if default is None:
        return _apply(qm.Case, cases)
    else:
        return _apply(qm.Case, cases, default)
