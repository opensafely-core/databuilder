import dataclasses
import datetime

import pytest

from databuilder import query_model
from databuilder.population_validation import (
    ValidationError,
    evaluate,
    validate_population_definition,
)
from databuilder.query_model import (
    AggregateByPatient,
    Case,
    Function,
    SelectColumn,
    SelectPatientTable,
    SelectTable,
    Series,
    TableSchema,
    Value,
)


def test_rejects_non_series():
    aint_no_series = "I am not a Series"
    with pytest.raises(ValidationError, match="must be a `query_model.Series`"):
        validate_population_definition(aint_no_series)


def test_rejects_invalid_dimension():
    event_series = SelectColumn(
        SelectTable("events", schema=TableSchema(value=bool)), "value"
    )
    with pytest.raises(ValidationError, match="one-row-per-patient series"):
        validate_population_definition(event_series)


def test_rejects_invalid_type():
    int_series = SelectColumn(
        SelectPatientTable("patients", schema=TableSchema(value=int)), "value"
    )
    with pytest.raises(ValidationError, match="boolean type"):
        validate_population_definition(int_series)


def test_accepts_basic_reasonable_population():
    has_registration = AggregateByPatient.Exists(SelectTable("registrations"))
    assert validate_population_definition(has_registration)


def test_rejects_basic_unreasonable_population():
    not_died = Function.Not(AggregateByPatient.Exists(SelectTable("ons_deaths")))
    with pytest.raises(ValidationError, match="must not evaluate as True for NULL"):
        validate_population_definition(not_died)


# TEST EVALUATE FUNCTION
#

patients = SelectPatientTable("patients", schema=TableSchema(value=int))
events = SelectTable("events", schema=TableSchema(value=int))
patients_value = SelectColumn(patients, "value")
events_value = SelectColumn(events, "value")

cases = [
    (
        True,
        # events.count_for_patient() + 3 >= 3
        Function.GE(
            Function.Add(AggregateByPatient.Count(events), Value(3)),
            Value(3),
        ),
    ),
    (
        False,
        # events.count_for_patient() + 3 >= 4
        Function.GE(
            Function.Add(AggregateByPatient.Count(events), Value(3)),
            Value(4),
        ),
    ),
    (
        True,
        # patients.v.is_null()
        Function.IsNull(patients_value),
    ),
    (
        False,
        # ~patients.v.is_null()
        Function.Not(Function.IsNull(patients_value)),
    ),
    (
        True,
        # events.v.max_for_patient().is_null()
        Function.IsNull(AggregateByPatient.Max(events_value)),
    ),
    (
        None,
        # events.v.max_for_patient() > 100
        Function.GT(
            AggregateByPatient.Max(events_value),
            Value(100),
        ),
    ),
    (
        True,
        # patients.v == 1 | (10 < 20)
        Function.Or(
            Function.EQ(patients_value, Value(1)),
            Function.LT(Value(2), Value(3)),
        ),
    ),
    (
        None,
        # patients.v == 100 | (patients.v <= 20)
        Function.Or(
            Function.EQ(patients_value, Value(100)),
            Function.LE(patients_value, Value(3)),
        ),
    ),
    (
        False,
        # ~patients.v.is_null | events.exists_for_patient()
        Function.Or(
            Function.Not(Function.IsNull(patients_value)),
            AggregateByPatient.Exists(events),
        ),
    ),
    (
        True,
        # (1 == 1) & (2 == 2)
        Function.And(
            Function.EQ(Value(1), Value(1)),
            Function.EQ(Value(2), Value(2)),
        ),
    ),
    (
        False,
        # ~(patients.v == 100 & events.exists_for_patient())
        Function.And(
            Function.EQ(patients_value, Value(100)),
            AggregateByPatient.Exists(events),
        ),
    ),
    (
        None,
        # patients.v == 1 & events.v.sum_for_patient() == 2
        Function.And(
            Function.EQ(patients_value, Value(1)),
            Function.EQ(AggregateByPatient.Sum(events_value), Value(2)),
        ),
    ),
    (
        True,
        Function.In(Value(10), Value({30, 20, 10})),
    ),
    (
        None,
        Function.In(
            patients_value,
            AggregateByPatient.CombineAsSet(events_value),
        ),
    ),
    (
        "bar",
        Case(
            {
                Function.EQ(patients_value, Value(1)): Value("foo"),
                Function.EQ(Value(1), Value(1)): Value("bar"),
            },
            default=None,
        ),
    ),
    (
        "baz",
        Case(
            {
                Function.EQ(patients_value, Value(1)): Value("foo"),
                Function.EQ(patients_value, Value(2)): Value("bar"),
            },
            default=Value("baz"),
        ),
    ),
    (
        None,
        Case(
            {
                Function.EQ(patients_value, Value(1)): Value("foo"),
                Function.EQ(patients_value, Value(2)): Value("bar"),
            },
            default=None,
        ),
    ),
    (
        datetime.date(2021, 6, 13),
        Function.DateAddDays(Value(datetime.date(2021, 5, 4)), Value(40)),
    ),
    (
        datetime.date(2021, 3, 25),
        Function.DateSubtractDays(Value(datetime.date(2021, 5, 4)), Value(40)),
    ),
    (
        datetime.date(2021, 5, 1),
        Function.ToFirstOfMonth(Value(datetime.date(2021, 5, 4))),
    ),
    (
        2022,
        Function.YearFromDate(Value(datetime.date(2022, 4, 29))),
    ),
    (
        4,
        Function.MonthFromDate(Value(datetime.date(2022, 4, 29))),
    ),
    (
        29,
        Function.DayFromDate(Value(datetime.date(2022, 4, 29))),
    ),
    (
        29,
        Function.DateDifferenceInYears(
            Value(datetime.date(1990, 1, 30)), Value(datetime.date(2020, 1, 15))
        ),
    ),
    (
        20,
        Function.DateDifferenceInYears(
            Value(datetime.date(2000, 1, 15)), Value(datetime.date(2020, 1, 15))
        ),
    ),
    (
        -1,
        Function.DateDifferenceInYears(
            Value(datetime.date(2020, 1, 20)), Value(datetime.date(2020, 1, 15))
        ),
    ),
    (
        -2,
        Function.DateDifferenceInYears(
            Value(datetime.date(2022, 1, 10)), Value(datetime.date(2020, 1, 15))
        ),
    ),
]


@pytest.mark.parametrize("expected,query", cases)
def test_evaluate(expected, query):
    result = evaluate(query)
    assert result == expected, f"Expected {expected}, got {result} in:\n{query}"


# TEST EVALUTE DEFINITION IS EXHAUSTIVE
#
# Test that the `evaluate()` single dispatch function is exhaustively defined so we
# can't forget to update it if we add a new query model operation.
#


def get_all_operations():
    "Return every operation defined in the query model"
    return [cls for cls in iterate_query_model_namespace() if is_operation(cls)]


def is_operation(cls):
    "Return whether an arbitrary value is a query model operation class"
    # We need to check this first or the `issubclass` check can fail
    if not isinstance(cls, type):
        return False
    # We need to check it's a proper subclass as the Node base class isn't itself a
    # dataclass so the `fields()` call will fail
    if not issubclass(cls, query_model.Node) or cls is query_model.Node:
        return False
    # If it takes arguments it's an operation, otherwise it's an abstract type
    return len(dataclasses.fields(cls)) > 0


def iterate_query_model_namespace():
    "Yield every public thing in the query_model module"
    yield from [getattr(query_model, name) for name in query_model.__all__]
    yield from vars(query_model.Function).values()
    yield from vars(query_model.AggregateByPatient).values()


@pytest.mark.parametrize(
    "operation", [op for op in get_all_operations() if issubclass(op, Series)]
)
def test_evaluate_function_defined_for(operation):
    default = evaluate.dispatch(object)
    function = evaluate.dispatch(operation)
    assert function != default, f"No `evaluate()` implementation for {operation}"
