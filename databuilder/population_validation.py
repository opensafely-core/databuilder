import dataclasses
import datetime
import operator
from functools import singledispatch

from databuilder.query_model import (
    AggregateByPatient,
    Case,
    Function,
    SelectColumn,
    Series,
    ValidationError,
    Value,
    get_series_type,
    has_one_row_per_patient,
)


def validate_population_definition(population):
    """
    Test that a given Series is suitable for use as a population definition
    """
    if not isinstance(population, Series):
        raise ValidationError(
            "population definition must be a `query_model.Series` instance"
        )
    if not has_one_row_per_patient(population) or get_series_type(population) != bool:
        raise ValidationError(
            "population definition must be a one-row-per-patient series of boolean type"
        )
    # We exclude population definitions which evaluate as True when all their inputs are
    # NULL. Here's why.
    #
    # A population definition is a boolean series which is True for all patients who
    # should be included in the population. In SQL terms, you can think of a boolean
    # series as a function, or expression, which takes a row of patient values
    # (potentially drawn from many different tables) and returns True, False or NULL.
    #
    # It is possible to construct definitions which take the value True when all their
    # inputs are NULL. The most trivial example is:
    #
    #   Value(True)
    #
    # i.e. always, unconditionally True for everyone. This might seem silly, but we used
    # to think it was a convenient way to say "just give me all the patients" in test
    # cases.
    #
    # A slightly less trivial example is:
    #
    #   Function.Not(AggregateByPatient.Exists(SelectTable("ons_deaths")))
    #
    # i.e. give me all patients for whom we do not have a death record from ONS. Here
    # the `Exists` aggregation returns False when there is no corresponding row for the
    # patient in the `ons_deaths` table, and the `Not` function transforms that value
    # into True.
    #
    # In both these cases, the expression can be True when evaluated on patients which
    # do not exist in any of the tables referenced by the expression itself. This means
    # that the population we end up with is sensitive to the "universe" of patients we
    # decide to feed in. And some natural-sounding universes, e.g. the union of all
    # patients featuring anywhere in the database, are awkward and expensive to compute.
    #
    # Excluding such definitions means that the minimum set of patients needed to
    # evalute the expression (all patients featuring in tables referenced in the
    # expression) gives the same results as with any larger set. So the question of
    # exactly how we define the patient "universe" goes away.
    #
    # As well as simplifying the implementation there are also good scientific reasons
    # to exclude these kinds of population definitions: a well constructed study should
    # start with a positively defined population (e.g. all patients registered with a
    # practice as of a certain date) from which some patients are then excluded. A
    # population definition like "all patients who have not died" doesn't unambiguously
    # define a universe of patients and so should not be allowed even if we could
    # reliably interpret it.
    #
    # To determine whether a population definition meets this criterion we walk the tree
    # of query model nodes and for each operation which would normally involve fetching
    # data we substitute the result of that operation when given no data (this is often,
    # but not always, NULL). The end result of the process will be either True, False or
    # NULL. It is definitions which return True here that we reject.
    if evaluate(population) is True:
        # TODO: Wording could do with more thought here
        raise ValidationError(
            "population definition must not evaluate as True for NULL inputs"
        )
    return True


@singledispatch
def evaluate(node):
    """
    Given a Series node, return the value it would take if all its inputs were NULL
    """
    assert False, f"Unhandled node type: {type(node)}"


@evaluate.register(Value)
def evaluate_value(node):
    return node.value


@evaluate.register(SelectColumn)
def evaluate_select_column(node):
    return None


@evaluate.register(AggregateByPatient.Exists)
def evaluate_exists(node):
    return False


@evaluate.register(AggregateByPatient.Count)
def evaluate_count(node):
    return 0


@evaluate.register(AggregateByPatient.CombineAsSet)
def evaluate_combine_as_set(node):
    return ()


@evaluate.register(AggregateByPatient.Min)
@evaluate.register(AggregateByPatient.Max)
@evaluate.register(AggregateByPatient.Sum)
def evaluate_aggregation(node):
    return None


@evaluate.register(Function.IsNull)
def evaluate_is_null(node):
    return evaluate(node.source) is None


@evaluate.register(Function.And)
def evaluate_and(node):
    lhs = evaluate(node.lhs)
    rhs = evaluate(node.rhs)
    if lhs is True and rhs is True:
        return True
    elif lhs is False or rhs is False:
        return False
    else:
        return None


@evaluate.register(Function.Or)
def evaluate_or(node):
    lhs = evaluate(node.lhs)
    rhs = evaluate(node.rhs)
    if lhs is True or rhs is True:
        return True
    elif lhs is False and rhs is False:
        return False
    else:
        return None


@evaluate.register(Case)
def evaluate_case(node):
    for condition, value in node.cases.items():
        if evaluate(condition) is True:
            return evaluate(value)
    if node.default is not None:
        return evaluate(node.default)


def register_op(cls):
    """
    Handle registration for any operations with the default NULL behaviour, which is
    that they return NULL if any of their arguments are NULL
    """
    getters = [operator.attrgetter(field.name) for field in dataclasses.fields(cls)]

    def register(function):
        @evaluate.register(cls)
        def apply(node):
            args = [evaluate(get_arg(node)) for get_arg in getters]
            if any(arg is None for arg in args):
                return None
            return function(*args)

        return apply

    return register


@register_op(Function.YearFromDate)
def year_from_date(date):
    return date.year


@register_op(Function.MonthFromDate)
def month_from_date(date):
    return date.month


@register_op(Function.DayFromDate)
def day_from_date(date):
    return date.day


@register_op(Function.DateDifferenceInYears)
def date_difference_in_years(start, end):
    year_diff = end.year - start.year
    if (end.month, end.day) < (start.month, start.day):
        return year_diff - 1
    else:
        return year_diff


@register_op(Function.In)
def in_(lhs, rhs):
    return operator.contains(rhs, lhs)


@register_op(Function.DateAddDays)
def date_add_days(date, num_days):
    return date + datetime.timedelta(days=num_days)


@register_op(Function.DateSubtractDays)
def date_subtract_days(date, num_days):
    return date - datetime.timedelta(days=num_days)


@register_op(Function.ToFirstOfMonth)
def to_first_of_month(date):
    return date.replace(day=1)


register_op(Function.Not)(operator.not_)
register_op(Function.Negate)(operator.neg)
register_op(Function.EQ)(operator.eq)
register_op(Function.NE)(operator.ne)
register_op(Function.LT)(operator.lt)
register_op(Function.LE)(operator.le)
register_op(Function.GT)(operator.gt)
register_op(Function.GE)(operator.ge)
register_op(Function.Add)(operator.add)
register_op(Function.Subtract)(operator.sub)
