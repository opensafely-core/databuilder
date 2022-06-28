import contextlib
from functools import cached_property

import sqlalchemy
import sqlalchemy.engine.interfaces
from sqlalchemy.sql import operators

from databuilder.backends.base import DefaultBackend
from databuilder.functools_utils import singledispatchmethod_with_cache
from databuilder.query_model import (
    AggregateByPatient,
    Case,
    Filter,
    Function,
    Position,
    SelectColumn,
    SelectPatientTable,
    SelectTable,
    Sort,
    Value,
    get_domain,
    has_many_rows_per_patient,
)
from databuilder.query_model_transforms import (
    PickOneRowPerPatientWithColumns,
    apply_transforms,
)
from databuilder.sqlalchemy_utils import get_setup_and_cleanup_queries, is_predicate

from .base import BaseQueryEngine


class BaseSQLQueryEngine(BaseQueryEngine):

    sqlalchemy_dialect: sqlalchemy.engine.interfaces.Dialect

    intermediate_table_prefix = "cte_"
    intermediate_table_count = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.backend:
            self.backend = DefaultBackend()

    def get_queries(self, variable_definitions):
        """
        Return the SQL queries to fetch the results for `variable_definitions`

        These are specified as a triple:

            list_of_setup_queries, query_to_fetch_results, list_of_cleanup_queries
        """
        variable_definitions = apply_transforms(variable_definitions)
        population_definition = variable_definitions.pop("population")
        variable_expressions = {
            name: self.get_expr(definition)
            for name, definition in variable_definitions.items()
        }
        population_expression = self.get_predicate(population_definition)
        query = self.select_patient_id_for_population(population_expression)
        query = query.add_columns(
            *[expr.label(name) for name, expr in variable_expressions.items()]
        )
        query = query.where(population_expression)
        query = apply_patient_joins(query)

        setup_queries, cleanup_queries = get_setup_and_cleanup_queries(query)
        return setup_queries, query, cleanup_queries

    def select_patient_id_for_population(self, population_expression):
        """
        Return a SELECT query which selects all the patient_ids that _might_ be included
        in the population (the WHERE clause later will filter this down to just those which
        should actually be included)

        This needs to cover, at a minimum, all patient_ids included in all tables
        referenced in the population expression. But including more won't affect the
        correctness of the result, so the only consideration here is performance.

        TODO: Give backends a mechanism for marking certain tables as containing all
        available patient_ids. We could then use such tables if available rather than
        messing aroud with UNIONS.
        """
        # Get all the tables needed to evaluate the population expression
        tables = sqlalchemy.select(population_expression).get_final_froms()
        if len(tables) > 1:
            # Select all patient IDs from all tables referenced in the expression
            id_selects = [
                sqlalchemy.select(table.c.patient_id.label("patient_id"))
                for table in tables
            ]
            # Create a table which contains the union of all these IDs. (Note UNION
            # rather than UNION ALL so we don't get duplicates.)
            population_table = self.reify_query(sqlalchemy.union(*id_selects))
            return sqlalchemy.select(population_table.c.patient_id)
        elif len(tables) == 1:
            # If there's only one table then use the IDs from that
            return sqlalchemy.select(tables[0].c.patient_id.label("patient_id"))
        else:
            # Gracefully handle the degenerate case where the population expression
            # doesn't reference any tables at all. Our validation rules ensure that such
            # an expression will never evaluate True so the population will always be
            # empty. But we can at least return an empty result, rather than blowing up.
            return sqlalchemy.select(sqlalchemy.literal(None).label("patient_id"))

    # Some databases care about the distinction between "predicates" (expressions which
    # are guaranteed boolean-typed by virtue of their syntax) and other forms of
    # expression. As these are semantically equivalent we can transform from one to the
    # other, we just need to know which one we're expecting in a context.
    def get_expr(self, node):
        sql = self.get_sql(node)
        return self.predicate_to_expression(sql) if is_predicate(sql) else sql

    def get_predicate(self, node):
        sql = self.get_sql(node)
        return self.expression_to_predicate(sql) if not is_predicate(sql) else sql

    def predicate_to_expression(self, sql):
        # Using the expression twice in the CASE statement is a bit inelegant, but I
        # can't immediately think of another way of doing this which preserves the
        # nullability of the expression (which `case((sql, True), else_=False)` doesnt'
        # do).
        return sqlalchemy.case((sql, True), (~sql, False))

    def expression_to_predicate(self, sql):
        return operators.eq(sql, True)

    # Without caching here we emit unnecessarily verbose and duplicative SQL because any
    # nodes which generate CTEs or temporary tables end up generating new ones each time
    # they're referenced — we want to generate them only once and then reuse them
    @singledispatchmethod_with_cache
    def get_sql(self, node):
        assert False, f"Unhandled node: {node}"

    @get_sql.register(Value)
    def get_sql_value(self, node):
        if isinstance(node.value, frozenset):
            value = tuple(self.convert_value(v) for v in node.value)
        else:
            value = self.convert_value(node.value)
        return sqlalchemy.literal(value)

    def convert_value(self, value):
        if hasattr(value, "_to_primitive_type"):
            return value._to_primitive_type()
        else:
            return value

    @get_sql.register(Function.EQ)
    def get_sql_eq(self, node):
        return operators.eq(self.get_expr(node.lhs), self.get_expr(node.rhs))

    @get_sql.register(Function.NE)
    def get_sql_ne(self, node):
        return operators.ne(self.get_expr(node.lhs), self.get_expr(node.rhs))

    @get_sql.register(Function.IsNull)
    def get_sql_is_null(self, node):
        return operators.is_(self.get_expr(node.source), None)

    @get_sql.register(Function.In)
    def get_sql_in(self, node):
        lhs = self.get_expr(node.lhs)
        rhs = self.get_expr(node.rhs)
        return lhs.in_(rhs)

    @get_sql.register(Function.Not)
    def get_sql_not(self, node):
        return sqlalchemy.not_(self.get_predicate(node.source))

    @get_sql.register(Function.And)
    def get_sql_and(self, node):
        return sqlalchemy.and_(
            self.get_predicate(node.lhs), self.get_predicate(node.rhs)
        )

    @get_sql.register(Function.Or)
    def get_sql_or(self, node):
        return sqlalchemy.or_(
            self.get_predicate(node.lhs), self.get_predicate(node.rhs)
        )

    @get_sql.register(Function.LT)
    def get_sql_lt(self, node):
        return operators.lt(self.get_expr(node.lhs), self.get_expr(node.rhs))

    @get_sql.register(Function.LE)
    def get_sql_le(self, node):
        return operators.le(self.get_expr(node.lhs), self.get_expr(node.rhs))

    @get_sql.register(Function.GT)
    def get_sql_gt(self, node):
        return operators.gt(self.get_expr(node.lhs), self.get_expr(node.rhs))

    @get_sql.register(Function.GE)
    def get_sql_ge(self, node):
        return operators.ge(self.get_expr(node.lhs), self.get_expr(node.rhs))

    @get_sql.register(Function.Negate)
    def get_sql_negate(self, node):
        return operators.neg(self.get_expr(node.source))

    @get_sql.register(Function.Add)
    def get_sql_add(self, node):
        return operators.add(self.get_expr(node.lhs), self.get_expr(node.rhs))

    @get_sql.register(Function.Subtract)
    def get_sql_subtract(self, node):
        return operators.sub(self.get_expr(node.lhs), self.get_expr(node.rhs))

    @get_sql.register(Function.YearFromDate)
    def get_sql_year_from_date(self, node):
        return self.get_date_part(self.get_expr(node.source), "YEAR")

    @get_sql.register(Function.MonthFromDate)
    def get_sql_month_from_date(self, node):
        return self.get_date_part(self.get_expr(node.source), "MONTH")

    @get_sql.register(Function.DayFromDate)
    def get_sql_day_from_date(self, node):
        return self.get_date_part(self.get_expr(node.source), "DAY")

    @get_sql.register(Function.DateDifferenceInYears)
    def get_sql_date_difference_in_years(self, node):
        return self.date_difference_in_years(
            self.get_expr(node.lhs), self.get_expr(node.rhs)
        )

    def date_difference_in_years(self, start, end):
        year_diff = self.get_date_part(end, "YEAR") - self.get_date_part(start, "YEAR")
        month_diff = self.get_date_part(end, "MONTH") - self.get_date_part(
            start, "MONTH"
        )
        day_diff = self.get_date_part(end, "DAY") - self.get_date_part(start, "DAY")
        # The CASE condition below detects the situation where the end day-of-year is
        # earlier than the start day-of-year. In such cases we need to subtract one from
        # the year delta to give the number of whole years that has elapsed between the
        # two dates.
        #
        # The most obvious way to determine this duplicates calls to the "get month"
        # function:
        #
        #     end_month < start_month OR (end_month == start_month AND end_day < start_day)
        #
        # or equivalently:
        #
        #     month_diff < 0 OR (month_diff == 0 AND day_diff < 0)
        #
        # However this expression has a simplified form that takes advantage of the fact
        # that `day_diff` has a maximum magnitude of `30`:
        #
        #     month_diff * -31 > day_diff
        #
        # To satisfy yourself that these expressions are equivalent, consider this truth
        # table:
        #
        #      month_diff | day_diff | result
        #     ============|==========|========
        #          +ve    |    any   | false
        #     ------------|----------|--------
        #                 |    +ve   | false
        #                 |----------|--------
        #           0     |     0    | false
        #                 |----------|--------
        #                 |    -ve   | true
        #     ------------|----------|--------
        #          -ve    |    any   | true
        #
        # By inspection, this truth table is correct for both expressions and therefore
        # they are equivalent.
        return year_diff - sqlalchemy.case((month_diff * -31 > day_diff, 1), else_=0)

    def get_date_part(self, date, part):
        raise NotImplementedError()

    @get_sql.register(Function.DateAddDays)
    def get_sql_date_add_days(self, node):
        return self.date_add_days(self.get_expr(node.lhs), self.get_expr(node.rhs))

    @get_sql.register(Function.DateSubtractDays)
    def get_sql_date_subtract_days(self, node):
        return self.date_subtract_days(self.get_expr(node.lhs), self.get_expr(node.rhs))

    def date_add_days(self, date, num_days):
        raise NotImplementedError()

    def date_subtract_days(self, date, num_days):
        return self.date_add_days(date, -num_days)

    @get_sql.register(Function.ToFirstOfMonth)
    def get_sql_to_first_of_month(self, node):
        return self.to_first_of_month(self.get_expr(node.source))

    def to_first_of_month(self, date):
        raise NotImplementedError()

    @get_sql.register(SelectColumn)
    def get_sql_select_column(self, node):
        table = self.get_table(node.source)
        return table.c[node.name]

    @get_sql.register(Case)
    def get_sql_case(self, node):
        cases = [
            (self.get_predicate(condition), self.get_expr(value))
            for (condition, value) in node.cases.items()
        ]
        if node.default is not None:
            default = self.get_expr(node.default)
        else:
            default = None
        return sqlalchemy.case(*cases, else_=default)

    @get_sql.register(AggregateByPatient.Sum)
    def get_sql_sum(self, node):
        return self.aggregate_series_by_patient(node.source, sqlalchemy.func.sum)

    @get_sql.register(AggregateByPatient.Min)
    def get_sql_min(self, node):
        return self.aggregate_series_by_patient(node.source, sqlalchemy.func.min)

    @get_sql.register(AggregateByPatient.Max)
    def get_sql_max(self, node):
        return self.aggregate_series_by_patient(node.source, sqlalchemy.func.max)

    @get_sql.register(AggregateByPatient.Exists)
    def get_sql_exists(self, node):
        return self.aggregate_frame_by_patient(
            node.source,
            aggregation_expr=sqlalchemy.literal(True),
            empty_value=False,
            single_row_value=True,
        )

    @get_sql.register(AggregateByPatient.Count)
    def get_sql_count(self, node):
        return self.aggregate_frame_by_patient(
            node.source,
            aggregation_expr=sqlalchemy.func.count("*"),
            empty_value=0,
            single_row_value=1,
        )

    def aggregate_series_by_patient(self, source_node, aggregation_func):
        query = self.get_select_query_for_node_domain(source_node)
        aggregation_expr = aggregation_func(self.get_expr(source_node))
        return self.apply_sql_aggregation(query, aggregation_expr)

    def aggregate_frame_by_patient(
        self, source_node, aggregation_expr, empty_value, single_row_value
    ):
        # Frame aggregations can operate on both one- and many-rows-per-patient frames
        # and these cases require different handling.
        if has_many_rows_per_patient(source_node):
            query = self.get_select_query_for_node_domain(source_node)
            value = self.apply_sql_aggregation(query, aggregation_expr)
            # We want to guarantee that our frame level aggregations are never
            # null-valued, even when combined with data involving patient_ids for which
            # they're not defined
            return sqlalchemy.func.coalesce(value, empty_value)
        else:
            # Given that we have a one-row-per-patient frame here, we can get a
            # reference to the table
            table = self.get_table(source_node)
            # Create an expression which is True if there's a matching row for this
            # patient and False otherwise (this differs from the SQL for event frame
            # aggregations because there's no need for a GROUP BY)
            has_row = table.c.patient_id.is_not(None)
            # If True and False are the values we're looking for then just return that
            # expression
            if single_row_value is True and empty_value is False:
                return has_row
            else:
                # Otherwise convert to the appropriate values
                return sqlalchemy.case((has_row, single_row_value), else_=empty_value)

    def apply_sql_aggregation(self, query, aggregation_expression):
        query = query.add_columns(aggregation_expression.label("value"))
        query = query.group_by(query.selected_columns[0])
        query = apply_patient_joins(query)
        aggregated_table = self.reify_query(query)
        return aggregated_table.c.value

    # The caching here is required for correctness: without it we can generate distinct
    # objects representing the same table and this confuses SQLAlchemy into generating
    # queries with ambiguous table references
    @singledispatchmethod_with_cache
    def get_table(self, node):
        assert False, f"Unhandled node: {node}"

    @get_table.register(SelectTable)
    @get_table.register(SelectPatientTable)
    def get_table_select_table(self, node):
        return self.backend.get_table_expression(node.name, node.schema)

    # We ignore Filter and Sort operations completely at this point in the code and just
    # pass the underlying table reference through. It's only later, when building the
    # SELECT query for a given Frame, that we make use of these. This is in order to
    # mirror the semantics of SQL whereby columns are selected directly from the
    # underlying table and filters and sorts are handled separately using WHERE/ORDER BY
    # clauses.
    @get_table.register(Sort)
    @get_table.register(Filter)
    def get_table_sort_and_filter(self, node):
        return self.get_table(node.source)

    @get_table.register(PickOneRowPerPatientWithColumns)
    def get_table_pick_one_row_per_patient(self, node):
        selected_columns = [self.get_expr(c) for c in node.selected_columns]
        order_clauses = [self.get_expr(c) for c in get_sort_conditions(node.source)]

        if node.position == Position.LAST:
            order_clauses = [c.desc() for c in order_clauses]

        query = self.get_select_query_for_node_domain(node.source)
        query = query.add_columns(*selected_columns)
        # Add an extra "row number" column to the query which gives the position of each
        # row within its patient_id partition as implied by the order clauses
        query = query.add_columns(
            sqlalchemy.func.row_number().over(
                partition_by=query.selected_columns[0], order_by=order_clauses
            )
        )

        query = apply_patient_joins(query)

        # Make the above into a subquery and pull out the relevant columns. Note, we're
        # deliberately using a subquery rather than `reify_query()` here as we want the
        # database to have the chance to spot that we're just fetching the first row
        # from each partition and optimise the query.
        subquery = query.alias()
        subquery_columns = list(subquery.columns)
        output_columns = subquery_columns[:-1]
        row_number = subquery_columns[-1]

        # Select the first row for each patient according to the above row numbering
        partitioned_query = sqlalchemy.select(output_columns).where(row_number == 1)

        return self.reify_query(partitioned_query)

    def reify_query(self, query):
        """
        By "reify" we just mean turning a SELECT query into something that can function
        as a table in other SQLAlchemy constructs. There are various ways to do this
        e.g. using `.alias()` to make a sub-query, using `.cte()` to make a Common Table
        Expression, or writing the results of the query to a temporary table.
        """
        return query.cte(name=self.next_intermediate_table_name())

    def next_intermediate_table_name(self):
        self.intermediate_table_count += 1
        return f"{self.intermediate_table_prefix}{self.intermediate_table_count}"

    def get_select_query_for_node_domain(self, node):
        """
        Given a many-rows-per-patient node, return the SELECT query corresponding to
        domain of that node
        """
        frame = get_domain(node).get_node()
        select_table, conditions = get_table_and_filter_conditions(frame)
        table = self.get_table(select_table)
        where_clauses = [self.get_predicate(condition) for condition in conditions]
        query = sqlalchemy.select(table.c.patient_id.label("patient_id"))
        if where_clauses:
            query = query.where(sqlalchemy.and_(*where_clauses))
        return query

    @contextlib.contextmanager
    def execute_query(self, variable_definitions):
        setup_queries, results_query, cleanup_queries = self.get_queries(
            variable_definitions
        )
        with self.engine.connect() as cursor:
            for setup_query in setup_queries:
                cursor.execute(setup_query)

            yield cursor.execute(results_query)

            assert not cleanup_queries, "Support these once tests exercise them"

    @cached_property
    def engine(self):
        engine_url = sqlalchemy.engine.make_url(self.dsn)
        # Hardcode the specific SQLAlchemy dialect we want to use: this is the
        # dialect the query engine will have been written for and tested with and we
        # don't want to allow global config changes to alter this
        engine_url._get_entrypoint = lambda: self.sqlalchemy_dialect
        engine = sqlalchemy.create_engine(engine_url, future=True)
        # The above relies on abusing SQLAlchemy internals so it's possible it will
        # break in future -- we want to know immediately if it does
        assert isinstance(engine.dialect, self.sqlalchemy_dialect)
        return engine


def apply_patient_joins(query):
    """
    Find any table references in `query` which aren't yet part of an explicit JOIN and
    LEFT OUTER JOIN them into the query using the first selected column as the join key

    A core feature of the Query Model/Engine is that we can arbitrarily include data
    from patient-level tables in a query because, in effect, there is always an implicit
    join on `patient_id`. This function makes those implicit joins explicit.
    """
    # We use the convention that the column to be joined on is always the first selected
    # column. This avoids having to hardcode, or pass around, the name of the column.
    join_key = query.selected_columns[0]
    join_key_name = join_key.key
    # The table referenced by `join_key`, and any tables already explicitly joined with
    # it, will be returned as the first value from the `get_final_froms()` method
    # (because `join_key` is the first column). Any remaining tables which aren't yet
    # explicitly joined on will be returned as additional clauses in the list. The best
    # explanation of SQLAlchemy's behaviour here is probably this:
    # https://docs.sqlalchemy.org/en/14/changelog/migration_14.html#change-4737
    implicit_joins = query.get_final_froms()[1:]
    for table in implicit_joins:
        query = query.join(table, table.c[join_key_name] == join_key, isouter=True)
    return query


def get_table_and_filter_conditions(frame):
    """
    Given a ManyRowsPerPatientFrame, return a base SelectTable operation and a list of
    filter conditions (or predicates) to be applied
    """
    root_frame, filters, _ = get_frame_operations(frame)
    return root_frame, [f.condition for f in filters]


def get_sort_conditions(frame):
    """
    Given a SortedFrame, return a tuple of Series which gives the sort order
    """
    _, _, sorts = get_frame_operations(frame)
    # Sort operations are given to us in order of application which is the reverse of
    # order of priority (i.e. the most recently applied sort gives us the primary sort
    # condition) so we reverse them here
    return tuple(s.sort_by for s in reversed(sorts))


def get_frame_operations(frame):
    """
    Given a ManyRowsPerPatientFrame, destructure it into a base SelectTable operation,
    plus separate lists of Filter and Sort operations
    """
    filters = []
    sorts = []
    while True:
        type_ = type(frame)
        if type_ is Filter:
            filters.insert(0, frame)
            frame = frame.source
        elif type_ is Sort:
            sorts.insert(0, frame)
            frame = frame.source
        elif type_ is SelectTable:
            return frame, filters, sorts
        else:
            assert False, f"Unexpected type: {frame}"
