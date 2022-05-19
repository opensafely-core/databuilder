"""
This module provides most of the logic for transforming an abstract representation of a
data query (what we call the Query Model) into SQL and executing it against a database.
Most, but not all: in order to run against a real database it's necessary to subclass
BaseSQLQueryEngine and provide some database-specific attributes and methods.

The primary entry point is `BaseSQLQueryEngine.get_queries()`.

The job of this code is to transform one graph structure (the Query Model, consisting of
QueryNode instances) into another (a SQLAlchemy query, consisting of ClauseElement
instances). For each type of QueryNode there is a method which knows how to transform
nodes of that type into SQLAlchemy ClauseElements. As nodes may reference other nodes,
transforming node A to SQL may require transforming nodes B and C to SQL first. So the
implementation relies heavily on recursion to do its job.

The methods for handling specific QueryNode types are brought together using the
"singledispatch" decorator into a single method which takes an arbitrary QueryNode and
returns the appropriate SQL for it. Calling this method on the root nodes in the Query
Model will force the code to recurse its way over the entire Query Model and generate
the full SQL required.

Before converting the Query Model graph we have the opportunity to modify it in order to
make it easier to work with, or to generate more efficient SQL. This is handled by the
`apply_optimisations` function.

Also worth noting is the TemporaryTable class. This subclasses the standard SQLAlchemy
Table to add a pair of extra attributes: a list of setup queries needed to create and
populate the table, and a list of cleanup queries needed to get rid of it. This means we
don't need a separate out-of-band mechanism to track and manage temporary tables. The
function `get_setup_and_cleanup_queries` takes care of finding all the TemporaryTables
embedded in a query and returning their setup and cleanup queries in the appropriate
order.
"""

import contextlib
import copy
import dataclasses
import datetime
import os
import typing
from collections import defaultdict
from functools import cached_property
from typing import Any, Optional, Union

import sqlalchemy
import sqlalchemy.dialects.mssql
import sqlalchemy.schema
from sqlalchemy import type_coerce
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.sql import ClauseElement

from .. import query_model, sqlalchemy_types
from ..functools_utils import singledispatchmethod_with_unions
from ..sqlalchemy_utils import (
    TemporaryTable,
    get_primary_table,
    get_referenced_tables,
    get_setup_and_cleanup_queries,
    group_and_aggregate,
    include_joined_tables,
    select_first_row_per_partition,
)
from .base import BaseQueryEngine
from .query_model_convert_to_old import convert as convert_to_old
from .query_model_old import (
    Codelist,
    Column,
    Comparator,
    DateAddition,
    DateDeltaAddition,
    DateDeltaSubtraction,
    DateDifference,
    DateSubtraction,
    FilteredTable,
    QueryNode,
    RoundToFirstOfMonth,
    RoundToFirstOfYear,
    Row,
    RowFromAggregate,
    Table,
    Value,
    ValueFromAggregate,
    ValueFromCategory,
    ValueFromFunction,
    ValueFromRow,
    YearFromDate,
)

# These are nodes which select a single column from a query (regardless of whether that
# results in a single value per patient or in multiple values per patient)
ColumnSelectorNode = Union[ValueFromRow, ValueFromAggregate, Column]

# These are the basic types we accept as arguments in the Query Model
Scalar = Union[None, bool, int, float, str, datetime.datetime, datetime.date]
StaticValue = Union[Scalar, tuple[Scalar], list[Scalar]]

SCALAR_TYPES = typing.get_args(Scalar)


# This is an internal class that is injected into the DAG by the QueryEngine, but that
# does not form part of the public Query Model
@dataclasses.dataclass(frozen=True)
class ReifiedQuery(QueryNode):
    source: QueryNode
    columns: tuple[str]

    def _get_referenced_nodes(self):
        return (self.source,)  # pragma: no cover


class MissingString(str):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        raise NotImplementedError(self.message)


class BaseSQLQueryEngine(BaseQueryEngine):

    sqlalchemy_dialect: type[Dialect]

    # No limit by default although some DBMSs may impose one
    max_rows_per_insert: Optional[int] = None

    # Per-instance cache for SQLAlchemy Engine
    _engine: Optional[sqlalchemy.engine.Engine] = None

    # Force subclasses to define this
    temp_table_prefix: str = MissingString("'temp_table_prefix' is undefined")

    # Use a simple counter to generate unique (per session) temporary table names
    temp_table_count: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # See docstring on `get_sql_element` for details on this
        self.sql_element_cache: dict[QueryNode, ClauseElement] = {}

    def get_queries(self, column_definitions):
        """
        Build the list of SQL queries to execute

        This is returned as a triple:

            list_of_setup_queries, query_to_fetch_results, list_of_cleanup_queries
        """
        # Check that we are being passed the new-style query model and convert it to
        # the old, which we use internally for now.
        assert isinstance(list(column_definitions.values())[0], query_model.Node)
        column_definitions = convert_to_old(column_definitions)

        # Modify the Query Model graph to make it easier to work with, or to generate
        # more efficient SQL
        column_definitions = apply_optimisations(column_definitions)

        # Convert each column definition to SQL
        column_queries = {
            column: self.get_sql_element(definition)
            for column, definition in column_definitions.items()
        }

        # `population` is a special-cased boolean column, it doesn't appear
        # itself in the output but it determines what rows are included
        population_query = column_queries.pop("population")

        # TODO: Not sure why we require just a single table here for the population. I
        # think this could be lifted as long as we did a FULL OUTER JOIN between all the
        # tables.
        tables = get_referenced_tables(population_query)
        assert len(tables) == 1
        population_table = tables[0]

        # Start the query by selecting the "patient_id" column from all rows where the
        # "population" condition evaluates true
        results_query = (
            sqlalchemy.select([population_table.c.patient_id.label("patient_id")])
            .select_from(population_table)
            .where(population_query)
        )

        # For each column to be included in the output ...
        for column_name, column_query in column_queries.items():
            # Ensure the results_query JOINs on all the tables it needs to be able to
            # include this column in the output
            results_query = include_joined_tables(
                results_query,
                get_referenced_tables(column_query),
                join_column="patient_id",
            )
            # Add this column to the final selected results using the supplied name
            results_query = results_query.add_columns(column_query.label(column_name))

        # Get all the setup queries needed to populate the various temporary tables used
        # by the results_query, and the queries needed to clean them up afterwards
        setup_queries, cleanup_queries = get_setup_and_cleanup_queries(results_query)

        return setup_queries, results_query, cleanup_queries

    @contextlib.contextmanager
    def execute_query(self, column_definitions):
        setup_queries, results_query, cleanup_queries = self.get_queries(
            column_definitions
        )
        with self.engine.connect() as cursor:
            for query in setup_queries:
                cursor.execute(query)

            yield cursor.execute(results_query)

            for query in cleanup_queries:
                cursor.execute(query)

    @singledispatchmethod_with_unions
    def get_sql_element_or_value(self, value: Any):
        """
        Certain places in our Query Model support values which can either be QueryNodes
        themselves or plain static values (booleans, integers, dates, list of dates
        etc). Note that the type signature really ought to be:

            Union[QueryNode, StaticValue] -> Union[ClauseElement, StaticValue]

        But the singledispatch decorator can't handle the StaticValue type properly (see
        `get_static_value` below).
        """
        # Fallback for unhandled types
        assert False, f"Unhandled type {value!r}"

    @get_sql_element_or_value.register
    def get_static_value(self, value: Union[Scalar, tuple, list]):
        # This is a fudge: the type of `value` above really ought to be StaticValue but
        # the singledispatch decorator can't handle the parameterized tuple and list
        # types. So instead we accept all tuples and lists and enforce the types of
        # their elements at runtime. See: https://bugs.python.org/issue46191
        if isinstance(value, (tuple, list)) and any(
            not isinstance(v, SCALAR_TYPES) for v in value
        ):  # pragma: no cover
            assert False, f"Unhandled type {value!r}"
        return value

    @get_sql_element_or_value.register
    def get_sql_element(self, node: QueryNode):
        """
        Caching wrapper around `get_sql_element_no_cache()` below, which is the
        entrypoint method for converting QueryNodes to SQL ClauseElements

        The caching is a peformance enhancement, not for the Python code, but rather for
        the SQL we generate. It ensures that we don't get needlessly complex (though
        correct) SQL by generating duplicated temporary tables or table-like clauses.
        """
        # Note: we can't just use the `functools.cache` decorator here because it
        # doesn't play nicely with the `singledispatchmethod` decorator
        if node in self.sql_element_cache:
            return self.sql_element_cache[node]
        else:
            element = self.get_sql_element_no_cache(node)
            self.sql_element_cache[node] = element
            return element

    @singledispatchmethod_with_unions
    def get_sql_element_no_cache(self, node: QueryNode):
        """
        Given a QueryNode object from the Query Model return the SQLAlchemy
        ClauseElement which represents it.

        This uses single dispatch to handle each type of QueryNode via the appropriate
        method below.
        """
        # Fallback for unhandled types
        assert False, f"Unhandled query node type: {node!r}"

    @get_sql_element_no_cache.register
    def get_element_from_table(self, node: Table):
        table = self.backend.get_table_expression(node.name)
        return table.select()

    @get_sql_element_no_cache.register
    def get_element_from_filtered_table(self, node: FilteredTable):
        query = self.get_sql_element(node.source)
        filter_value = self.get_sql_element_or_value(node.value)
        return apply_filter(
            query,
            column=node.column,
            operator=node.operator,
            value=filter_value,
            # Ideally we wouldn't be passing these through, but we need to until we can
            # rewrite the `apply_filter` function
            value_query_node=node.value,
            or_null=node.or_null,
        )

    @get_sql_element_no_cache.register
    def get_element_from_row(self, node: Row):
        query = self.get_sql_element(node.source)
        return select_first_row_per_partition(
            query,
            partition_column="patient_id",
            sort_columns=node.sort_columns,
            descending=node.descending,
        )

    @get_sql_element_no_cache.register
    def get_element_from_row_from_aggregate(self, node: RowFromAggregate):
        query = self.get_sql_element(node.source)
        return group_and_aggregate(
            query,
            group_by_column="patient_id",
            input_column=node.input_column,
            function_name=node.function,
            output_column=node.output_column,
        )

    @get_sql_element_no_cache.register
    def get_element_from_reified_query(self, node: ReifiedQuery):
        """
        Take a query and return something table-like which contains the results of this
        query.

        At present, we do this via creating a temporary table and writing the results of
        the query to that table. But this is an implementation detail, chosen for its
        performance characteristics. It's possible to replace the below with either of:

            return query.cte()
            return query.subquery()

        and the tests will still pass (other than those which assert specific things
        about the generated SQL)
        """
        query = self.get_sql_element(node.source)
        # Select just the specified columns. This is a performance optimisation to avoid
        # reifying more data than we need.
        column_names = {"patient_id"} | set(node.columns)
        columns = [query.selected_columns[name] for name in column_names]
        query = query.with_only_columns(columns)

        table_columns = [sqlalchemy.Column(c.name, c.type) for c in columns]
        table_name = self.get_temp_table_name("group_table")
        table = TemporaryTable(table_name, sqlalchemy.MetaData(), *table_columns)

        create_query = self.query_to_create_temp_table_from_select_query(table, query)
        cleanup_queries = (
            [sqlalchemy.schema.DropTable(table, if_exists=True)]
            if self.temp_table_needs_dropping(create_query)
            else []
        )
        table.setup_queries = [create_query]
        table.cleanup_queries = cleanup_queries
        return table

    @get_sql_element_no_cache.register
    def get_element_from_column_selector(self, node: ColumnSelectorNode):
        table = self.get_sql_element(node.source)
        return table.c[node.column]

    @get_sql_element_no_cache.register
    def get_element_from_category_node(self, node: ValueFromCategory):
        # TODO: I think that both `label` and `default` should get passed through
        # `get_sql_element_or_value` as there's no reason in principle these couldn't be
        # dynamic values
        return sqlalchemy.case(
            [
                (self.get_sql_element(condition), label)
                for label, condition in node.definitions.items()
            ],
            else_=node.default,
        )

    @get_sql_element_no_cache.register
    def get_element_from_comparator_node(self, comparator: Comparator):
        # TODO: I think we can simplify things here, in particular the distinction
        # between `operator` and `connector` and the handling of negatation. But that
        # involves changing the Query Model which is a task for another day
        operator = comparator.operator
        if comparator.connector is not None:
            assert operator is None
            if comparator.connector == "and_":
                operator = "__and__"
            elif comparator.connector == "or_":
                operator = "__or__"
            else:
                assert False

        lhs = self.get_sql_element_or_value(comparator.lhs)
        rhs = self.get_sql_element_or_value(comparator.rhs)

        condition_expression = getattr(lhs, operator)(rhs)
        if comparator.negated:
            condition_expression = sqlalchemy.not_(condition_expression)

        return condition_expression

    @get_sql_element_no_cache.register
    def get_element_from_codelist(self, codelist: Codelist):
        """
        Given a codelist, build a SQLAlchemy representation of the temporary table
        needed to store that codelist and then generate the queries necessary to create
        and populate that table
        """
        codes = codelist.codes
        max_code_len = max(map(len, codes))
        table_name = self.get_temp_table_name("codelist")
        table = TemporaryTable(
            table_name,
            sqlalchemy.MetaData(),
            sqlalchemy.Column(
                "code",
                sqlalchemy.types.String(max_code_len),
                nullable=False,
            ),
            sqlalchemy.Column(
                "system",
                sqlalchemy.types.String(6),
                nullable=False,
            ),
            # If this backend has a temp db, we use it to store codelists
            # tables. This helps with permissions management, as we can have
            # write acces to the temp db but not the main db
            schema=self.get_temp_database(),
        )

        # Constuct the queries needed to create and populate this table
        create_query = sqlalchemy.schema.CreateTable(table)
        insert_queries = []
        for codes_batch in split_list_into_batches(
            codes, size=self.max_rows_per_insert
        ):
            insert_query = table.insert().values(
                [(code, codelist.system) for code in codes_batch]
            )
            insert_queries.append(insert_query)

        # Construct the queries needed to clean it up
        cleanup_queries = (
            [sqlalchemy.schema.DropTable(table, if_exists=True)]
            if self.temp_table_needs_dropping(create_query)
            else []
        )

        table.setup_queries = [create_query] + insert_queries
        table.cleanup_queries = cleanup_queries
        return table

    @get_sql_element_no_cache.register
    def get_element_from_value_from_function(self, value: ValueFromFunction):
        # TODO: I'd quite like to build this map by decorating the methods e.g.
        #
        #   @handler_for(DateDifferenceInYears)
        #   def my_handle_fun(...)
        #
        # but the simple thing will do for now. Note we can't use `singledispatchmethod`
        # for this because it doesn't play nicely with subclassing.
        class_method_map = {
            DateDifference: self.date_difference,
            DateAddition: self.date_add,
            DateSubtraction: self.date_subtract,
            DateDeltaAddition: self.date_delta_add,
            DateDeltaSubtraction: self.date_delta_subtract,
            RoundToFirstOfMonth: self.round_to_first_of_month,
            RoundToFirstOfYear: self.round_to_first_of_year,
            YearFromDate: self.year_from_date,
        }

        assert value.__class__ in class_method_map, f"Unsupported function: {value}"

        method = class_method_map[value.__class__]
        argument_expressions = [
            self.get_sql_element_or_value(arg) for arg in value.arguments
        ]
        return method(*argument_expressions)

    def date_difference(self, start_date, end_date, units):
        start_date = type_coerce(start_date, sqlalchemy_types.Date())
        end_date = type_coerce(end_date, sqlalchemy_types.Date())

        unit_conversions = {
            "years": self._convert_date_diff_to_years,
            "months": self._convert_date_diff_to_months,
            "days": self._convert_date_diff_to_days,
            "weeks": self._convert_date_diff_to_weeks,
        }
        return unit_conversions[units](start_date, end_date)

    @staticmethod
    def _date_to_parts(date):
        return (
            sqlalchemy.func.year(date),
            sqlalchemy.func.month(date),
            sqlalchemy.func.day(date),
        )

    def _convert_date_diff_to_years(self, start, end):
        # We do the arithmetic ourselves, to be portable across dbs.
        start_year, start_month, start_day = self._date_to_parts(start)
        end_year, end_month, end_day = self._date_to_parts(end)
        year_diff = end_year - start_year
        date_diff = sqlalchemy.case(
            (end_month > start_month, year_diff),
            (
                sqlalchemy.and_(end_month == start_month, end_day >= start_day),
                year_diff,
            ),
            else_=year_diff - 1,
        )
        return type_coerce(date_diff, sqlalchemy_types.Integer())

    def _convert_date_diff_to_months(
        self, start, end
    ):  # pragma: no cover (re-implement when the QL is in)
        start_year, start_month, start_day = self._date_to_parts(start)
        end_year, end_month, end_day = self._date_to_parts(end)
        year_diff = end_year - start_year

        date_diff = sqlalchemy.case(
            (
                sqlalchemy.and_(end_day >= start_day),
                year_diff * 12 + (end_month - start_month),
            ),
            else_=year_diff * 12 + (end_month - start_month - 1),
        )
        return type_coerce(date_diff, sqlalchemy_types.Integer())

    def _convert_date_diff_to_days(self, start, end):
        """
        Calculate difference between dates in days
        """
        raise NotImplementedError()

    def _convert_date_diff_to_weeks(
        self, start, end
    ):  # pragma: no cover (re-implement when the QL is in)
        """
        Calculate difference in weeks
        Datediff calculates weeks by boundaries crossed.  Since we want the duration in total
        number of whole weeks, use the days calculation to calculate weeks also.
        """
        return sqlalchemy.func.floor(self._convert_date_diff_to_days(start, end) / 7)

    def date_add(self, start_date, number_of_days):
        """
        Add a number of days to a date.
        """
        raise NotImplementedError()

    def date_subtract(self, start_date, number_of_days):
        """
        Add a number of days to a date.
        """
        raise NotImplementedError()

    def date_delta_add(
        self, delta1, delta2
    ):  # pragma: no cover (re-implement when the QL is in)
        return type_coerce((delta1 + delta2), sqlalchemy_types.Integer())

    def date_delta_subtract(
        self, delta1, delta2
    ):  # pragma: no cover (re-implement when the QL is in)
        return type_coerce((delta1 - delta2), sqlalchemy_types.Integer())

    def round_to_first_of_month(self, date):
        raise NotImplementedError

    def round_to_first_of_year(self, date):
        raise NotImplementedError

    def year_from_date(self, date):
        date = type_coerce(date, sqlalchemy_types.Date())
        return sqlalchemy.func.year(date, type_=sqlalchemy_types.Integer())

    def query_to_create_temp_table_from_select_query(
        self, table: TemporaryTable, select_query
    ):
        """
        Return a query to create `table` and populate it with the results of
        `select_query`
        """
        raise NotImplementedError()

    def temp_table_needs_dropping(self, create_table_query):
        """
        Given the query used to create a temporary table, return whether the table needs
        to be explicitly dropped or whether the database will discard it automatically
        when the session terminates
        """
        raise NotImplementedError()

    #
    # DATABASE CONNECTION
    #
    @cached_property
    def engine(self):
        engine_url = sqlalchemy.engine.make_url(self.dsn)
        # Hardcode the specific SQLAlchemy dialect we want to use: this is the
        # dialect the query engine will have been written for and tested with and we
        # don't want to allow global config changes to alter this
        engine_url._get_entrypoint = lambda: self.sqlalchemy_dialect

        autocommit = os.environ.get('ENABLE_ISOLATION_LEVEL_AUTOCOMMIT', default="0") == "1"
        if autocommit:
            engine = sqlalchemy.create_engine(engine_url, future=True, isolation_level="AUTOCOMMIT")
        else:
            engine = sqlalchemy.create_engine(engine_url, future=True)
        # The above relies on abusing SQLAlchemy internals so it's possible it will
        # break in future -- we want to know immediately if it does
        assert isinstance(engine.dialect, self.sqlalchemy_dialect)
        return engine

    def get_temp_table_name(self, name_hint):
        """
        Return a table name based on `name_hint` suitable for use as a temporary table.

        `name_hint` is arbitrary and is only present to make the resulting SQL slightly
        more comprehensible when debugging.
        """
        self.temp_table_count += 1
        return f"{self.temp_table_prefix}{name_hint}_{self.temp_table_count}"

    def get_temp_database(self):
        """Which schema/database should we write temporary tables to."""
        return None


def apply_filter(query, column, operator, value, value_query_node, or_null=False):
    """
    Applies a WHERE condition to the supplied query, specifically:

        WHERE <column> <operator> <value>

    The extra arguments are needed to handle extra complexity in this function which we
    haven't yet had time to refactor away.
    """
    # TODO: This function needs work. Once we've ironed out some of the complexities
    # here it should be possible to have a function which doesn't know anyting about
    # QueryNodes and can therefore be moved to `sqlalchemy_utils`

    # Get the base table
    table_expr = get_primary_table(query)

    # Is the filter value itself potentially drawn from another table?
    if isinstance(value_query_node, (Value, Column)):
        # Find the tables to which it refers
        other_tables = get_referenced_tables(value)
        # If we have a "Value" (i.e. a single value per patient) then we
        # include the other tables in the join
        if isinstance(value_query_node, Value):
            query = include_joined_tables(query, other_tables, "patient_id")
        # If we have a "Column" (i.e. multiple values per patient) then we
        # can't directly join this with our single-value-per-patient query,
        # so we have to use a correlated subquery
        elif isinstance(value_query_node, Column):
            # TODO: I actually think this check is wrong and we'll eventually need to
            # support e.g. a column which is a boolean expression over multiple source
            # columns. But I'll leave it in place for now.
            assert len(other_tables) == 1
            other_table = other_tables[0]
            value = (
                sqlalchemy.select(value)
                .select_from(other_table)
                .where(other_table.c.patient_id == table_expr.c.patient_id)
            )
        else:
            assert False

    if isinstance(value_query_node, Codelist):
        value = sqlalchemy.select(value.c.code).scalar_subquery()
        # FIXME: check this condition is still valid once the new DSL is in,
        # and remove the no cover
        if "system" in table_expr.c:  # pragma: no cover
            # Codelist queries must also match on `system` column if it's present
            system_column = table_expr.c["system"]
            value = value.where(system_column == value_query_node.system)

    column_expr = table_expr.c[column]
    method = getattr(column_expr, operator)
    filter_expr = method(value)

    if or_null:
        null_expr = column_expr.__eq__(None)
        filter_expr = sqlalchemy.or_(filter_expr, null_expr)
    return query.where(filter_expr)


def apply_optimisations(column_definitions):
    """
    Apply various transformations to the supplied query DAG which make it easier to
    generate better performing SQL
    """
    # It's algorithmically easier to apply these transformations by modifying the graph
    # in place, so we copy it first
    column_definitions = copy.deepcopy(column_definitions)

    reify_query_before_selecting_column(column_definitions)

    return column_definitions


def reify_query_before_selecting_column(column_definitions):
    """
    We sometimes need to be able to take a SQLAlchemy query and treat it as something
    table-like from which we can select columns (a process we describe as
    "reification"). In fact, SQLAlchemy will do this implictly for us: we can just
    select columns from a query and it will automatically create a subquery for us.
    However this triggers a warning not to rely on this behaviour. And in any case we
    don't want to use subqueries, we want to use our own TemporaryTable class. So here
    we walk the query graph, find all the places where we select columns and inject an
    operation to explicitly reify the query before doing so.
    """
    # Find all ColumnSelectorNodes and group them by the source node they reference
    selector_types = typing.get_args(ColumnSelectorNode)
    nodes_by_source = defaultdict(list)
    for node in get_all_nodes(column_definitions):
        if isinstance(node, selector_types):
            nodes_by_source[node.source].append(node)
    # Inject a ReifyQuery node between the ColumnSelectorNodes and their source
    for source, child_nodes in nodes_by_source.items():
        # As an optimisation to avoid reifying more data than we need, we determine what
        # columns we're selecting and pass these to the reification method
        columns = {node.column for node in child_nodes}
        new_source = ReifiedQuery(source, tuple(columns))
        for node in child_nodes:
            # These are frozen instances, so we can't set the attribute directly
            object.__setattr__(node, "source", new_source)


def get_all_nodes(column_definitions):
    return list(recurse_over_nodes(column_definitions.values(), set()))


def recurse_over_nodes(nodes, seen):
    for node in nodes:
        yield from recurse_over_nodes(node._get_referenced_nodes(), seen)
        if node not in seen:
            seen.add(node)
            yield node


def split_list_into_batches(lst, size=None):
    # If no size limit specified yield the whole list in one batch
    if size is None:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i : i + size]
