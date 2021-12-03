import contextlib
from collections import defaultdict
from typing import Optional

import sqlalchemy
import sqlalchemy.dialects.mssql
import sqlalchemy.schema
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.sql.expression import type_coerce

from .. import sqlalchemy_types
from ..query_language import (
    Codelist,
    Column,
    Comparator,
    DateDifferenceInYears,
    FilteredTable,
    Row,
    Table,
    Value,
    ValueFromAggregate,
    ValueFromCategory,
    ValueFromFunction,
    ValueFromRow,
)
from .base import BaseQueryEngine


def get_joined_tables(query):
    """
    Given a query object return a list of all tables referenced
    """
    tables = []
    from_exprs = list(query.get_final_froms())
    while from_exprs:
        next_expr = from_exprs.pop()
        if isinstance(next_expr, sqlalchemy.sql.selectable.Join):
            from_exprs.extend([next_expr.left, next_expr.right])
        else:
            tables.append(next_expr)
    # The above algorithm produces tables in right to left order, but it makes
    # more sense to return them as left to right
    tables.reverse()
    return tables


def get_primary_table(query):
    """
    Return the left-most table referenced in the query
    """
    return get_joined_tables(query)[0]


class BaseSQLQueryEngine(BaseQueryEngine):

    sqlalchemy_dialect: type[Dialect]

    # No limit by default although some DBMSs may impose one
    max_rows_per_insert: Optional[int] = None

    def __init__(self, column_definitions, backend):
        super().__init__(column_definitions, backend)
        self._engine = None

        # Walk the nodes and identify output groups
        all_nodes = self.get_all_query_nodes(column_definitions)
        self.output_groups = self.get_output_groups(all_nodes)
        self.codelists = [node for node in all_nodes if isinstance(node, Codelist)]
        self.codelist_tables = {}
        self.codelist_tables_queries = []
        self.output_group_tables = {}
        self.output_group_tables_queries = {}

    #
    # QUERY DAG METHODS AND NODE INTERACTION
    #
    def get_output_groups(self, all_nodes):
        """
        Walk over all nodes in the query DAG looking for output nodes (leaf nodes which
        represent a value or a column of values) and group them together by "type" and
        "source" (source being the parent node from which they are derived). Each such
        group of outputs can be generated by a single query so we want them grouped together.
        """
        output_groups = defaultdict(list)
        for node in all_nodes:
            if self.is_output_node(node):
                output_groups[self.get_output_group(node)].append(node)
        return output_groups

    def get_all_query_nodes(self, column_definitions):
        """
        Return a list of all QueryNodes used in the supplied column_definitions
        in topological order (i.e. a node will never be referenced before it
        appears). We need this so that we construct temporary tables in the
        right order and never try to reference a table which we haven't yet
        populated.
        """
        # Given the way the query DAG is currently constructed we can use a
        # simple trick to get the nodes in topological order. We exploit the
        # fact that the Python-based DSL will naturally enforce a topological
        # order on the leaf nodes (you can't reference variables before you've
        # defined them). Then for each leaf node we traverse depth-first,
        # returning parents before their children. Further down the line we
        # might need to this "properly" (maybe using the networkx library?) but
        # this will do us for now.
        leaf_nodes = column_definitions.values()
        return self.walk_query_dag(leaf_nodes)

    def walk_query_dag(self, nodes):
        def recurse(nodes, seen):
            for node in nodes:
                yield from recurse(node._get_referenced_nodes(), seen)
                if node not in seen:
                    seen.add(node)
                    yield node

        return list(recurse(nodes, set()))

    @staticmethod
    def is_output_node(node):
        return isinstance(node, (ValueFromRow, ValueFromAggregate, Column))

    @staticmethod
    def is_category_node(node):
        return isinstance(node, ValueFromCategory)

    def get_output_group(self, node):
        assert self.is_output_node(node)
        return type(node), node.source

    @staticmethod
    def get_output_column_name(node):
        if isinstance(node, ValueFromAggregate):
            return f"{node.column}_{node.function}"
        elif isinstance(node, (ValueFromRow, Column)):
            return node.column
        else:
            assert False, f"Unhandled type: {node}"

    def get_node_list(self, node):
        """For a single node, get a list of it and all its parents in order"""
        node_list = []
        while True:
            node_list.append(node)
            if type(node) is Table:
                break
            else:
                node = node.source
        node_list.reverse()
        return node_list

    #
    # DATABASE CONNECTION
    #
    @property
    def engine(self):
        if self._engine is None:
            engine_url = sqlalchemy.engine.make_url(self.backend.database_url)
            # Hardcode the specific SQLAlchemy dialect we want to use: this is the
            # dialect the query engine will have been written for and tested with and we
            # don't want to allow global config changes to alter this
            engine_url._get_entrypoint = lambda: self.sqlalchemy_dialect
            self._engine = sqlalchemy.create_engine(engine_url, future=True)
            # The above relies on abusing SQLAlchemy internals so it's possible it will
            # break in future -- we want to know immediately if it does
            assert isinstance(self._engine.dialect, self.sqlalchemy_dialect)
        return self._engine

    def create_output_group_tables(self):
        """Queries to generate and populate interim tables for each output"""
        # For each group of "output nodes" (roughly, "nodes which we know how
        # to build a single select query for"), build a SQLAlchemy query to get
        # their values.
        for i, (group, output_nodes) in enumerate(self.output_groups.items()):
            query = self.get_query_expression(group, output_nodes)
            self.output_group_tables_queries[group] = query
            # Create a Table object representing a temporary table into which
            # we'll write the results of the query
            table_name = self.get_temp_table_name(f"group_table_{i}")
            columns = [
                sqlalchemy.Column(c.name, c.type) for c in query.selected_columns
            ]
            table = sqlalchemy.Table(
                table_name,
                sqlalchemy.MetaData(),
                *columns,
            )
            self.output_group_tables[group] = table

    def create_codelist_tables(self):
        """
        For each codelist in use, build a SQLAlchemy representation of the
        temporary table needed to store that codelist and then generate the
        queries necessary to create and populate those tables
        """
        for n, codelist in enumerate(self.codelists):
            codes = codelist.codes
            max_code_len = max(map(len, codes))
            collation = "Latin1_General_BIN"
            table_name = self.get_temp_table_name(f"codelist_{n}")
            table = sqlalchemy.Table(
                table_name,
                sqlalchemy.MetaData(),
                sqlalchemy.Column(
                    "code",
                    sqlalchemy.types.String(max_code_len, collation=collation),
                    nullable=False,
                ),
                sqlalchemy.Column(
                    "system",
                    sqlalchemy.types.String(6),
                    nullable=False,
                ),
            )
            self.codelist_tables[codelist] = table
            # Constuct the queries needed to create and populate this table
            self.codelist_tables_queries.append(sqlalchemy.schema.CreateTable(table))
            for codes_batch in split_list_into_batches(
                codes, size=self.max_rows_per_insert
            ):
                insert_query = table.insert().values(
                    [(code, codelist.system) for code in codes_batch]
                )
                self.codelist_tables_queries.append(insert_query)

    def get_temp_table_name(self, table_name):
        """
        Return a table name based on `table_name` but suitable for use as a
        temporary table.

        It's the caller's responsibility to ensure `table_name` is unique
        within this session; it's this function's responsibility to ensure it
        doesn't clash with any concurrent extracts
        """
        raise NotImplementedError()

    def get_select_expression(self, base_table, columns):
        # every table must have a patient_id column; select it and the specified columns
        columns = sorted({"patient_id"}.union(columns))
        table_expr = self.backend.get_table_expression(base_table.name)
        try:
            column_objs = [table_expr.c[column] for column in columns]
        except KeyError as unknown_key:
            raise KeyError(
                f"Column {unknown_key} not found in table '{base_table.name}'"
            )

        query = sqlalchemy.select(column_objs).select_from(table_expr)
        return query

    def get_query_expression(self, group, output_nodes):
        """
        From a group of output nodes that represent the route to a single output value,
        generate the query that will return the value from its source table(s)
        """
        output_type, query_node = group

        # Queries (currently) always have a linear structure so we can
        # decompose them into a list
        node_list = self.get_node_list(query_node)
        # The start of the list should always be an unfiltered Table
        base_table = node_list.pop(0)
        assert isinstance(base_table, Table)

        # If there's an operation applied to reduce the results to a single row
        # per patient, then that will be the final element of the list
        row_selector = None
        if issubclass(output_type, ValueFromRow):
            row_selector = node_list.pop()
            assert isinstance(row_selector, Row)

        # All remaining nodes should be filter operations
        filters = node_list
        assert all(isinstance(filter_node, FilteredTable) for filter_node in filters)

        # Get all the required columns from the base table
        selected_columns = {node.column for node in output_nodes}
        query = self.get_select_expression(base_table, selected_columns)
        # Apply all filter operations
        for filter_node in filters:
            query = self.apply_filter(query, filter_node)

        # Apply the row selector to select the single row per patient
        if row_selector is not None:
            query = self.apply_row_selector(
                query,
                sort_columns=row_selector.sort_columns,
                descending=row_selector.descending,
            )

        if issubclass(output_type, ValueFromAggregate):
            query = self.apply_aggregates(query, output_nodes)

        return query

    def get_population_table_query(self, population):
        """Build the query that selects the patient population we're interested in"""
        is_included, tables = self.get_value_expression(population)
        assert len(tables) == 1
        population_table = tables[0]
        return (
            sqlalchemy.select([population_table.c.patient_id.label("patient_id")])
            .select_from(population_table)
            .where(is_included == True)  # noqa: E712
        )

    def build_condition_statement(self, comparator):
        """
        Traverse a comparator's left and right hand sides in order and build the nested
        condition statement along with a tuple of the tables referenced
        """
        if comparator.connector is not None:
            assert isinstance(comparator.lhs, Comparator) and isinstance(
                comparator.rhs, Comparator
            )
            left_conditions, left_tables = self.build_condition_statement(
                comparator.lhs
            )
            right_conditions, right_tables = self.build_condition_statement(
                comparator.rhs
            )
            connector = getattr(sqlalchemy, comparator.connector)
            condition_statement = connector(left_conditions, right_conditions)
            tables = tuple(set(left_tables + right_tables))
        else:
            lhs, tables = self.get_value_expression(comparator.lhs)
            method = getattr(lhs, comparator.operator)
            condition_statement = method(comparator.rhs)

        if comparator.negated:
            condition_statement = sqlalchemy.not_(condition_statement)

        return condition_statement, tables

    def get_value_expression(self, value):
        """
        Given a single value output node, select it from its interim table(s)
        Return the expression to select it, and the table(s) to select it from
        """
        tables = ()
        value_expr = value
        if self.is_category_node(value):
            category_mapping = {}
            tables = set()
            for label, category_definition in value.definitions.items():
                # A category definition is always a single Comparator, which may contain
                # nested Comparators
                condition_statement, condition_tables = self.build_condition_statement(
                    category_definition
                )
                category_mapping[label] = condition_statement
                tables.update(condition_tables)
            value_expr = self.get_case_expression(category_mapping, value.default)
            tables = tuple(tables)
        elif self.is_output_node(value):
            table = self.output_group_tables[self.get_output_group(value)]
            column = self.get_output_column_name(value)
            value_expr = table.c[column]
            tables = (table,)
        elif isinstance(value, Codelist):
            codelist_table = self.codelist_tables[value]
            value_expr = sqlalchemy.select(codelist_table.c.code).scalar_subquery()
        elif isinstance(value, ValueFromFunction):
            value_expr, tables = self.get_expression_for_value_from_function(value)
        return value_expr, tables

    def get_case_expression(self, mapping, default):
        return sqlalchemy.case(
            [(expression, label) for label, expression in mapping.items()],
            else_=default,
        )

    def get_expression_for_value_from_function(self, value):
        argument_expressions = []
        tables = set()
        for arg in value.arguments:
            arg_expr, arg_tables = self.get_value_expression(arg)
            argument_expressions.append(arg_expr)
            tables.update(arg_tables)

        # TODO: I'd quite like to build this map by decorating the methods e.g.
        #
        #   @handler_for(DateDifferenceInYears)
        #   def my_handle_fun(...)
        #
        # but the simple thing will do for now.
        class_method_map = {DateDifferenceInYears: self.date_difference_in_years}

        assert value.__class__ in class_method_map, f"Unsupported function: {value}"

        method = class_method_map[value.__class__]
        value_expression = method(*argument_expressions)

        return value_expression, tuple(tables)

    def date_difference_in_years(self, start_date, end_date):
        start_date = type_coerce(start_date, sqlalchemy_types.Date())
        end_date = type_coerce(end_date, sqlalchemy_types.Date())

        # We do the arithmetic ourselves, to be portable across dbs.
        start_year = sqlalchemy.func.year(start_date)
        start_month = sqlalchemy.func.month(start_date)
        start_day = sqlalchemy.func.day(start_date)

        end_year = sqlalchemy.func.year(end_date)
        end_month = sqlalchemy.func.month(end_date)
        end_day = sqlalchemy.func.day(end_date)

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

    def apply_aggregates(self, query, aggregate_nodes):
        """
        For each aggregate node, get the query that will select it with its generated
        column label, plus the patient id column, and then group by the patient id.
        """
        columns = [
            self.get_aggregate_column(query, aggregate_node)
            for aggregate_node in aggregate_nodes
        ]
        query = query.with_only_columns([query.selected_columns.patient_id] + columns)
        query = query.group_by(query.selected_columns.patient_id)

        return query

    def get_aggregate_column(self, query, aggregate_node):
        """
        For an aggregate node, build the column to hold its value
        Aggregate column names are a combination of column and aggregate function,
        e.g. "patient_id_exists"
        """
        output_column = self.get_output_column_name(aggregate_node)
        if aggregate_node.function == "exists":
            return sqlalchemy.literal(True).label(output_column)
        else:
            # The aggregate node function is a string corresponding to an available
            # sqlalchemy function (e.g. "exists", "count")
            function = getattr(sqlalchemy.func, aggregate_node.function)
            source_column = aggregate_node.column
            return function(query.selected_columns[source_column]).label(output_column)

    def apply_filter(self, query, filter_node):
        # Get the base table
        table_expr = get_primary_table(query)

        column_name = filter_node.column
        operator_name = filter_node.operator
        # Does this filter require another table? i.e. is the filter value itself an
        # Output node, which has a source that we may need to include here
        value_expr, other_tables = self.get_value_expression(filter_node.value)
        if other_tables:
            assert len(other_tables) == 1
            other_table = other_tables[0]
            # If we have a "Value" (i.e. a single value per patient) then we
            # include the other table in the join
            if isinstance(filter_node.value, Value):
                query = self.include_joined_table(query, other_table)
            # If we have a "Column" (i.e. multiple values per patient) then we
            # can't directly join this with our single-value-per-patient query,
            # so we have to use a correlated subquery
            elif isinstance(filter_node.value, Column):
                value_expr = (
                    sqlalchemy.select(value_expr)
                    .select_from(other_table)
                    .where(other_table.c.patient_id == table_expr.c.patient_id)
                )
            else:
                # Shouldn't get any other type here
                assert False

        if isinstance(filter_node.value, Codelist) and "system" in table_expr.c:
            # Codelist queries must also match on `system` column if it's present
            system_column = table_expr.c["system"]
            value_expr = value_expr.where(system_column == filter_node.value.system)

        column = table_expr.c[column_name]
        method = getattr(column, operator_name)
        query_expr = method(value_expr)

        if filter_node.or_null:
            null_expr = column.__eq__(None)
            query_expr = sqlalchemy.or_(query_expr, null_expr)
        return query.where(query_expr)

    @staticmethod
    def apply_row_selector(query, sort_columns, descending):
        """
        Generate query to apply a row selector by sorting by sort_columns in
        specified direction, and then selecting the first row
        """
        # Get the base table - the first in the FROM clauses
        table_expr = get_primary_table(query)

        # Find all the selected column names
        column_names = [column.name for column in query.selected_columns]

        # Query to select the columns that we need to sort on
        order_columns = [table_expr.c[column] for column in sort_columns]
        # change ordering to descending on all order columns if necessary
        if descending:
            order_columns = [c.desc() for c in order_columns]

        # Number rows sequentially over the order by columns for each patient id
        row_num = (
            sqlalchemy.func.row_number()
            .over(order_by=order_columns, partition_by=table_expr.c.patient_id)
            .label("_row_num")
        )
        # Add the _row_num column and select just the first row
        query = query.add_columns(row_num)
        subquery = query.alias()
        query = sqlalchemy.select([subquery.c[column] for column in column_names])
        query = query.select_from(subquery).where(subquery.c._row_num == 1)
        return query

    @staticmethod
    def include_joined_table(query, table):
        if table.name in [t.name for t in get_joined_tables(query)]:
            return query
        join = sqlalchemy.join(
            query.get_final_froms()[0],
            table,
            query.selected_columns.patient_id == table.c.patient_id,
            isouter=True,
        )
        return query.select_from(join)

    def generate_results_query(self):
        """Query to generate the final single results table"""
        # `population` is a special-cased boolean column, it doesn't appear
        # itself in the output but it determines what rows are included
        # Build the base results table from the population table
        column_definitions = self.column_definitions.copy()
        population = column_definitions.pop("population")
        results_query = self.get_population_table_query(population)

        # Build big JOIN query which selects the results
        for column_name, output_node in column_definitions.items():
            # For each output column, generate the query that selects it from its interim table(s)
            # For most outputs there will just be a single interim table.  Category outputs
            # may require more than one.
            column, tables = self.get_value_expression(output_node)
            # Then generate the query to join on it
            for table in tables:
                results_query = self.include_joined_table(results_query, table)

            # Add this column to the final selected results
            results_query = results_query.add_columns(column.label(column_name))

        return results_query

    def get_queries(self):
        """Build the list of SQL queries to execute"""
        self.create_codelist_tables()
        self.create_output_group_tables()
        queries = []
        # Create and populate tables containing codelists
        queries.extend(self.codelist_tables_queries)
        # Generate each of the interim output group tables and populate them
        for group, table in self.output_group_tables.items():
            query = self.output_group_tables_queries[group]
            queries.append(self.write_query_to_table(table, query))
        # Add the big query that creates the base population table and its columns,
        # selected from the output group tables
        queries.append(self.generate_results_query())
        return queries

    def write_query_to_table(self, table, query):
        """
        Returns a new query which, when executed, writes the results of `query`
        into `table`
        """
        raise NotImplementedError()

    @contextlib.contextmanager
    def execute_query(self):
        queries = self.get_queries()
        with self.engine.connect() as cursor:
            for query in queries:
                result = cursor.execute(query)
            # We're only interested in the results from the final query
            yield result
            self.post_execute_cleanup(cursor)

    def post_execute_cleanup(self, cursor):
        """
        A no-op by default but subclasses can implement cleanup logic here
        """


def split_list_into_batches(lst, size=None):
    # If no size limit specified yield the whole list in one batch
    if size is None:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i : i + size]