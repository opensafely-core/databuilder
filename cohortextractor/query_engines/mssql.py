import contextlib
from collections import defaultdict

import sqlalchemy
import sqlalchemy.dialects.mssql

from ..query_language import (
    Column,
    FilteredTable,
    QueryNode,
    Row,
    Table,
    Value,
    ValueFromAggregate,
    ValueFromCategory,
    ValueFromRow,
)
from .base import BaseQueryEngine


def make_table_expression(table_name, columns):
    """
    Return a SQLAlchemy object representing a table with the given name and
    columns
    """
    return sqlalchemy.Table(
        table_name,
        sqlalchemy.MetaData(),
        *[sqlalchemy.Column(column) for column in columns],
    )


def get_joined_tables(query):
    """
    Given a query object return a list of all tables referenced
    """
    tables = []
    from_exprs = list(query.froms)
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


class MssqlQueryEngine(BaseQueryEngine):

    sqlalchemy_dialect = sqlalchemy.dialects.mssql

    def __init__(self, column_definitions, backend):
        super().__init__(column_definitions, backend)
        self._engine = None
        # If no "population" was specified in the column definitions, use a default value
        # which just selects rows that exist by patient_id from the default population
        # table (practice_registrations)
        if "population" not in column_definitions:
            column_definitions["population"] = ValueFromAggregate(
                source=Table(name="practice_registrations"),
                function="exists",
                column="patient_id",
            )

        # Walk the nodes and identify output groups
        self.output_groups = self.get_output_groups(column_definitions)
        self.output_group_tables = {}
        self.output_group_tables_queries = {}

    #
    # QUERY DAG METHODS AND NODE INTERACTION
    #
    def get_output_groups(self, column_definitions):
        """
        Walk over all nodes in the query DAG looking for output nodes (leaf nodes which
        represent a value or a column of values) and group them together by "type" and
        "source" (source being the parent node from which they are derived). Each such
        group of outputs can be generated by a single query so we want them grouped together.
        """
        output_groups = defaultdict(list)
        for node in self.get_all_query_nodes(column_definitions):
            if self.is_output_node(node):
                output_groups[self.get_type_and_source(node)].append(node)
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
        # defined them), and the fact that `walk_query_dag` gives its results
        # in reverse-topological order (it starts with the leaf nodes and works
        # its way back up). So we reverse the order of the leaf nodes, pass
        # them through `walk_query_dag` and then reverse the whole list of
        # results. Further down the line we might need to this "properly"
        # (maybe using the networkx library?) but this will do us for now.
        leaf_nodes = list(column_definitions.values())
        leaf_nodes.reverse()
        all_nodes = list(self.walk_query_dag(leaf_nodes))
        all_nodes.reverse()
        return all_nodes

    def walk_query_dag(self, nodes, seen=None):
        if seen is None:
            seen = set(nodes)
        else:
            seen.update(nodes)
        parents = set()
        for node in nodes:
            yield node
            for reference in [
                *self.get_query_node_references(node),
                getattr(node, "value", None),
            ]:
                if isinstance(reference, QueryNode) and reference not in seen:
                    parents.add(reference)
        if parents:
            yield from self.walk_query_dag(parents, seen=seen)

    @staticmethod
    def is_output_node(node):
        return isinstance(node, (ValueFromRow, ValueFromAggregate, Column))

    @staticmethod
    def is_category_node(node):
        return isinstance(node, ValueFromCategory)

    def get_type_and_source(self, node):
        assert self.is_output_node(node)
        return type(node), self.get_query_node_references(node)[0]

    @staticmethod
    def get_output_column_name(node):
        if isinstance(node, ValueFromAggregate):
            return f"{node.column}_{node.function}"
        elif isinstance(node, (ValueFromRow, Column)):
            return node.column
        else:
            raise TypeError(f"Unhandled type: {node}")

    @staticmethod
    def get_query_node_references(node):
        if hasattr(node, "definitions"):
            return tuple({group.source for group in node.definitions.values()})
        elif hasattr(node, "source"):
            return (node.source,)
        else:
            return []

    def get_node_list(self, node):
        """For a single node, get a list of it and all its parents in order"""
        node_list = []
        while True:
            node_list.append(node)
            if type(node) is Table:
                break
            else:
                node = self.get_query_node_references(node)[0]
        node_list.reverse()
        return node_list

    #
    # DATABASE CONNECTION
    #
    @property
    def engine(self):
        if self._engine is None:
            engine_url = sqlalchemy.engine.make_url(self.backend.database_url)
            engine_url = engine_url.set(drivername="mssql+pymssql")
            self._engine = sqlalchemy.create_engine(engine_url, echo=True, future=True)
        return self._engine

    #
    # MSSQL-SPECIFIC QUERIES
    #
    def create_output_group_tables(self):
        """Queries to generate and populate interim tables for each output"""
        # For each group of output nodes (nodes that produce a single output value),
        # make a table object representing a temporary table into which we will write the required
        # values
        for i, (group, output_nodes) in enumerate(self.output_groups.items()):
            # The `#` prefix makes this a session-scoped temporary table
            table_name = f"#group_table_{i}"
            columns = {self.get_output_column_name(output) for output in output_nodes}
            self.output_group_tables[group] = make_table_expression(
                table_name, {"patient_id"} | columns
            )

        # For each group of output nodes, build a query expression
        # to populate the associated temporary table
        self.output_group_tables_queries = {
            group: self.get_query_expression(output_nodes)
            for group, output_nodes in self.output_groups.items()
        }

    def get_select_expression(self, base_table, columns):
        # every table must have a patient_id column; select it and the specified columns
        columns = sorted({"patient_id"}.union(columns))
        table_expr = self.backend.get_table_expression(base_table.name)
        column_objs = [table_expr.c[column] for column in columns]
        query = sqlalchemy.select(column_objs).select_from(table_expr)
        return query

    def get_query_expression(self, output_nodes):
        """
        From a group of output nodes that represent the route to a single output value,
        generate the query that will return the value from its source table(s)
        """
        # output_nodes must all be of the same group, and all have the same output
        # type and source, so we arbitrarily use the first one
        output_type, query_node = self.get_type_and_source(output_nodes[0])

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

    def get_value_expression(self, value):
        """
        Given a single value output node, select it from its interim table(s)
        Return the expression to select it, and the table(s) to select it from
        """
        tables = None
        value_expr = value
        if self.is_category_node(value):
            category_definitions = value.definitions.copy()
            tables = {
                label: self.output_group_tables[
                    self.get_type_and_source(category_definition.source)
                ]
                for label, category_definition in category_definitions.items()
            }
            category_mapping = {}
            for label, category_definition in category_definitions.items():
                table = tables[label]
                column = table.c[category_definition.source.column]
                method = getattr(column, category_definition.operator)
                category_mapping[label] = method(category_definition.value)
            value_expr = self.get_case_expression(category_mapping, value.default)
            tables = tuple(tables.values())
        elif self.is_output_node(value):
            table = self.output_group_tables[self.get_type_and_source(value)]
            column = self.get_output_column_name(value)
            value_expr = table.c[column]
            tables = (table,)
        return value_expr, tables

    def get_case_expression(self, mapping, default):
        return sqlalchemy.case(
            [(expression, label) for label, expression in mapping.items()],
            else_=default,
        )

    def apply_aggregates(self, query, aggregate_nodes):
        """
        For each aggregate node, get the query that will select it with its generated
        column label, plus the patient id column, and then group by the patient id.

        e.g. For the default population exists query, it will select patient_id as a column
        labelled patient_id_exists from the entire column of patient_id and then group
         by patient id; i.e.

        SELECT practice_registrations.patient_id, :param_1 AS patient_id_exists
        FROM (SELECT PatientId AS patient_id FROM practice_registrations) AS practice_registrations
        GROUP BY practice_registrations.patient_id

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
        if other_tables is not None:
            assert len(other_tables) == 1
            other_table = other_tables[0]
            # If we have a "Value" (i.e. a single value per patient) then we
            # include the other table in the join
            if isinstance(filter_node.value, Value):
                query = self.include_joined_table(query, other_table)
            # If we have a "Column" (i.e. multipe values per patient) then we
            # can directly join this with our single-value-per-patient query,
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

        column = table_expr.c[column_name]
        method = getattr(column, operator_name)
        return query.where(method(value_expr))

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
            query.froms[0],
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

    def get_sql(self):
        """Build the SQL"""
        self.create_output_group_tables()
        sql = []
        # Generate each of the interim output group tables and populate them
        for group, table in self.output_group_tables.items():
            query = self.output_group_tables_queries[group]
            query_sql = self.query_expression_to_sql(query)
            sql.append(f"SELECT * INTO {table.name} FROM (\n{query_sql}\n) t")
        # Add the big query that creates the base population table and its columns,
        # selected from the output group tables
        sql.append(self.query_expression_to_sql(self.generate_results_query()))
        return "\n\n\n".join(sql)

    def query_expression_to_sql(self, query):
        return str(
            query.compile(
                dialect=self.sqlalchemy_dialect.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )

    @contextlib.contextmanager
    def execute_query(self):
        """Execute a query against an MSSQL backend"""
        sql = self.get_sql()
        with self.engine.connect() as cursor:
            result = cursor.execute(sqlalchemy.text(sql))
            yield result
