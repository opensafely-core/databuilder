import sqlalchemy
from sqlalchemy.sql.functions import Function as SQLFunction

from databuilder import sqlalchemy_types
from databuilder.query_engines.base_sql import BaseSQLQueryEngine
from databuilder.query_engines.spark_dialect import CreateTemporaryViewAs, SparkDialect
from databuilder.sqlalchemy_utils import GeneratedTable


class SparkQueryEngine(BaseSQLQueryEngine):
    sqlalchemy_dialect = SparkDialect

    def get_date_part(self, date, part):
        assert part in {"YEAR", "MONTH", "DAY"}
        return SQLFunction(part, date, type_=sqlalchemy_types.Integer)

    def date_add_days(self, date, num_days):
        return SQLFunction("DATE_ADD", date, num_days, type_=sqlalchemy_types.Date)

    def to_first_of_month(self, date):
        return SQLFunction("TRUNC", date, "month", type_=sqlalchemy_types.Date)

    def reify_query(self, query):
        # Define a table object with the same columns as the query
        table_name = self.next_intermediate_table_name()
        columns = [
            sqlalchemy.Column(c.name, c.type, key=c.key) for c in query.selected_columns
        ]
        table = GeneratedTable(table_name, sqlalchemy.MetaData(), *columns)
        # Create a temporary (session-scoped) view from the query
        table.setup_queries = [CreateTemporaryViewAs(table, query)]
        return table
