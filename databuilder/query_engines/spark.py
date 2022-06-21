import sqlalchemy

from databuilder import sqlalchemy_types
from databuilder.query_engines.base_sql import BaseSQLQueryEngine
from databuilder.query_engines.spark_dialect import CreateTemporaryViewAs, SparkDialect
from databuilder.sqlalchemy_utils import GeneratedTable


class SparkQueryEngine(BaseSQLQueryEngine):
    sqlalchemy_dialect = SparkDialect

    def get_date_part(self, date, part):
        func = sqlalchemy.func
        get_part = {"YEAR": func.year, "MONTH": func.month, "DAY": func.day}[part]
        # Tell SQLAlchemy that the result is an int without doing any CASTing in the SQL
        return sqlalchemy.type_coerce(get_part(date), sqlalchemy_types.Integer())

    def date_add_days(self, date, num_days):
        new_date = sqlalchemy.func.date_add(date, num_days)
        # Tell SQLAlchemy that the result is a date without doing any CASTing in the SQL
        return sqlalchemy.type_coerce(new_date, sqlalchemy_types.Date())

    def to_first_of_month(self, date):
        new_date = sqlalchemy.func.trunc(date, "month")
        # Tell SQLAlchemy that the result is a date without doing any CASTing in the SQL
        return sqlalchemy.type_coerce(new_date, sqlalchemy_types.Date())

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
