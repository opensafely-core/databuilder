import sqlalchemy
from sqlalchemy.schema import CreateIndex
from sqlalchemy.sql.functions import Function as SQLFunction

from databuilder import sqlalchemy_types
from databuilder.query_engines.base_sql import BaseSQLQueryEngine
from databuilder.query_engines.mssql_dialect import MSSQLDialect, SelectStarInto
from databuilder.sqlalchemy_utils import GeneratedTable


class MSSQLQueryEngine(BaseSQLQueryEngine):
    sqlalchemy_dialect = MSSQLDialect

    # The `#` prefix is an MSSQL-ism which automatically makes the tables session-scoped
    # temporary tables
    intermediate_table_prefix = "#tmp_"

    def get_date_part(self, date, part):
        assert part in {"YEAR", "MONTH", "DAY"}
        return SQLFunction(part, date, type_=sqlalchemy_types.Integer)

    def date_add_days(self, date, num_days):
        return SQLFunction(
            "DATEADD",
            sqlalchemy.text("day"),
            num_days,
            date,
            type_=sqlalchemy_types.Date,
        )

    def to_first_of_year(self, date):
        return SQLFunction(
            "DATEFROMPARTS",
            self.get_date_part(date, "YEAR"),
            1,
            1,
            type_=sqlalchemy_types.Date,
        )

    def reify_query(self, query):
        # Define a table object with the same columns as the query
        table_name = self.next_intermediate_table_name()
        columns = [
            sqlalchemy.Column(c.name, c.type, key=c.key) for c in query.selected_columns
        ]
        table = GeneratedTable(table_name, sqlalchemy.MetaData(), *columns)
        table.setup_queries = [
            # Use the MSSQL `SELECT * INTO ...` construct to create and populate this
            # table
            SelectStarInto(table, query.alias()),
            # As we always join rows on `patient_id` it makes sense to store them on
            # disk in `patient_id` order, which is what creating a clustered index does.
            # (We use `None` as the index name to let SQLAlchemy generate one for us.)
            CreateIndex(
                sqlalchemy.Index(None, table.c.patient_id, mssql_clustered=True)
            ),
        ]
        # The "#" in `intermediate_table_prefix` ensures this is a session-scoped
        # temporary table so there's no explict cleanup needed
        return table
