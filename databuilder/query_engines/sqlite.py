import sqlalchemy
from sqlalchemy.dialects.sqlite.pysqlite import SQLiteDialect_pysqlite
from sqlalchemy.sql.functions import Function as SQLFunction

from databuilder import sqlalchemy_types
from databuilder.query_engines.base_sql import BaseSQLQueryEngine


class SQLiteQueryEngine(BaseSQLQueryEngine):
    sqlalchemy_dialect = SQLiteDialect_pysqlite

    def get_date_part(self, date, part):
        format_str = {"YEAR": "%Y", "MONTH": "%m", "DAY": "%d"}[part]
        part_as_str = SQLFunction("STRFTIME", format_str, date)
        return sqlalchemy.cast(part_as_str, sqlalchemy_types.Integer)

    def date_add_days(self, date, num_days):
        num_days_str = sqlalchemy.cast(num_days, sqlalchemy_types.String)
        modifier = num_days_str.concat(" days")
        return SQLFunction("DATE", date, modifier, type_=sqlalchemy_types.Date)

    def to_first_of_month(self, date):
        return SQLFunction("DATE", date, "start of month", type_=sqlalchemy_types.Date)
