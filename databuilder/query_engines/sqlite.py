import sqlalchemy
from sqlalchemy.dialects.sqlite.pysqlite import SQLiteDialect_pysqlite

from databuilder import sqlalchemy_types
from databuilder.query_engines.base_sql import BaseSQLQueryEngine


class SQLiteQueryEngine(BaseSQLQueryEngine):
    sqlalchemy_dialect = SQLiteDialect_pysqlite

    def get_date_part(self, date, part):
        format_str = {"YEAR": "%Y", "MONTH": "%m", "DAY": "%d"}[part]
        part_as_str = sqlalchemy.func.strftime(format_str, date)
        return sqlalchemy.cast(part_as_str, sqlalchemy_types.Integer())

    def date_add_days(self, date, num_days):
        num_days_str = sqlalchemy.cast(num_days, sqlalchemy_types.String())
        modifier = num_days_str.concat(" days")
        new_date = sqlalchemy.func.date(date, modifier)
        # Tell SQLAlchemy that the result is a date without doing any CASTing in the SQL
        return sqlalchemy.type_coerce(new_date, sqlalchemy_types.Date())

    def to_first_of_month(self, date):
        new_date = sqlalchemy.func.date(date, "start of month")
        # Tell SQLAlchemy that the result is a date without doing any CASTing in the SQL
        return sqlalchemy.type_coerce(new_date, sqlalchemy_types.Date())
