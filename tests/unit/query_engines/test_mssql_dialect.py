import datetime

import pytest
import sqlalchemy
from sqlalchemy.sql.visitors import iterate

from databuilder import sqlalchemy_types
from databuilder.query_engines.mssql_dialect import MSSQLDialect, SelectStarInto


def test_mssql_date_types():
    # Note: it would be nice to parameterize this test, but given that the
    # inputs are SQLAlchemy expressions I don't know how to do this without
    # constructing the column objects outside of the test, which I don't really
    # want to do.
    date_col = sqlalchemy.Column("date_col", sqlalchemy_types.Date())
    datetime_col = sqlalchemy.Column("datetime_col", sqlalchemy_types.DateTime())
    assert _str(date_col > "2021-08-03") == "date_col > '20210803'"
    assert _str(datetime_col < "2021-03-23") == "datetime_col < '2021-03-23T00:00:00'"
    assert _str(date_col == datetime.date(2021, 5, 15)) == "date_col = '20210515'"
    assert (
        _str(datetime_col == datetime.datetime(2021, 5, 15, 9, 10, 0))
        == "datetime_col = '2021-05-15T09:10:00'"
    )
    assert _str(date_col == None) == "date_col IS NULL"  # noqa: E711
    assert _str(datetime_col == None) == "datetime_col IS NULL"  # noqa: E711
    with pytest.raises(ValueError):
        _str(date_col > "2021")
    with pytest.raises(ValueError):
        _str(datetime_col == "2021-08")
    with pytest.raises(TypeError):
        _str(date_col > 2021)
    with pytest.raises(TypeError):
        _str(datetime_col == 2021)


def test_select_star_into():
    table = sqlalchemy.table("foo", sqlalchemy.Column("bar"))
    query = sqlalchemy.select(table.c.bar).where(table.c.bar > 1)
    target_table = sqlalchemy.table("test")
    select_into = SelectStarInto(target_table, query.alias())
    assert _str(select_into) == (
        "SELECT * INTO test FROM (SELECT foo.bar AS bar \n"
        "FROM foo \n"
        "WHERE foo.bar > 1) AS anon_1"
    )


def test_select_star_into_can_be_iterated():
    # If we don't define the `get_children()` method on `SelectStarInto` we won't get an
    # error when attempting to iterate the resulting element structure: it will just act
    # as a leaf node. But as we rely heavily on query introspection we need to ensure we
    # can iterate over query structures.
    table = sqlalchemy.table("foo", sqlalchemy.Column("bar"))
    query = sqlalchemy.select(table.c.bar).where(table.c.bar > 1)
    target_table = sqlalchemy.table("test")
    select_into = SelectStarInto(target_table, query.alias())

    # Check that SelectStarInto supports iteration by confirming that we can get back to
    # both the target table and the original table by iterating it
    assert any([e is table for e in iterate(select_into)]), "no `table`"
    assert any([e is target_table for e in iterate(select_into)]), "no `target_table`"


def _str(expression):
    compiled = expression.compile(
        dialect=MSSQLDialect(),
        compile_kwargs={"literal_binds": True},
    )
    return str(compiled).strip()
