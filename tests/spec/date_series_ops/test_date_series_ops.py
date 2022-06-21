from datetime import date

from ..tables import p

title = "Operations which apply to all series containing dates"

table_data = {
    p: """
          |     d1     | i1
        --+------------+-----
        1 | 1990-01-02 | 100
        2 | 2000-03-04 | 200
        3 |            |
        """,
}


def test_get_year(spec_test):
    spec_test(
        table_data,
        p.d1.year,
        {
            1: 1990,
            2: 2000,
            3: None,
        },
    )


def test_get_month(spec_test):
    spec_test(
        table_data,
        p.d1.month,
        {
            1: 1,
            2: 3,
            3: None,
        },
    )


def test_get_day(spec_test):
    spec_test(
        table_data,
        p.d1.day,
        {
            1: 2,
            2: 4,
            3: None,
        },
    )


def test_add_days(spec_test):
    spec_test(
        table_data,
        p.d1.add_days(p.i1),
        {
            1: date(1990, 4, 12),
            2: date(2000, 9, 20),
            3: None,
        },
    )


def test_subtract_days(spec_test):
    spec_test(
        table_data,
        p.d1.subtract_days(p.i1),
        {
            1: date(1989, 9, 24),
            2: date(1999, 8, 17),
            3: None,
        },
    )


def test_difference_in_years(spec_test):
    # TODO: We should have some text in the documentation explaining that this is
    # "number of entire elapsed years", as you'd use to calculate someone's age, rather
    # than "time difference rounded to nearest year" or anything like that.
    table_data = {
        p: """
              |     d1
            --+------------
            1 | 1990-01-30
            2 | 2000-01-15
            3 | 2020-01-20
            4 | 2022-01-10
            5 |
            """,
    }
    spec_test(
        table_data,
        p.d1.difference_in_years(date(2020, 1, 15)),
        {
            1: 29,
            2: 20,
            3: -1,
            4: -2,
            5: None,
        },
    )


def test_to_first_of_month(spec_test):
    table_data = {
        p: """
              |     d1
            --+------------
            1 | 1990-01-01
            2 | 1990-01-31
            3 |
            """,
    }
    spec_test(
        table_data,
        p.d1.to_first_of_month(),
        {
            1: date(1990, 1, 1),
            2: date(1990, 1, 1),
            3: None,
        },
    )
