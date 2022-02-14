from datetime import date

import pytest

from ..lib.mock_backend import (
    CTV3Events,
    RegistrationHistory,
    ctv3_event,
    patient,
    positive_test,
)
from .query_model_convert_to_new import convert as convert_to_new
from .query_model_old import Codelist, categorise, table

# Mark the whole module as containing integration tests
pytestmark = pytest.mark.integration


def make_codelist(*codes, system="ctv3"):
    return Codelist(codes, system=system)


class OldCohortWithPopulation:
    def __init_subclass__(cls):
        if not hasattr(cls, "population"):  # pragma: no cover
            cls.population = table("practice_registrations").exists()


def convert(cohort_using_old_query_model):
    """
    Take a cohort class defined using the old Query Model and return one defined using
    the new. This allows us to retain these old test cases as a harness while we
    refactor the Query Engine.
    """
    columns = {
        key: value
        for key, value in vars(cohort_using_old_query_model).items()
        if not key.startswith("_")
    }
    new_columns = convert_to_new(columns)
    return type("Cohort", (), new_columns)


def test_query_engine_caches_sql_engine(engine):
    empty_cohort = {}
    query_engine = engine.query_engine_class(
        empty_cohort, engine.backend(database_url="foo://localhost")
    )
    # Check that the property caches the results and gives us the same object each time
    sql_engine_1 = query_engine.engine
    sql_engine_2 = query_engine.engine
    assert sql_engine_1 is sql_engine_2


def test_run_generated_sql_get_single_column_default_population(engine):
    input_data = [
        patient(
            1,
            ctv3_event("Code1"),
        ),
        # patient 2 has an event, but no RegistrationHistory entry
        CTV3Events(PatientId=2, EventCode="Code2", System="ctv3"),
    ]
    engine.setup(input_data)

    # Cohort to extract all clinical events and return just the code column
    # Note that the RegistrationHistory table is used for the default population query
    # It will join the two tables on patient_id and only rows that exist in the RegistrationHistory
    # table will be returned
    class Cohort(OldCohortWithPopulation):
        output_value = table("clinical_events").first_by("patient_id").get("code")

    assert engine.extract(convert(Cohort)) == [dict(patient_id=1, output_value="Code1")]


def test_run_generated_sql_get_single_column_specified_population(engine):
    input_data = [
        patient(1, ctv3_event("Code1")),
        # patient 2 has an event, but no RegistrationHistory entry
        CTV3Events(PatientId=2, EventCode="Code2", System="ctv3"),
    ]
    engine.setup(input_data)

    # Cohort to extract all clinical events and return just the code column
    # Note that the RegistrationHistory table is used for the default population query
    # It will join the two tables on patient_id and only rows that exist in the RegistrationHistory
    # table will be returned
    class Cohort:
        output_value = table("clinical_events").first_by("patient_id").get("code")
        population = table("practice_registrations").exists()

    assert engine.extract(convert(Cohort)) == [dict(patient_id=1, output_value="Code1")]


def test_run_generated_sql_get_multiple_columns(engine):
    input_data = [
        patient(1, ctv3_event("Code1"), positive_test(True)),
        patient(2, ctv3_event("Code2"), positive_test(False)),
    ]
    engine.setup(input_data)

    # Cohort to extract all clinical events and positive tests
    class Cohort(OldCohortWithPopulation):
        output_value = table("clinical_events").first_by("patient_id").get("code")
        positive = table("positive_tests").first_by("patient_id").get("result")

    assert engine.extract(convert(Cohort)) == [
        dict(patient_id=1, output_value="Code1", positive=True),
        dict(patient_id=2, output_value="Code2", positive=False),
    ]


def test_extract_get_single_column(engine):
    input_data = [
        patient(1, ctv3_event("Code1")),
        CTV3Events(PatientId=2, EventCode="Code2", System="ctv3"),
    ]
    engine.setup(input_data)

    # Cohort to extract all clinical events and return just the code column
    # Note that the RegistrationHistory table is used for the default population query
    # It will join the two tables on patient_id and only rows that exist in the RegistrationHistory
    # table will be returned
    class Cohort(OldCohortWithPopulation):
        output_value = table("clinical_events").first_by("patient_id").get("code")

    result = engine.extract(convert(Cohort))
    assert list(result) == [dict(patient_id=1, output_value="Code1")]


@pytest.mark.parametrize(
    "code_output,date_output,expected",
    [
        (
            table("clinical_events").latest().get("code"),
            table("clinical_events").latest().get("date"),
            [
                dict(patient_id=1, code="Code2", date=date(2021, 5, 2)),
                dict(patient_id=2, code="Code1", date=date(2021, 6, 5)),
            ],
        ),
        (
            table("clinical_events").earliest().get("code"),
            table("clinical_events").earliest().get("date"),
            [
                dict(patient_id=1, code="Code1", date=date(2021, 1, 3)),
                dict(patient_id=2, code="Code1", date=date(2021, 2, 4)),
            ],
        ),
    ],
)
def test_run_generated_sql_get_single_row_per_patient(
    engine, code_output, date_output, expected
):
    input_data = [
        patient(
            1,
            ctv3_event("Code1", "2021-01-03"),
            ctv3_event("Code1", "2021-02-01"),
            ctv3_event("Code2", "2021-05-02"),
        ),
        patient(
            2, ctv3_event("Code1", "2021-06-05"), ctv3_event("Code1", "2021-02-04")
        ),
    ]
    engine.setup(input_data)

    # Cohort to extract the earliest/latest event for each patient, and return code and date
    class Cohort(OldCohortWithPopulation):
        code = code_output
        date = date_output

    assert engine.extract(convert(Cohort)) == expected


@pytest.mark.parametrize(
    "data,filtered_table,expected",
    [
        (
            [
                patient(1, ctv3_event("Code1", "2021-01-01", 10)),  # equal
                patient(2, ctv3_event("Code2", "2021-02-02", 20)),  # not equal
            ],
            table("clinical_events").filter("code", is_in=make_codelist("Code1")),
            [
                dict(patient_id=1, code="Code1", date=date(2021, 1, 1), value=10),
            ],
        ),
        (
            [
                patient(1, ctv3_event("Code1", "2021-01-01", 10)),  # both equal
                patient(2, ctv3_event("Code1", "2021-01-02", 20)),  # only one equal
            ],
            table("clinical_events")
            .filter("code", is_in=make_codelist("Code1"))
            .filter(date="2021-01-01"),
            [
                dict(patient_id=1, code="Code1", date=date(2021, 1, 1), value=10),
            ],
        ),
        (
            [
                patient(1, ctv3_event("Code1", "2021-01-01", 10)),  # before
                patient(2, ctv3_event("Code2", "2021-01-02", 20)),  # start of range
                patient(3, ctv3_event("Code3", "2021-01-03", 30)),  # within range
                patient(4, ctv3_event("Code4", "2021-01-04", 40)),  # end of range
                patient(5, ctv3_event("Code5", "2021-01-05", 50)),  # after
            ],
            table("clinical_events").filter(
                "date", between=["2021-01-02", "2021-01-04"]
            ),
            [
                dict(patient_id=2, code="Code2", date=date(2021, 1, 2), value=20),
                dict(patient_id=3, code="Code3", date=date(2021, 1, 3), value=30),
                dict(patient_id=4, code="Code4", date=date(2021, 1, 4), value=40),
            ],
        ),
        (
            [
                patient(1, ctv3_event("Code1", "2021-01-01", 10)),  # less than
                patient(2, ctv3_event("Code2", "2021-01-02", 20)),  # equal
                patient(3, ctv3_event("Code3", "2021-01-03", 30)),  # greater than
            ],
            table("clinical_events").filter("value", greater_than=20),
            [
                dict(patient_id=3, code="Code3", date=date(2021, 1, 3), value=30),
            ],
        ),
        (
            [
                patient(1, ctv3_event("Code1", "2021-01-01", 10)),  # less than
                patient(2, ctv3_event("Code2", "2021-01-02", 20)),  # equal
                patient(3, ctv3_event("Code3", "2021-01-03", 30)),  # greater than
            ],
            table("clinical_events").filter("date", greater_than="2021-01-02"),
            [
                dict(patient_id=3, code="Code3", date=date(2021, 1, 3), value=30),
            ],
        ),
        (
            [
                patient(1, ctv3_event("Code1", "2021-01-01", 10)),  # less than
                patient(2, ctv3_event("Code2", "2021-01-02", 20)),  # equal
                patient(3, ctv3_event("Code3", "2021-01-03", 30)),  # greater than
            ],
            table("clinical_events").filter(
                "date", greater_than_or_equals="2021-01-02"
            ),
            [
                dict(patient_id=2, code="Code2", date=date(2021, 1, 2), value=20),
                dict(patient_id=3, code="Code3", date=date(2021, 1, 3), value=30),
            ],
        ),
        (
            [
                patient(1, ctv3_event("Code1", "2021-01-01", 10)),  # less than
                patient(2, ctv3_event("Code2", "2021-01-02", 20)),  # equal
                patient(3, ctv3_event("Code3", "2021-01-03", 30)),  # greater than
            ],
            table("clinical_events").filter("date", on_or_after="2021-01-02"),
            [
                dict(patient_id=2, code="Code2", date=date(2021, 1, 2), value=20),
                dict(patient_id=3, code="Code3", date=date(2021, 1, 3), value=30),
            ],
        ),
        (
            [
                patient(1, ctv3_event("Code1", "2021-01-01", 10)),  # less than
                patient(2, ctv3_event("Code2", "2021-01-02", 20)),  # equal
                patient(3, ctv3_event("Code3", "2021-01-03", 30)),  # greater than
            ],
            table("clinical_events").filter("date", less_than="2021-01-02"),
            [
                dict(patient_id=1, code="Code1", date=date(2021, 1, 1), value=10),
            ],
        ),
        (
            [
                patient(1, ctv3_event("Code1", "2021-01-01", 10)),  # less than
                patient(2, ctv3_event("Code2", "2021-01-02", 20)),  # equal
                patient(3, ctv3_event("Code3", "2021-01-03", 30)),  # greater than
            ],
            table("clinical_events").filter("date", less_than_or_equals="2021-01-02"),
            [
                dict(patient_id=1, code="Code1", date=date(2021, 1, 1), value=10),
                dict(patient_id=2, code="Code2", date=date(2021, 1, 2), value=20),
            ],
        ),
        (
            [
                patient(1, ctv3_event("Code1", "2021-01-01", 10)),  # before
                patient(2, ctv3_event("Code2", "2021-01-02", 20)),  # on
                patient(3, ctv3_event("Code3", "2021-01-03", 30)),  # after
            ],
            table("clinical_events").filter("date", on_or_before="2021-01-02"),
            [
                dict(patient_id=1, code="Code1", date=date(2021, 1, 1), value=10),
                dict(patient_id=2, code="Code2", date=date(2021, 1, 2), value=20),
            ],
        ),
        (
            [
                patient(1, ctv3_event("Code1", "2021-01-01", 10)),  # less than
                patient(2, ctv3_event("Code2", "2021-01-02", 20)),  # equal
                patient(3, ctv3_event("Code3", "2021-01-03", 30)),  # greater than
            ],
            table("clinical_events").filter("value", less_than_or_equals=20),
            [
                dict(patient_id=1, code="Code1", date=date(2021, 1, 1), value=10),
                dict(patient_id=2, code="Code2", date=date(2021, 1, 2), value=20),
            ],
        ),
        (
            [
                patient(1, ctv3_event("Code1", "2021-01-01", 10)),  # in
                patient(2, ctv3_event("Code2", "2021-01-02", 20)),  # not in
            ],
            table("clinical_events").filter(
                "code", is_in=make_codelist("Code1", "Code999")
            ),
            [
                dict(patient_id=1, code="Code1", date=date(2021, 1, 1), value=10),
            ],
        ),
        (
            [
                patient(1, ctv3_event("Code1", "2021-01-01", 10)),  # in
                patient(2, ctv3_event("Code2", "2021-01-02", 20)),  # not in
            ],
            table("clinical_events").filter(
                "code", not_in=make_codelist("Code1", "Code999")
            ),
            [
                dict(patient_id=2, code="Code2", date=date(2021, 1, 2), value=20),
            ],
        ),
        (
            [
                patient(
                    1, ctv3_event("Code1", "2021-01-01", 10)
                ),  # excluded by first filter
                patient(
                    2, ctv3_event("Code2", "2021-01-02", 20)
                ),  # excluded by second filter
                patient(
                    3, ctv3_event("Code3", "2021-01-03", 30)
                ),  # excluded by third filter
                patient(4, ctv3_event("Code4", "2021-01-04", 40)),  # included
            ],
            table("clinical_events")
            .filter("value", greater_than=15)
            .filter("date", between=["2021-01-03", "2021-06-06"])
            .filter("code", is_in=make_codelist("Code4")),
            [
                dict(patient_id=4, code="Code4", date=date(2021, 1, 4), value=40),
            ],
        ),
    ],
    ids=[
        "test single equals filter",
        "test multiple equals filter",
        "test between filter",
        "test greater than filter on numeric data",
        "test greater than filter on date data",
        "test greater than or equals filter on date data",
        "test on or after filter (alias for gte)",
        "test less than filter on date data",
        "test less than or equals filter on date data",
        "test on or before filter (alias for lte)",
        "test less than or equals filter on numeric data",
        "test in filter",
        "test not in filter",
        "test multiple chained filters",
    ],
)
def test_simple_filters(engine, data, filtered_table, expected):
    engine.setup(data)

    class Cohort:
        population = filtered_table.exists()
        _filtered = filtered_table.first_by("patient_id")
        code = _filtered.get("code")
        date = _filtered.get("date")
        value = _filtered.get("value")

    assert engine.extract(convert(Cohort)) == expected


@pytest.mark.parametrize("filter_value", [[170, 180], (170, 180), {170, 180}])
def test_is_in_filter(engine, filter_value, request):
    if engine.name == "in_memory":
        # The cohort in this test filters a patient-dimensioned table, which would not
        # be allowed in the new QM.
        pytest.xfail()

    data = [
        patient(1, ctv3_event("Code1", "2021-01-01", 10), height=180),  # in
        patient(2, ctv3_event("Code2", "2021-01-02", 20), height=170),  # not in
    ]

    engine.setup(data)

    class Cohort:
        _filtered_table = table("patients").filter("height", is_in=filter_value)
        population = _filtered_table.exists()
        _filtered = _filtered_table.first_by("patient_id")
        height = _filtered.get("height")

    expected = [
        dict(patient_id=1, height=180),
        dict(patient_id=2, height=170),
    ]
    assert engine.extract(convert(Cohort)) == expected


@pytest.mark.parametrize(
    "filtered_table,expected",
    [
        (
            table("clinical_events").filter(
                "value", greater_than=15, include_null=True
            ),
            [
                dict(patient_id=2, code="Code2", date=date(2021, 1, 2), value=None),
                dict(patient_id=3, code="Code3", date=date(2021, 1, 3), value=30),
            ],
        ),
        (
            table("clinical_events").filter("value", greater_than=15),
            [dict(patient_id=3, code="Code3", date=date(2021, 1, 3), value=30)],
        ),
    ],
)
def test_filter_with_nulls(engine, filtered_table, expected):
    engine.setup(
        [
            patient(
                1, ctv3_event("Code1", "2021-01-01", 10)
            ),  # excluded by filter on value
            patient(
                2, ctv3_event("Code2", "2021-01-02", None)
            ),  # only included if include_null is True
            patient(3, ctv3_event("Code3", "2021-01-03", 30)),  # included
        ]
    )

    class Cohort:
        population = filtered_table.exists()
        _filtered_per_patient = filtered_table.first_by("patient_id")
        code = _filtered_per_patient.get("code")
        date = _filtered_per_patient.get("date")
        value = _filtered_per_patient.get("value")

    assert engine.extract(convert(Cohort)) == expected


def test_filter_between_other_query_values(engine):
    # set up input data for 3 patients, with positive test dates and clinical event results
    input_data = [
        patient(
            1,
            positive_test(True, "2021-01-01"),
            positive_test(True, "2021-02-15"),
            positive_test(True, "2021-03-02"),
            # first=2021-01-01, last=2021-03-02; 2 values between dates, 1 outside
            ctv3_event("Code1", "2021-02-01", system="ctv3", value=10.1),
            ctv3_event("Code1", "2021-04-12", system="ctv3", value=10.2),
            ctv3_event("Code1", "2021-03-01", system="ctv3", value=10.3),
        ),
        patient(
            2,
            positive_test(True, "2021-01-21"),
            positive_test(False, "2021-02-17"),
            positive_test(True, "2021-05-01"),
            # first=2021-01-21, last=2021-05-01, 1 between, 2 outside
            ctv3_event("Code1", "2021-01-10", system="ctv3", value=50.1),
            ctv3_event("Code1", "2021-02-01", system="ctv3", value=50.2),
            ctv3_event("Code1", "2021-05-02", system="ctv3", value=50.3),
        ),
        patient(
            3,
            positive_test(True, "2021-01-10"),
            positive_test(True, "2021-02-23"),
            positive_test(True, "2021-03-01"),
            # first=2021-01-10, last=2021-03-01, none inside
            ctv3_event("Code1", "2021-03-15", system="ctv3", value=60.1),
            ctv3_event("Code1", "2021-04-01", system="ctv3", value=60.2),
            # within dates, but different code
            ctv3_event("Code2", "2021-02-01", system="ctv3", value=60.3),
            # within dates, but different system
            ctv3_event("Code2", "2021-02-01", system="snomed", value=60.3),
        ),
    ]
    engine.setup(input_data)

    # Cohort to extract the last Code1 result between a patient's first and last positive test dates
    class Cohort(OldCohortWithPopulation):
        _positive_tests = table("positive_tests").filter(result=True)
        first_pos = _positive_tests.earliest("test_date").get("test_date")
        last_pos = _positive_tests.latest("test_date").get("test_date")
        _events = (
            table("clinical_events")
            .filter("code", is_in=make_codelist("Code1"))
            .filter("date", between=[first_pos, last_pos])
            .latest()
        )
        date = _events.get("date")
        value = _events.get("value")

    result = engine.extract(convert(Cohort))
    assert result == [
        dict(
            patient_id=1,
            date=date(2021, 3, 1),
            first_pos=date(2021, 1, 1),
            last_pos=date(2021, 3, 2),
            value=10.3,
        ),
        dict(
            patient_id=2,
            date=date(2021, 2, 1),
            first_pos=date(2021, 1, 21),
            last_pos=date(2021, 5, 1),
            value=50.2,
        ),
        dict(
            patient_id=3,
            date=None,
            first_pos=date(2021, 1, 10),
            last_pos=date(2021, 3, 1),
            value=None,
        ),
    ]


def test_date_in_range_filter(engine):
    if engine.name == "in_memory":
        # This test uses count(), whose behaviour will change in the new QM.  (See
        # test_aggregation.)
        pytest.xfail()

    input_data = [
        # (9999-12-31 is the default TPP null value)
        # registraion start date before target date; no end date - included
        RegistrationHistory(
            PatientId=1, StpId="STP1", StartDate="2021-01-02", EndDate="9999-12-31"
        ),
        # registration starts after target date; no end date - not included
        RegistrationHistory(
            PatientId=2, StpId="STP2", StartDate="2021-03-03", EndDate="9999-12-31"
        ),
        # 2 registrations, not overlapping; include the one that contains the target date
        RegistrationHistory(
            PatientId=3, StpId="STP1", StartDate="2021-02-02", EndDate="2021-03-01"
        ),
        RegistrationHistory(
            PatientId=3, StpId="STP2", StartDate="2021-03-01", EndDate="2021-04-01"
        ),
        # registered with 2 STPs overlapping target date; both are included
        RegistrationHistory(
            PatientId=4, StpId="STP2", StartDate="2021-02-02", EndDate="2021-04-01"
        ),
        RegistrationHistory(
            PatientId=4, StpId="STP3", StartDate="2021-01-01", EndDate="2021-03-03"
        ),
    ]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        _registrations = table("practice_registrations").date_in_range("2021-03-02")
        stp = _registrations.first_by("date_start").get("stp")
        count = _registrations.count("patient_id")

    result = engine.extract(convert(Cohort))
    assert result == [
        dict(patient_id=1, stp="STP1", count=1),
        dict(patient_id=2, stp=None, count=None),
        dict(patient_id=3, stp="STP2", count=1),
        dict(patient_id=4, stp="STP3", count=2),
    ]


def test_in_filter_on_query_values(engine):
    if engine.name == "in_memory":
        # Filtering on code filters on both a code's value and system.  The QM should
        # not treat codes specially, and in the new QM, two separate filters should be
        # created.
        pytest.xfail()

    # set up input data for 2 patients, with positive test dates and clinical event results
    input_data = [
        patient(
            1,
            positive_test(True, "2021-01-01"),
            positive_test(True, "2021-02-15"),
            positive_test(True, "2021-03-02"),
            # 2 results with dates matching a positive result: SELECTED
            ctv3_event("Code1", "2021-01-01", value=10.1),
            ctv3_event("Code1", "2021-02-15", value=10.2),
            # 1 result that doesn't match a positive result date
            ctv3_event("Code1", "2021-05-01", value=10.3),
        ),
        patient(
            2,
            positive_test(True, "2021-01-10"),
            positive_test(False, "2021-02-01"),
            positive_test(True, "2021-05-01"),
            # 1 result matches a positive result date: SELECTED
            ctv3_event("Code1", "2021-01-10", value=50.1),
            # 1 matches a negative result date
            ctv3_event("Code1", "2021-02-01", value=50.2),
            # 1 result matches a positive result date but a different code
            ctv3_event("Code2", "2021-05-01", value=50.3),
            # 1 result matches a positive result date but a different system
            ctv3_event("Code1", "2021-05-01", value=50.4, system="snomed"),
        ),
    ]

    engine.setup(input_data)

    # Cohort to extract the Code1 results that were on a positive test date
    class Cohort(OldCohortWithPopulation):
        _positive_test_dates = (
            table("positive_tests").filter(result=True).get("test_date")
        )
        _last_code1_events_on_positive_test_dates = (
            table("clinical_events")
            .filter("code", is_in=make_codelist("Code1"))
            .filter("date", is_in=_positive_test_dates)
            .latest()
        )
        date = _last_code1_events_on_positive_test_dates.get("date")
        value = _last_code1_events_on_positive_test_dates.get("value")

    result = engine.extract(convert(Cohort))
    assert result == [
        dict(patient_id=1, date=date(2021, 2, 15), value=10.2),
        dict(patient_id=2, date=date(2021, 1, 10), value=50.1),
    ]


def test_not_in_filter_on_query_values(engine):
    # set up input data for 2 patients, with positive test dates and clinical event results

    input_data = [
        patient(
            1,
            positive_test(True, "2021-01-01"),
            positive_test(True, "2021-02-15"),
            positive_test(True, "2021-03-02"),
            positive_test(True, "2021-05-01"),
            # 2 results with dates matching a positive result
            ctv3_event("Code1", "2021-01-01", value=10.1),
            ctv3_event("Code1", "2021-02-15", value=10.2),
            # 1 result that doesn't match a positive result date: SELECTED
            ctv3_event("Code1", "2021-04-01", value=10.3),
        ),
        patient(
            2,
            positive_test(True, "2021-01-10"),
            positive_test(False, "2021-02-01"),
            positive_test(True, "2021-05-01"),
            # 1 result matches a positive result date
            ctv3_event("Code1", "2021-01-10", value=50.1),
            # pt2 1 matches a negative result date
            ctv3_event("Code1", "2021-02-01", value=50.2),
            # pt2 1 result doesn't match any result date: SELECTED
            ctv3_event("Code1", "2021-05-02", value=50.3),
        ),
    ]

    engine.setup(input_data)

    # Cohort to extract the results that were NOT on a test date (positive or negative)
    class Cohort(OldCohortWithPopulation):
        _test_dates = table("positive_tests").get("test_date")
        _last_event_not_on_test_date = (
            table("clinical_events").filter("date", not_in=_test_dates).latest()
        )
        date = _last_event_not_on_test_date.get("date")
        value = _last_event_not_on_test_date.get("value")

    result = engine.extract(convert(Cohort))
    assert result == [
        dict(patient_id=1, date=date(2021, 4, 1), value=10.3),
        dict(patient_id=2, date=date(2021, 5, 2), value=50.3),
    ]


@pytest.mark.parametrize(
    "aggregation,column,expected",
    [
        (
            "exists",
            "code",
            [
                dict(patient_id=1, value=True),
                dict(patient_id=2, value=True),
                dict(patient_id=3, value=None),
            ],
        ),
        (
            "count",
            "code",
            [
                dict(patient_id=1, value=2),
                dict(patient_id=2, value=1),
                dict(patient_id=3, value=None),
            ],
        ),
        (
            "sum",
            "value",
            [
                # Due to the usual floating point shennanigans we don't always get
                # _exactly_ the result we're expecting here, depending on the database
                dict(patient_id=1, value=pytest.approx(20.6)),
                dict(patient_id=2, value=pytest.approx(50.1)),
                dict(patient_id=3, value=None),
            ],
        ),
    ],
    ids=[],
)
def test_aggregation(engine, aggregation, column, expected):
    if engine.name == "in_memory":
        if aggregation == "exists":
            # The old QM returns None for patients that do not match a condition, but we
            # want the new QM to return False instead.
            pytest.xfail()
        if aggregation == "count":
            # The old QM returns None for patients that do not match a condition, but we
            # want the new QM to return zero instead.
            pytest.xfail()

    input_data = [
        patient(
            1,
            ctv3_event("Code1", value=10.1),
            ctv3_event("Code1", value=10.5),
            ctv3_event("Code2", value=30.1),
        ),
        patient(
            2,
            ctv3_event("Code1", value=50.1),
            ctv3_event("Code2", value=60.1),
        ),
        patient(
            3,
            ctv3_event("Code2", value=70.1),
        ),
    ]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        _filtered_table = table("clinical_events").filter(
            "code", is_in=make_codelist("Code1")
        )
        value = getattr(_filtered_table, aggregation)(column)

    assert engine.extract(convert(Cohort)) == expected


def test_categorise_simple_comparisons(engine):
    input_data = [patient(1, height=180), patient(2, height=200.5), patient(3)]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        _height = table("patients").first_by("patient_id").get("height")
        _height_categories = {
            "tall": _height > 190,
            "short": _height <= 190,
        }
        height_group = categorise(_height_categories, default="missing")

    result = engine.extract(convert(Cohort))
    assert result == [
        dict(patient_id=1, height_group="short"),
        dict(patient_id=2, height_group="tall"),
        dict(patient_id=3, height_group="missing"),
    ]


@pytest.mark.parametrize(
    "categories,default,expected",
    [
        (
            lambda height_value: {
                "tall": height_value > 190,
                "medium": (height_value <= 190) & (height_value > 150),
                "short": height_value < 150,
            },
            "missing",
            [
                dict(patient_id=1, height_group="medium"),
                dict(patient_id=2, height_group="tall"),
                dict(patient_id=3, height_group="missing"),
                dict(patient_id=4, height_group="short"),
            ],
        ),
        (
            lambda height_value: {
                "short_or_tall": (height_value < 150) | (height_value > 190)
            },
            "medium",
            [
                dict(patient_id=1, height_group="medium"),
                dict(patient_id=2, height_group="short_or_tall"),
                dict(patient_id=3, height_group="medium"),
                dict(patient_id=4, height_group="short_or_tall"),
            ],
        ),
        (
            lambda height_value: {
                "short_or_tall": (height_value < 150) | (height_value > 190)
            },
            None,
            [
                dict(patient_id=1, height_group=None),
                dict(patient_id=2, height_group="short_or_tall"),
                dict(patient_id=3, height_group=None),
                dict(patient_id=4, height_group="short_or_tall"),
            ],
        ),
        (
            lambda height_value: {
                "tallish": (height_value > 175) & (height_value != 180),
                "short": height_value <= 175,
            },
            "missing",
            [
                dict(patient_id=1, height_group="missing"),
                dict(patient_id=2, height_group="tallish"),
                dict(patient_id=3, height_group="missing"),
                dict(patient_id=4, height_group="short"),
            ],
        ),
    ],
    ids=[
        "test simple and on two conditions",
        "test simple or on two conditions",
        "test simple or with None default",
        "test a not-equals condition",
    ],
)
def test_categorise_single_combined_conditions(engine, categories, default, expected):
    input_data = [
        patient(1, height=180),
        patient(2, height=200.5),
        patient(3),
        patient(4, height=145),
    ]
    engine.setup(input_data)

    default_kwarg = {"default": default} if default is not None else {}

    class Cohort(OldCohortWithPopulation):
        _height = table("patients").first_by("patient_id").get("height")
        _height_categories = categories(_height)
        height_group = categorise(_height_categories, **default_kwarg)

    result = list(engine.extract(convert(Cohort)))
    assert result == expected


def test_categorise_multiple_values(engine):
    """Test that categories can combine conditions that use different source values"""
    input_data = [
        patient(1, ctv3_event("abc"), height=200),
        patient(2, ctv3_event("xyz"), height=150),
        patient(3, ctv3_event("abc"), height=160),
    ]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        _height = table("patients").first_by("patient_id").get("height")
        _code = table("clinical_events").first_by("patient_id").get("code")
        _height_with_codes_categories = {
            "short": (_height < 190) & (_code == "abc"),
            "tall": (_height > 190) & (_code == "abc"),
        }
        height_group = categorise(_height_with_codes_categories, default="missing")

    result = engine.extract(convert(Cohort))
    assert result == [
        dict(patient_id=1, height_group="tall"),
        dict(patient_id=2, height_group="missing"),
        dict(patient_id=3, height_group="short"),
    ]


def test_categorise_nested_comparisons(engine):
    input_data = [
        patient(1, ctv3_event("abc"), height=194),  # tall with code - matches
        patient(2, ctv3_event("xyz"), height=200.5),  # tall no code  - matches
        patient(3, ctv3_event("abc"), height=140.5),  # short with code - matches
        patient(4, height=140.5),  # short no code
        patient(5),
    ]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        _height = table("patients").first_by("patient_id").get("height")
        _code = table("clinical_events").first_by("patient_id").get("code")

        # make sure the parentheses precedence is followed; these two expressions are equivalent
        _height_with_codes_categories = {
            "tall_or_code": (_height > 190) | ((_height < 150) & (_code == "abc")),
        }
        _codes_with_height_categories = {
            "code_or_tall": ((_height < 150) & (_code == "abc")) | (_height > 190),
        }
        height_group = categorise(_height_with_codes_categories, default="na")
        height_group1 = categorise(_codes_with_height_categories, default="na")

    result = engine.extract(convert(Cohort))

    assert result == [
        dict(patient_id=1, height_group="tall_or_code", height_group1="code_or_tall"),
        dict(patient_id=2, height_group="tall_or_code", height_group1="code_or_tall"),
        dict(patient_id=3, height_group="tall_or_code", height_group1="code_or_tall"),
        dict(patient_id=4, height_group="na", height_group1="na"),
        dict(patient_id=5, height_group="na", height_group1="na"),
    ]


def test_categorise_on_truthiness(engine):
    """Test truthiness of a Value from an exists aggregation"""

    if engine.name == "in_memory":
        # This test uses exists(), whose behaviour will change in the new QM.  (See
        # test_aggregation.)
        pytest.xfail()

    input_data = [
        patient(1, ctv3_event("abc")),
        patient(2, ctv3_event("xyz")),
        patient(3, ctv3_event("abc")),
        patient(4),
    ]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        _code = (
            table("clinical_events").filter("code", is_in=make_codelist("abc")).exists()
        )
        _codes_categories = {"yes": _code}
        abc = categorise(_codes_categories, default="na")

    result = engine.extract(convert(Cohort))
    assert result == [
        dict(patient_id=1, abc="yes"),
        dict(patient_id=2, abc="na"),
        dict(patient_id=3, abc="yes"),
        dict(patient_id=4, abc="na"),
    ]


def test_categorise_on_truthiness_from_filter(engine):
    """Test truthiness of a Value from a filtered value"""

    if engine.name == "in_memory":
        # The old QM wraps each value in a categorise() in a __ne__ comparison against
        # None.  The new QM expects each value to be a boolean series.
        pytest.xfail()

    input_data = [
        patient(1, ctv3_event("abc")),
        patient(2, ctv3_event("xyz")),
        patient(3, ctv3_event("abc")),
        patient(4, ctv3_event("def")),
    ]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        _code = (
            table("clinical_events")
            .filter("code", is_in=make_codelist("abc", "def"))
            .latest()
            .get("code")
        )
        _codes_categories = {"yes": _code}
        has_code = categorise(_codes_categories, default="na")

    result = engine.extract(convert(Cohort))
    assert result == [
        dict(patient_id=1, has_code="yes"),
        dict(patient_id=2, has_code="na"),
        dict(patient_id=3, has_code="yes"),
        dict(patient_id=4, has_code="yes"),
    ]


def test_categorise_multiple_truthiness_values(engine):
    """Test truthiness of a Value from a filtered value"""

    if engine.name == "in_memory":
        # This test uses exists(), whose behaviour will change in the new QM.  (See
        # test_aggregation.)
        pytest.xfail()

    input_data = [
        patient(1, ctv3_event("abc"), positive_test(True)),
        patient(2, ctv3_event("xyz"), positive_test(False)),
        patient(3, ctv3_event("abc"), positive_test(False)),
        patient(4, ctv3_event("def"), positive_test(True)),
    ]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        _code = (
            table("clinical_events")
            .filter("code", is_in=make_codelist("abc", "def"))
            .latest()
            .get("code")
        )
        _has_positive_test = table("positive_tests").filter(result=True).exists()
        _codes_categories = {"yes": _code & _has_positive_test}
        has_positive_code = categorise(_codes_categories, default="na")

    result = engine.extract(convert(Cohort))
    assert result == [
        dict(patient_id=1, has_positive_code="yes"),
        dict(patient_id=2, has_positive_code="na"),
        dict(patient_id=3, has_positive_code="na"),
        dict(patient_id=4, has_positive_code="yes"),
    ]


def test_categorise_invert(engine):
    input_data = [
        patient(1, height=194),
        patient(2, height=160.5),
        patient(3, height=155.5),
        patient(4, height=140.5),
        RegistrationHistory(PatientId=5),
    ]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        _height = table("patients").first_by("patient_id").get("height")
        _code = table("clinical_events").first_by("patient_id").get("code")

        # make sure the parentheses precedence is followed; these two expressions are equivalent
        _height_inverted = {
            "tall": _height > 190,
            "not_tall": ~(_height > 190),
        }
        height_group = categorise(_height_inverted, default="na")

    result = engine.extract(convert(Cohort))

    assert result == [
        dict(patient_id=1, height_group="tall"),
        dict(patient_id=2, height_group="not_tall"),
        dict(patient_id=3, height_group="not_tall"),
        dict(patient_id=4, height_group="not_tall"),
        dict(patient_id=5, height_group="na"),
    ]


def test_categorise_invert_truthiness_values(engine):
    if engine.name == "in_memory":
        # This test uses exists(), whose behaviour will change in the new QM.  (See
        # test_aggregation.)
        pytest.xfail()

    input_data = [
        patient(1, ctv3_event("abc")),
        patient(2, ctv3_event("xyz")),
        patient(3, ctv3_event("abc")),
        patient(4, ctv3_event("def")),
    ]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        _code = (
            table("clinical_events")
            .filter("code", is_in=make_codelist("abc", "def"))
            .latest()
            .get("code")
        )
        _codes_categories = {"yes": _code, "no": ~_code}
        has_code = categorise(_codes_categories, default="na")

    result = engine.extract(convert(Cohort))
    assert result == [
        dict(patient_id=1, has_code="yes"),
        dict(patient_id=2, has_code="no"),
        dict(patient_id=3, has_code="yes"),
        dict(patient_id=4, has_code="yes"),
    ]


def test_categorise_invert_combined_values(engine):
    if engine.name == "in_memory":
        # This test uses exists(), whose behaviour will change in the new QM.  (See
        # test_aggregation.)
        pytest.xfail()

    input_data = [
        patient(1, ctv3_event("abc"), positive_test(True)),
        patient(2, ctv3_event("xyz"), positive_test(False)),
        patient(3, ctv3_event("abc"), positive_test(False)),
        patient(4, ctv3_event("def"), positive_test(True)),
    ]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        _code = (
            table("clinical_events")
            .filter("code", is_in=make_codelist("abc", "def"))
            .latest()
            .get("code")
        )
        _has_positive_test = table("positive_tests").filter(result=True).exists()
        _codes_categories = {"neg_or_no_code": ~(_code & _has_positive_test)}
        result_group = categorise(_codes_categories, default="pos")

    result = engine.extract(convert(Cohort))
    assert result == [
        dict(patient_id=1, result_group="pos"),
        dict(patient_id=2, result_group="neg_or_no_code"),
        dict(patient_id=3, result_group="neg_or_no_code"),
        dict(patient_id=4, result_group="pos"),
    ]


def test_categorise_double_invert(engine):
    if engine.name == "in_memory":
        # The old QM wraps each value in a categorise() in a __ne__ comparison against
        # None.  The new QM expects each value to be a boolean series.
        pytest.xfail()

    input_data = [
        patient(1, ctv3_event("abc")),
        patient(2, ctv3_event("xyz")),
        patient(3, ctv3_event("abc")),
        patient(4, ctv3_event("def")),
    ]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        _code = (
            table("clinical_events")
            .filter("code", is_in=make_codelist("abc", "def"))
            .latest()
            .get("code")
        )
        _codes_categories = {"yes": ~~_code}
        has_code = categorise(_codes_categories, default="na")

    result = engine.extract(convert(Cohort))
    assert result == [
        dict(patient_id=1, has_code="yes"),
        dict(patient_id=2, has_code="na"),
        dict(patient_id=3, has_code="yes"),
        dict(patient_id=4, has_code="yes"),
    ]


def test_categorise_multiple_truthiness_categories(engine):
    """
    Test categorisation on multiple truthy values
    This tests for a previous bug in sorting the reference nodes in category definitions.
    A truthy categorisation ({`x`: _codes}) creates a Comparator instance with a ValueFromRow
    as its LHS and tests for not-equal to None. In the query engine, we find all reference
    nodes from category definitions and sort them to ensure a consistent order (largely for
    tests) using the node column and source as a sort key,  The reference node for a
    ValueFromRow is a Row, which can't be sorted.  THis test checks the workaround for
    this scenario.
    """

    if engine.name == "in_memory":
        # The old QM wraps each value in a categorise() in a __ne__ comparison against
        # None.  The new QM expects each value to be a boolean series.
        pytest.xfail()

    input_data = [
        patient(1, ctv3_event("abc")),
        patient(2, ctv3_event("xyz"), ctv3_event("lmn")),
        patient(3, ctv3_event("uvw")),
        patient(4, ctv3_event("def")),
        patient(5),
    ]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        _codes_1 = (
            table("clinical_events")
            .filter("code", is_in=make_codelist("abc", "def"))
            .latest()
            .get("code")
        )
        _codes_2 = (
            table("clinical_events")
            .filter("code", is_in=make_codelist("uvw", "xyz"))
            .latest()
            .get("code")
        )
        _codes_categories = {"1": _codes_1, "2": _codes_2}
        has_positive_code = categorise(_codes_categories, default="na")

    result = engine.extract(convert(Cohort))
    assert result == [
        dict(patient_id=1, has_positive_code="1"),
        dict(patient_id=2, has_positive_code="2"),
        dict(patient_id=3, has_positive_code="2"),
        dict(patient_id=4, has_positive_code="1"),
        dict(patient_id=5, has_positive_code="na"),
    ]


def test_age_as_of(engine):
    input_data = [
        patient(1, ctv3_event("abc", "2020-10-01"), dob="1990-08-10"),
        patient(2, ctv3_event("abc", "2018-02-01"), dob="2000-03-20"),
    ]
    engine.setup(input_data)

    class Cohort(OldCohortWithPopulation):
        age_in_2010 = table("patients").age_as_of("2010-06-01")
        age_at_last_event = table("patients").age_as_of(
            table("clinical_events").latest().get("date")
        )

    result = engine.extract(convert(Cohort))
    assert result == [
        {"patient_id": 1, "age_in_2010": 19, "age_at_last_event": 30},
        {"patient_id": 2, "age_in_2010": 10, "age_at_last_event": 17},
    ]


def test_fetching_results_using_temporary_database(engine):
    if engine.name == "spark":
        pytest.skip("Spark does not support 'fetch using temporary database'")

    engine.setup(
        [
            patient(1, ctv3_event("abc", "2020-01-01")),
            patient(2, ctv3_event("xyz", "2020-01-01")),
        ]
    )

    class Cohort(OldCohortWithPopulation):
        code = table("clinical_events").latest().get("code")

    assert engine.extract(convert(Cohort), temporary_database="temp_tables") == [
        dict(patient_id=1, code="abc"),
        dict(patient_id=2, code="xyz"),
    ]
