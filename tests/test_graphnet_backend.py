from datetime import date, datetime

import pytest
from lib.graphnet_schema import (
    ClinicalEvents,
    CovidTestResults,
    Patients,
    PracticeRegistrations,
    hospitalization,
    patient,
    patient_address,
    registration,
)
from lib.util import extract

from cohortextractor import table
from cohortextractor.backends.graphnet import GraphnetBackend


@pytest.mark.integration
def test_basic_events_and_registration(database, setup_backend_database):
    setup_backend_database(
        Patients(Patient_ID=1),
        PracticeRegistrations(Patient_ID=1),
        ClinicalEvents(Patient_ID=1, CTV3Code="Code1"),
        backend="graphnet",
    )

    class Cohort:
        code = table("clinical_events").first_by("patient_id").get("code")

    assert extract(Cohort, GraphnetBackend, database) == [
        dict(patient_id=1, code="Code1")
    ]


@pytest.mark.integration
def test_registration_dates(database, setup_backend_database):
    setup_backend_database(
        Patients(Patient_ID=1),
        PracticeRegistrations(
            Patient_ID=1, StartDate="2001-01-01", EndDate="2012-12-12"
        ),
        PracticeRegistrations(Patient_ID=1, StartDate="2013-01-01"),
        backend="graphnet",
    )

    class Cohort:
        _registrations = table("practice_registrations").first_by("patient_id")
        arrived = _registrations.get("date_start")
        left = _registrations.get("date_end")

    assert extract(Cohort, GraphnetBackend, database) == [
        dict(patient_id=1, arrived=datetime(2001, 1, 1), left=datetime(2012, 12, 12))
    ]


@pytest.mark.xfail
def test_registration_dates_no_end(database, setup_backend_database):
    setup_backend_database(
        Patients(Patient_ID=1),
        PracticeRegistrations(
            Patient_ID=1, StartDate="2011-01-01", EndDate="2012-12-31"
        ),
        PracticeRegistrations(
            Patient_ID=1, StartDate="2013-01-01", EndDate=None
        ),
        backend="graphnet",
    )

    class Cohort:
        _registrations = table("practice_registrations").date_in_range("2014-01-01")
        arrived = _registrations.get("date_start")
        left = _registrations.get("date_end")

    assert extract(Cohort, GraphnetBackend, database) == [
        dict(patient_id=1, arrived=datetime(2013, 1, 1), left=None)
    ]


@pytest.mark.integration
def test_covid_test_positive_result(database, setup_backend_database):
    setup_backend_database(
        Patients(Patient_ID=1),
        PracticeRegistrations(
            Patient_ID=1, StartDate="2001-01-01", EndDate="2026-06-26"
        ),
        CovidTestResults(
            Patient_ID=1,
            SpecimenDate="2020-05-05",
            positive_result=True,
        ),
        backend="graphnet",
    )

    class Cohort:
        date = (
            table("covid_test_results")
            .filter(positive_result=True)
            .earliest()
            .get("date")
        )

    assert extract(Cohort, GraphnetBackend, database) == [
        dict(patient_id=1, date=date(2020, 5, 5))
    ]


@pytest.mark.integration
def test_covid_test_negative_result(database, setup_backend_database):
    setup_backend_database(
        Patients(Patient_ID=1),
        PracticeRegistrations(
            Patient_ID=1, StartDate="2001-01-01", EndDate="2026-06-26"
        ),
        CovidTestResults(
            Patient_ID=1,
            SpecimenDate="2020-05-05",
            positive_result=False,
        ),
        backend="graphnet",
    )

    class Cohort:
        date = (
            table("covid_test_results")
            .filter(positive_result=False)
            .earliest()
            .get("date")
        )

    assert extract(Cohort, GraphnetBackend, database) == [
        dict(patient_id=1, date=date(2020, 5, 5))
    ]


@pytest.mark.integration
def test_patients_table(database, setup_backend_database):
    setup_backend_database(
        Patients(Patient_ID=1, Sex="F", DateOfBirth="1950-01-01"),
        PracticeRegistrations(
            Patient_ID=1, StartDate="2001-01-01", EndDate="2026-06-26"
        ),
        backend="graphnet",
    )

    class Cohort:
        _patients = table("patients").first_by("patient_id")
        sex = _patients.get("sex")
        dob = _patients.get("date_of_birth")

    assert extract(Cohort, GraphnetBackend, database) == [
        dict(patient_id=1, sex="F", dob=date(1950, 1, 1))
    ]


@pytest.mark.integration
def test_hospitalization_table_returns_admission_date_and_code(
    database, setup_backend_database
):
    setup_backend_database(
        *patient(
            1,
            "M",
            "1990-1-1",
            registration("2001-01-01", "2026-06-26"),
            hospitalization(admit_date="2020-12-12", code="xyz"),
        ),
        backend="graphnet",
    )

    class Cohort:
        _hospitalization = table("hospitalizations").first_by("patient_id")
        admission = _hospitalization.get("date")
        code = _hospitalization.get("code")

    assert extract(Cohort, GraphnetBackend, database) == [
        dict(patient_id=1, admission=date(2020, 12, 12), code="xyz")
    ]


""" Removing hospitalization tests - code parsing doesn't occur in the graphnet backend, so these won't work
@pytest.mark.parametrize(
    "raw, codes",
    [
        ("flim", ["flim"]),
        ("flim ,flam ,flum", ["flim", "flam", "flum"]),
        ("flim ||flam ||flum", ["flim", "flam", "flum"]),
        ("abc ,def ||ghi ,jkl", ["abc", "def", "ghi", "jkl"]),
        ("ABCX ,XYZ ,OXO", ["ABC", "XYZ", "OXO"]),
    ],
    ids=[
        "returns a single code",
        "returns multiple space comma separated codes",
        "returns multiple space double pipe separated codes",
        "copes with comma pipe combinations",
        "strips just trailing xs",
    ],
)
@pytest.mark.integration
def test_hospitalization_table_code_conversion(
    database, setup_backend_database, raw, codes
):
    setup_backend_database(
        *patient(
            1,
            "M",
            "1990-1-1",
            registration("2001-01-01", "2026-06-26"),
            hospitalization(code=raw),
        ),
        backend="graphnet"
    )

    query = GraphnetBackend.hospitalizations.get_query(type_map=None)

    results = list(run_query(database, query))

    # Because of the way that we split the raw codes, the order in which they are returned is not the same as the order
    # they appear in the table.
    assert len(results) == len(codes)
    for code in codes:
        assert (1, date(2012, 12, 12), code) in results


def run_query(database, query):
    with database.engine().connect() as cursor:
        yield from cursor.execute(query)


@pytest.mark.integration
def test_hospitalization_code_parsing_works_with_filters(
    database, setup_backend_database
):
    setup_backend_database(
        *patient(
            1,
            "X",
            "1990-1-1",
            registration("2001-01-01", "2026-06-26"),
            hospitalization(code="abc"),
        ),
        *patient(
            2,
            "X",
            "1990-1-1",
            registration("2001-01-01", "2026-06-26"),
            hospitalization(code="xyz"),
        ),
        backend="graphnet"
    )

    class Cohort:
        code = (
            table("hospitalizations")
            # .filter("code", is_in=codelist(["xyz"], system="ctv3"))
            .filter(code="xyz")
            .first_by("patient_id")
            .get("code")
        )

    assert extract(Cohort, GraphnetBackend, database) == [
        dict(patient_id=1, code=None),
        dict(patient_id=2, code="xyz"),
    ]
"""


@pytest.mark.integration
def test_events_with_numeric_value(database, setup_backend_database):
    setup_backend_database(
        Patients(Patient_ID=1),
        PracticeRegistrations(Patient_ID=1),
        ClinicalEvents(Patient_ID=1, CTV3Code="Code1", NumericValue=34.7),
        backend="graphnet",
    )

    class Cohort:
        value = table("clinical_events").latest().get("numeric_value")

    assert extract(Cohort, GraphnetBackend, database) == [
        dict(patient_id=1, value=34.7)
    ]


@pytest.mark.integration
def test_organisation(database, setup_backend_database):
    setup_backend_database(
        # Organisation not a separate table, so will just move detail to registration record
        # organisation(1, "South"),
        # organisation(2, "North"),
        *patient(
            1,
            "M",
            "1990-1-1",
            registration("2001-01-01", "2021-06-26", "A83010", "North East"),
        ),
        *patient(
            2,
            "F",
            "1990-1-1",
            registration("2001-01-01", "2026-06-26", "J82031", "South West"),
        ),
        backend="graphnet",
    )

    class Cohort:
        _registrations = table("practice_registrations").last_by("patient_id")
        region = _registrations.get("nuts1_region_name")
        practice_id = _registrations.get("pseudo_id")

    assert extract(Cohort, GraphnetBackend, database) == [
        dict(patient_id=1, region="North East", practice_id="A83010"),
        dict(patient_id=2, region="South West", practice_id="J82031"),
    ]


@pytest.mark.integration
def test_organisation_dates(database, setup_backend_database):
    setup_backend_database(
        # Organisation not a separate table, so will just move detail to registration record
        # organisation(1, "South"),
        # organisation(2, "North"),
        # organisation(3, "West"),
        # organisation(4, "East"),
        # registered at 2 practices, select the one active on 25/6
        *patient(
            1,
            "M",
            "1990-1-1",
            registration("2001-01-01", "2021-06-26", "A83010", "North East"),
            registration("2021-06-27", "2026-06-26", "J26003", "South West"),
        ),
        # registered at 2 practices with overlapping dates, select the latest
        *patient(
            2,
            "F",
            "1990-1-1",
            registration("2001-01-01", "2026-06-26", "S21021", "East"),
            registration("2021-01-01", "9999-12-31", "S33001", "East"),
        ),
        # registration not in range, not included
        *patient(
            3,
            "F",
            "1990-1-1",
            registration("2001-01-01", "2020-06-26", "S21021", "East"),
        ),
        backend="graphnet",
    )

    class Cohort:
        _registrations = table("practice_registrations").date_in_range("2021-06-25")
        population = _registrations.exists()
        _registration_table = _registrations.latest("date_end")
        region = _registration_table.get("nuts1_region_name")
        practice_id = _registration_table.get("pseudo_id")

    assert extract(Cohort, GraphnetBackend, database) == [
        dict(patient_id=1, region="North East", practice_id="A83010"),
        dict(patient_id=2, region="East", practice_id="S33001"),
    ]


@pytest.mark.integration
def test_index_of_multiple_deprivation(database, setup_backend_database):
    setup_backend_database(
        *patient(
            1,
            "M",
            "1990-1-1",
            registration("2001-01-01", "2026-06-26"),
            patient_address("2001-01-01", "2026-06-26", 1200, "E02000001", True),
        ),
        backend="graphnet",
    )

    class Cohort:
        imd = table("patient_address").imd_rounded_as_of("2021-06-01")

    assert extract(Cohort, GraphnetBackend, database) == [dict(patient_id=1, imd=1200)]


@pytest.mark.integration
@pytest.mark.parametrize(
    "patient_addresses,expected",
    [
        # two addresses recorded as current, choose the latest start date
        (
            [
                patient_address("2001-01-01", "9999-12-31", 100, "E02000002", True),
                patient_address("2021-01-01", "9999-12-31", 200, "E02000003", True),
            ],
            200,
        ),
        # two addresses with same start, choose the latest end date
        (
            [
                patient_address("2001-01-01", "9999-12-31", 300, "E02000003", True),
                patient_address("2001-01-01", "2021-01-01", 200, "E02000002", True),
            ],
            300,
        ),
        # same dates, prefer the one with a postcode
        (
            [
                patient_address("2001-01-01", "9999-12-31", 300, "E02000003", True),
                patient_address("2001-01-01", "9999-12-31", 400, "NPC", False),
            ],
            300,
        ),
        # same dates and both have postcodes, select latest patient address id as tie-breaker
        (
            [
                patient_address("2001-01-01", "9999-12-31", 300, "E02000003", True),
                patient_address("2001-01-01", "9999-12-31", 400, "E02000003", True),
                patient_address("2001-01-01", "9999-12-31", 500, "E02000003", True),
            ],
            500,
        ),
    ],
)
def test_index_of_multiple_deprivation_sorting(
    database, setup_backend_database, patient_addresses, expected
):
    setup_backend_database(
        *patient(
            1,
            "M",
            "1990-1-1",
            registration("2001-01-01", "2026-06-26"),
            *patient_addresses,
        ),
        backend="graphnet",
    )

    class Cohort:
        imd = table("patient_address").imd_rounded_as_of("2021-06-01")

    assert extract(Cohort, GraphnetBackend, database) == [
        dict(patient_id=1, imd=expected)
    ]