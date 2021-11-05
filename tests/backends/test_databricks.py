import random
from datetime import date

import pytest

from cohortextractor import codelist, table
from cohortextractor.backends.databricks import DatabricksBackend
from tests.lib.databricks_schema import HESApc, HESApcOtr, MPSHESApc, PCareMeds
from tests.lib.util import extract, iter_flatten


# Keep things deterministic
rand = random.Random(20211019)


@pytest.fixture
def setup_databricks_database(setup_spark_database):
    return lambda *data: setup_spark_database(*data, backend="databricks")


def patient(patient_id, date_of_birth="1980-01-01", *entities):
    if not entities:
        entities = [PCareMeds()]
    else:
        entities = list(iter_flatten(entities))
    for entity in entities:
        if isinstance(entity, PCareMeds):
            entity.Person_ID = patient_id
            entity.PatientDoB = date_of_birth
        if isinstance(entity, MPSHESApc):
            entity.PERSON_ID = patient_id
    return entities


def prescription(date=None, code=None, **kwargs):
    return PCareMeds(ProcessingPeriodDate=date, PrescribeddmdCode=code, **kwargs)


def admission(
    date,
    primary_diagnosis,
    method=None,
    episode_id=None,
    finished=None,
    spell_id=None,
):
    if episode_id is None:
        episode_id = rand.randint(1, 999999999)
    if spell_id is None:
        spell_id = rand.randint(1, 999999999)
    apc = HESApc(
        EPIKEY=episode_id,
        ADMIDATE=date,
        DIAG_4_01=primary_diagnosis,
        ADMINMETH=method,
        FAE=1 if finished else 0,
    )
    mps = MPSHESApc(EPIKEY=episode_id)
    otr = HESApcOtr(EPIKEY=episode_id, SUSSPELLID=spell_id)
    return [apc, mps, otr]


@pytest.mark.integration
def test_basic_databricks_study_definition(spark_database, setup_databricks_database):
    setup_databricks_database(
        patient(
            10,
            "1950-08-20",
            prescription(date="2018-01-01", code="0010"),
            prescription(date="2019-01-01", code="0020"),
            admission(
                date="2020-01-25", primary_diagnosis="N05", method=21, finished=True
            ),
        ),
        patient(
            15,
            "1955-07-17",
            prescription(date="2020-01-12", code="0050"),
        ),
    )

    class Cohort:
        population = table("patients").exists()
        dob = table("patients").first_by("patient_id").get("date_of_birth")
        prescribed_med = (
            table("prescriptions")
            .filter("processing_date", between=["2020-01-01", "2020-01-31"])
            .filter(
                "prescribed_dmd_code", is_in=codelist(["0010", "0050"], system="dmd")
            )
            .exists()
        )
        admitted = (
            table("hospital_admissions")
            .filter("admission_date", between=["2020-01-01", "2020-01-31"])
            .filter(primary_diagnosis="N05", episode_is_finished=True)
            .filter("admission_method", between=[20, 29])
            .exists()
        )

    results = extract(Cohort, DatabricksBackend, spark_database)
    # We don't care exactly what order the patients come back in
    results.sort(key=lambda i: i.get("patient_id"))

    assert results == [
        dict(patient_id=10, dob=date(1950, 8, 20), prescribed_med=None, admitted=True),
        dict(patient_id=15, dob=date(1955, 7, 17), prescribed_med=True, admitted=None),
    ]
