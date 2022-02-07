from databuilder.dsl import IdColumn
from databuilder.query_language import Dataset, DateColumn, PatientTable

# This table definition is here as a convenience.
# TODO: instantiate tables with a contract, and import tables from tests.lib.tables
from databuilder.query_model import Function, SelectColumn, SelectPatientTable, Value


class Patients(PatientTable):
    __name__ = "patients"

    patient_id = IdColumn("patient_id")
    date_of_birth = DateColumn("date_of_birth")


patients = Patients()


def test_simple_dataset() -> None:
    year_of_birth = patients.date_of_birth.year
    dataset = Dataset()
    dataset.set_population(year_of_birth <= 2000)  # TODO: why does this not typecheck?
    dataset.year_of_birth = year_of_birth

    assert dataset.compile() == {
        "year_of_birth": Function.YearFromDate(
            source=SelectColumn(
                name="date_of_birth", source=SelectPatientTable("patients")
            )
        ),
        "population": Function.LE(
            lhs=Function.YearFromDate(
                source=SelectColumn(
                    name="date_of_birth", source=SelectPatientTable("patients")
                )
            ),
            rhs=Value(2000),
        ),
    }


def test_get_column_from_patient_table():
    assert patients.date_of_birth.qm_node == SelectColumn(
        name="date_of_birth", source=SelectPatientTable("patients")
    )


def test_year_comparison():
    assert (patients.date_of_birth.year <= 2000).qm_node == Function.LE(
        lhs=Function.YearFromDate(
            SelectColumn(name="date_of_birth", source=SelectPatientTable("patients"))
        ),
        rhs=Value(2000),
    )
