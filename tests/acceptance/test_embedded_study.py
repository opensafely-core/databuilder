from datetime import datetime

import pytest

from databuilder.__main__ import main
from databuilder.validate_dummy_data import ValidationError
from tests.lib.fixtures import (
    invalid_dataset_attribute_dataset_definition,
    invalid_dataset_definition,
    no_dataset_attribute_dataset_definition,
    trivial_dataset_definition,
)
from tests.lib.tpp_schema import patient


# Possibly we can merge these features into the main Study fixture if we ever end up
# needing them anywhere else
class DummyDataStudy:
    def __init__(self, workspace, dataset_definition, dummy_data):
        self.dataset_path = workspace / "dataset.csv"
        self.dataset_definition_path = workspace / "dataset.py"
        self.dummy_data_path = workspace / "dummy-data.csv"
        self.dataset_definition_path.write_text(dataset_definition)
        self.dummy_data_path.write_text(dummy_data)

    def generate_dataset(self):
        main(
            [
                "generate-dataset",
                "--dataset-definition",
                str(self.dataset_definition_path),
                "--output",
                str(self.dataset_path),
                "--dummy-data-file",
                str(self.dummy_data_path),
            ]
        )


def test_generate_dataset(study, mssql_database):
    mssql_database.setup(
        patient(dob=datetime(1943, 5, 5)),
        patient(dob=datetime(1999, 5, 5)),
    )

    study.setup_from_string(trivial_dataset_definition)
    study.generate(mssql_database, "databuilder.backends.tpp.TPPBackend")
    results = study.results()

    assert len(results) == 2
    assert {r["year"] for r in results} == {"1943", "1999"}


def test_dump_dataset_sql_happy_path(study, mssql_database):
    study.setup_from_string(trivial_dataset_definition)
    study.dump_dataset_sql()


def test_dump_dataset_sql_error_path(study, mssql_database):
    study.setup_from_string(invalid_dataset_definition)
    with pytest.raises(SystemExit):
        study.dump_dataset_sql()


def test_dump_dataset_sql_with_no_dataset_attribute(study, mssql_database):
    study.setup_from_string(no_dataset_attribute_dataset_definition)
    with pytest.raises(
        AttributeError, match="A dataset definition must define one 'dataset'"
    ):
        study.dump_dataset_sql()


def test_dump_dataset_sql_attribute_invalid(study, mssql_database):
    study.setup_from_string(invalid_dataset_attribute_dataset_definition)
    with pytest.raises(
        AssertionError,
        match="'dataset' must be an instance of databuilder.query_language.Dataset()",
    ):
        study.dump_dataset_sql()


def test_validate_dummy_data_happy_path(tmp_path):
    dummy_data = "patient_id,year\n1,1971\n2,1992"
    study = DummyDataStudy(tmp_path, trivial_dataset_definition, dummy_data)
    study.generate_dataset()
    assert study.dataset_path.read_text() == dummy_data


def test_validate_dummy_data_error_path(tmp_path):
    dummy_data = "patient_id,year\n1,1971\n2,foo"
    study = DummyDataStudy(tmp_path, trivial_dataset_definition, dummy_data)
    with pytest.raises(ValidationError, match="Invalid int"):
        study.generate_dataset()
