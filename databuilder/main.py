import csv
import importlib.util
import shutil
import sys
from contextlib import contextmanager

import structlog

from . import query_language as ql
from .backends import BACKENDS
from .definition.base import dataset_registry
from .validate_dummy_data import validate_dummy_data_file, validate_file_types_match

log = structlog.getLogger()


def generate_dataset(
    definition_file,
    dataset_file,
    backend_id,
    db_url,
    temporary_database,
):
    log.info(f"Generating dataset for {str(definition_file)}")

    dataset_definition = load_definition(definition_file)
    backend = BACKENDS[backend_id]()
    query_engine = backend.query_engine_class(
        db_url, backend, temporary_database=temporary_database
    )
    backend.validate_contracts()
    results = extract(dataset_definition, query_engine)

    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    write_dataset(results, dataset_file)


def pass_dummy_data(definition_file, dataset_file, dummy_data_file):
    log.info(f"Generating dataset for {str(definition_file)}")

    dataset_definition = load_definition(definition_file)
    validate_dummy_data_file(dataset_definition, dummy_data_file)
    validate_file_types_match(dummy_data_file, dataset_file)

    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dummy_data_file, dataset_file)


def validate_dataset(definition_file, output_file, backend_id):
    log.info(f"Validating dataset for {str(definition_file)}")

    dataset_definition = load_definition(definition_file)
    backend = BACKENDS[backend_id]()
    query_engine = backend.query_engine_class(None, backend)
    results = validate(dataset_definition, query_engine)
    log.info("Validation succeeded")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open(mode="w") as f:
        for entry in results:
            f.write(f"{str(entry)}\n")


def load_tutorial_data():
    log.info("Loading tutorial dataset")
    raise NotImplementedError


def generate_measures(
    definition_path, input_file, dataset_file
):  # pragma: no cover (measures not implemented)
    raise NotImplementedError


def test_connection(backend, url):
    from sqlalchemy import select

    backend = BACKENDS[backend]()
    query_engine = backend.query_engine_class(url, backend)
    with query_engine.engine.connect() as connection:
        connection.execute(select(1))
    print("SUCCESS")


def load_definition(definition_file):
    load_module(definition_file)
    assert len(dataset_registry.datasets) == 1
    return dataset_registry.datasets.copy().pop()


def load_module(definition_path):
    # Taken from the official recipe for importing a module from a file path:
    # https://docs.python.org/3.9/library/importlib.html#importing-a-source-file-directly

    # The name we give the module is arbitrary
    spec = importlib.util.spec_from_file_location("dataset", definition_path)
    module = importlib.util.module_from_spec(spec)
    # Temporarily add the directory containing the definition to the path so that the
    # definition can import library modules from that directory
    with add_to_sys_path(str(definition_path.parent)):
        spec.loader.exec_module(module)


@contextmanager
def add_to_sys_path(directory):
    original = sys.path.copy()
    sys.path.append(directory)
    try:
        yield
    finally:
        sys.path = original


def extract(dataset_definition, query_engine):
    """
    Extracts the dataset from the backend specified
    Args:
        dataset_definition: The definition of the Dataset
        query_engine: The Query Engine with which the Dataset is being extracted
    Returns:
        Yields the dataset as rows
    """
    variable_definitions = ql.compile(dataset_definition)
    with query_engine.execute_query(variable_definitions) as results:
        for row in results:
            yield dict(row)


def validate(dataset_definition, query_engine):
    try:
        variable_definitions = ql.compile(dataset_definition)
        setup_queries, results_query, cleanup_queries = query_engine.get_queries(
            variable_definitions
        )
        return setup_queries + [results_query] + cleanup_queries
    except Exception:  # pragma: no cover (puzzle: dataset definition that compiles to QM but not SQL)
        log.error("Validation failed")
        # raise the exception to ensure the job fails and the error and traceback are logged
        raise


def write_dataset(results, dataset_file):
    with dataset_file.open(mode="w") as f:
        writer = csv.writer(f)
        headers = None
        for entry in results:
            fields = entry.keys()
            if not headers:
                headers = fields
                writer.writerow(headers)
            else:
                assert fields == headers, f"Expected fields {headers}, but got {fields}"
            writer.writerow(entry.values())
