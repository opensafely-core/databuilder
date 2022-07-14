import csv
import importlib.util
import shutil
import sys
import traceback
from contextlib import contextmanager

import structlog

from databuilder.query_language import Dataset
from databuilder.sqlalchemy_utils import clause_as_str

from . import query_language as ql
from .validate_dummy_data import validate_dummy_data_file, validate_file_types_match

log = structlog.getLogger()

DATABUILDER_CODE = True


def is_databuilder(tb: traceback) -> bool:
    """
    Check if the code in the traceback frame is
    from the Databuilder Code Base. This aims to remove
    traceback involving other libaries such as importlib.
    """
    db_globals = tb.tb_frame.f_globals
    return "DATABUILDER_CODE" in db_globals


def user_code_traceback_level(tb: traceback, error_type: Exception) -> int:
    """
    Find the length of the traceback that originates in the user code.
    """
    # This is needed as SyntaxError has additional lines to display exact
    # location of syntax error
    if error_type == SyntaxError:
        length = 3
    else:
        length = 1

    while tb and is_databuilder(tb):
        tb = tb.tb_next
        length += 1
    return length


def generate_dataset(
    definition_file,
    dataset_file,
    backend_id,
    db_url,
    environ,
):
    log.info(f"Generating dataset for {str(definition_file)}")

    dataset_definition = load_definition(definition_file)
    backend = import_string(backend_id)()
    query_engine = backend.query_engine_class(db_url, backend, config=environ)
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


def dump_dataset_sql(
    definition_file, output_file, backend_id, query_engine_id, environ
):
    log.info(f"Generating SQL for {str(definition_file)}")

    dataset_definition = load_definition(definition_file)
    query_engine = get_query_engine(backend_id, query_engine_id, environ)

    variable_definitions = ql.compile(dataset_definition)
    all_query_strings = get_sql_strings(query_engine, variable_definitions)
    log.info("SQL generation succeeded")

    with open_output_file(output_file) as f:
        for query_str in all_query_strings:
            f.write(f"{query_str};\n\n")


def get_sql_strings(query_engine, variable_definitions):
    setup_queries, results_query, cleanup_queries = query_engine.get_queries(
        variable_definitions
    )
    all_queries = setup_queries + [results_query] + cleanup_queries
    dialect = query_engine.sqlalchemy_dialect()
    return [clause_as_str(query, dialect) for query in all_queries]


def open_output_file(output_file):
    # If a file path is supplied, create it and open for writing
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        return output_file.open(mode="w")
    # Otherwise return `stdout` wrapped in a no-op context manager
    else:
        return nullcontext(sys.stdout)


def get_query_engine(backend_id, query_engine_id, environ):
    # Load backend if supplied
    if backend_id:
        backend = import_string(backend_id)()
    else:
        backend = None

    # If there's an explictly specified query engine class use that
    if query_engine_id:
        query_engine_class = import_string(query_engine_id)
    # Otherwise use whatever the backend specifies
    elif backend:
        query_engine_class = backend.query_engine_class
    # Default to using SQLite
    else:
        query_engine_class = import_string(
            "databuilder.query_engines.sqlite.SQLiteQueryEngine"
        )

    return query_engine_class(dsn=None, backend=backend, config=environ)


def generate_measures(
    definition_path, input_file, dataset_file
):  # pragma: no cover (measures not implemented)
    raise NotImplementedError


def test_connection(backend_id, url, environ):
    from sqlalchemy import select

    backend = import_string(backend_id)()
    query_engine = backend.query_engine_class(url, backend, config=environ)
    with query_engine.engine.connect() as connection:
        connection.execute(select(1))
    print("SUCCESS")


def load_definition(definition_file):
    module = load_module(definition_file)
    try:
        dataset = module.dataset
    except AttributeError:
        raise AttributeError("A dataset definition must define one 'dataset'")
    assert isinstance(
        dataset, Dataset
    ), "'dataset' must be an instance of databuilder.query_language.Dataset()"
    return dataset


def load_module(definition_path):
    # Taken from the official recipe for importing a module from a file path:
    # https://docs.python.org/3.9/library/importlib.html#importing-a-source-file-directly

    # The name we give the module is arbitrary
    spec = importlib.util.spec_from_file_location("dataset", definition_path)
    module = importlib.util.module_from_spec(spec)
    # Temporarily add the directory containing the definition to the path so that the
    # definition can import library modules from that directory
    with add_to_sys_path(str(definition_path.parent)):
        try:
            spec.loader.exec_module(module)
            return module
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            length = user_code_traceback_level(exc_tb, exc_type)

            tb = traceback.format_exception(exc_type, exc_value, exc_tb)[-length:]
            print(
                f"***********\nError Type: {exc_type.__name__}\nMessage: {exc_value}\n***********\nSpecific Details:"
            )
            for i in tb:
                print(i)
            sys.exit()


@contextmanager
def add_to_sys_path(directory):
    original = sys.path.copy()
    sys.path.append(directory)
    try:
        yield
    finally:
        sys.path = original


def import_string(dotted_path):
    module_name, _, attribute_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


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
