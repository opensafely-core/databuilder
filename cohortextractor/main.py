import csv
import importlib.util
import inspect
import shutil
import sys
from contextlib import contextmanager

from .backends import BACKENDS
from .query_utils import get_column_definitions
from .validate_dummy_data import validate_dummy_data


def generate_cohort(
    definition_path,
    output_file,
    backend_id,
    db_url,
    dummy_data_file=None,
    temporary_database=None,
):
    cohort = load_cohort(definition_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if dummy_data_file and not db_url:
        validate_dummy_data(cohort, dummy_data_file, output_file)
        shutil.copyfile(dummy_data_file, output_file)
    else:
        backend = BACKENDS[backend_id](db_url, temporary_database=temporary_database)
        results = extract(cohort, backend)
        write_output(results, output_file)


def generate_measures(definition_path, output_file):
    measures = load_measures(definition_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    for measure in measures:
        measure_output_file = str(output_file).replace("*", measure.id)
        results = measure.calculate()
        results.to_csv(measure_output_file)


def load_cohort(definition_path):
    definition_module = load_module(definition_path)
    cohort_classes = [
        obj
        for name, obj in inspect.getmembers(definition_module)
        if inspect.isclass(obj)
    ]
    assert len(cohort_classes) == 1, "A study definition must contain one class only"
    return cohort_classes[0]


def load_measures(definition_path):
    ...


def load_module(definition_path):
    definition_dir = definition_path.parent
    module_name = definition_path.stem

    # Add the directory containing the definition to the path so that the definition can import library modules from
    # that directory
    with added_to_path(str(definition_dir)):
        module = importlib.import_module(module_name)
        # Reload the module in case a module with the same name was loaded previously
        importlib.reload(module)
        return module


@contextmanager
def added_to_path(directory):
    original = sys.path.copy()
    sys.path.append(directory)
    try:
        yield
    finally:
        sys.path = original


def extract(cohort, backend):
    cohort = get_column_definitions(cohort)
    query_engine = backend.query_engine_class(cohort, backend)
    with query_engine.execute_query() as results:
        for row in results:
            yield dict(row)


def write_output(results, output_file):
    with output_file.open(mode="w") as f:
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
