import csv
import importlib.util
import inspect
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import structlog

from .backends import BACKENDS
from .date_utils import generate_date_range
from .measure import Measure, MeasuresManager
from .query_utils import get_column_definitions, get_measures
from .validate_dummy_data import validate_dummy_data


log = structlog.getLogger()


def generate_cohort(
    definition_path,
    output_file,
    backend_id,
    db_url,
    index_date_range=None,
    dummy_data_file=None,
    temporary_database=None,
):
    log.info(
        f"Generating cohort for {definition_path.name} as {output_file}",
    )
    log.debug(
        "args:",
        definition_path=definition_path,
        output_file=output_file,
        backend=backend_id,
        index_date_range=index_date_range,
        dummy_data_file=dummy_data_file,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    for index_date in generate_date_range(index_date_range):
        if index_date is not None:
            log.info(f"Setting index_date to {index_date}")
            date_suffix = f"_{index_date}"
        else:
            date_suffix = ""
        cohort = load_cohort(definition_path, index_date)
        output_file_with_date = Path(str(output_file).replace("*", date_suffix))
        if dummy_data_file and not db_url:
            dummy_data_file_with_date = Path(
                str(dummy_data_file).replace("*", date_suffix)
            )
            validate_dummy_data(
                cohort, dummy_data_file_with_date, output_file_with_date
            )
            shutil.copyfile(dummy_data_file_with_date, output_file_with_date)
        else:
            backend = BACKENDS[backend_id](
                db_url, temporary_database=temporary_database
            )
            results = extract(cohort, backend)
            write_output(results, output_file_with_date)


def generate_measures(definition_path, input_file, output_file):
    cohort = load_cohort(definition_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    for measure_id, results in calculate_measures_results(cohort, input_file):
        measure_output_file = str(output_file).replace("*", measure_id)
        results.to_csv(measure_output_file, index=False)
        log.info("Created measure output", output=output_file)


def calculate_measures_results(cohort, input_file):
    measures = get_measures(cohort)
    measures_manager = MeasuresManager(measures, input_file)
    yield from measures_manager.calculate_measures()


def load_cohort(definition_path, index_date=None):
    definition_module = load_module(definition_path, index_date)
    imported_classes = [Measure]
    cohort_classes = [
        obj
        for name, obj in inspect.getmembers(definition_module)
        if inspect.isclass(obj) and obj not in imported_classes
    ]
    assert len(cohort_classes) == 1, "A study definition must contain one class only"
    return cohort_classes[0]


def load_module(definition_path, index_date=None):
    """
    Load a study definition module from the given definition_path.  If an index date is
    provided, generate a temporary study definition file using that index date instead
    and load that instead.

    definition_path: Path to cohort definition
    index_date: date string in format YYYY-MM-DD
    """
    if index_date is not None:
        with temp_definition_path(definition_path, index_date) as temp_path:
            with open(definition_path, "r") as orig, open(temp_path, "w") as new:
                # Copy the cohort definition file, replace the BASE_INDEX_DATE definition
                # with the current index date and load this modified module
                contents = orig.read()
                base_index_date_pattern = r"BASE_INDEX_DATE\s*=\s*.+"
                if not re.findall(base_index_date_pattern, contents):
                    raise RuntimeError(
                        "index-date-range requires BASE_INDEX_DATE to be defined in study definition"
                    )
                index_date_definition_pattern = (
                    rf"({base_index_date_pattern})([\s|\n].*)"
                )
                contents = re.sub(
                    index_date_definition_pattern,
                    rf'BASE_INDEX_DATE = "{index_date}"\2',
                    contents,
                )
                new.write(contents)
            module = _load_module(temp_path)
        return module
    return _load_module(definition_path)


def _load_module(definition_path):
    # Add the directory containing the definition to the path so that the definition can import library modules from
    # that directory
    definition_dir = definition_path.parent
    module_name = definition_path.stem
    with added_to_path(str(definition_dir)):
        module = importlib.import_module(module_name)
        # Reload the module in case a module with the same name was loaded previously
        importlib.reload(module)
        return module


@contextmanager
def temp_definition_path(definition_path, index_date):
    temp_path = definition_path.parent / f"{definition_path.stem}_{index_date}.py"
    yield temp_path
    temp_path.unlink()


@contextmanager
def added_to_path(directory):
    original = sys.path.copy()
    sys.path.append(directory)
    try:
        yield
    finally:
        sys.path = original


def extract(cohort_class, backend):
    cohort = get_column_definitions(cohort_class)
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
