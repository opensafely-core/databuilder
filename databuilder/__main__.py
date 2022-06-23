import os
import sys
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

from .backends import BACKENDS
from .main import (
    generate_dataset,
    generate_measures,
    load_tutorial_data,
    pass_dummy_data,
    test_connection,
    validate_dataset,
)


def main(args=None):
    parser = build_parser()

    if args is None:
        # allow the passing in of args, for testing, but otherwise look them
        # up.  This saves a lot of laborious mocking in tests.
        args = sys.argv[1:]  # pragma: no cover

    options = parser.parse_args(args)

    if options.which == "generate-dataset":
        database_url = os.environ.get("DATABASE_URL")
        dummy_data_file = options.dummy_data_file

        if database_url:
            generate_dataset(
                definition_file=options.dataset_definition,
                dataset_file=options.output,
                db_url=database_url,
                backend_id=os.environ.get("OPENSAFELY_BACKEND"),
                temporary_database=os.environ.get("TEMP_DATABASE_NAME"),
            )
        elif dummy_data_file:
            pass_dummy_data(options.dataset_definition, options.output, dummy_data_file)
        else:
            parser.error(
                "error: either --dummy-data-file or DATABASE_URL environment variable is required"
            )
    elif options.which == "dump-dataset-sql":
        validate_dataset(
            options.dataset_definition,
            options.output,
            backend_id=options.backend,
        )
    elif options.which == "generate-measures":
        generate_measures(
            definition_path=options.dataset_definition,
            input_file=options.input,
            dataset_file=options.output,
        )
    elif options.which == "test-connection":
        test_connection(
            backend=options.backend,
            url=options.url,
        )
    elif options.which == "load-tutorial-data":
        load_tutorial_data()
    elif options.which == "print-help":
        parser.print_help()
    else:
        assert False, f"Unhandled subcommand: {options.which}"


def build_parser():
    parser = ArgumentParser(
        prog="databuilder", description="Generate datasets in OpenSAFELY"
    )
    parser.set_defaults(which="print-help")

    subparsers = parser.add_subparsers(help="sub-command help")
    add_generate_dataset(subparsers)
    add_dump_dataset_sql(subparsers)
    add_generate_measures(subparsers)
    add_test_connection(subparsers)
    add_load_tutorial_data(subparsers)

    return parser


def add_generate_dataset(subparsers):
    parser = subparsers.add_parser("generate-dataset", help="Generate a dataset")
    parser.set_defaults(which="generate-dataset")
    parser.add_argument(
        "--dataset-definition",
        help="The path of the file where the dataset is defined",
        type=existing_python_file,
    )
    parser.add_argument(
        "--output",
        help="Path and filename (or pattern) of the file(s) where the dataset will be written",
        type=Path,
    )
    parser.add_argument(
        "--dummy-data-file",
        help="Provide dummy data from a file to be validated and used as the dataset",
        type=Path,
    )


def add_dump_dataset_sql(subparsers):
    parser = subparsers.add_parser(
        "dump-dataset-sql",
        help="Validate the dataset definition against the specified backend",
    )
    parser.set_defaults(which="dump-dataset-sql")
    parser.add_argument(
        "backend",
        type=str,
        choices=BACKENDS,  # allow all registered backend subclasses
    )
    parser.add_argument(
        "--dataset-definition",
        help="The path of the file where the dataset is defined",
        type=existing_python_file,
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Path and filename (or pattern) of the file(s) where the output will be written",
        type=Path,
    )


def add_generate_measures(subparsers):
    parser = subparsers.add_parser(
        "generate-measures", help="Generate measures from a dataset"
    )
    parser.set_defaults(which="generate-measures")
    parser.add_argument(
        "--input",
        help="Path and filename (or pattern) of the input file(s)",
        type=Path,
    )
    parser.add_argument(
        "--output",
        help="Path and filename (or pattern) of the file(s) where the dataset will be written",
        type=Path,
    )
    parser.add_argument(
        "--dataset-definition",
        help="The path of the file where the dataset is defined",
        type=existing_python_file,
    )


def add_test_connection(subparsers):
    parser = subparsers.add_parser(
        "test-connection", help="test the database connection configuration"
    )
    parser.set_defaults(which="test-connection")
    parser.add_argument(
        "--backend",
        "-b",
        help="backend type to test",
        default=os.environ.get("BACKEND", os.environ.get("OPENSAFELY_BACKEND")),
    )
    parser.add_argument(
        "--url",
        "-u",
        help="db url",
        default=os.environ.get("DATABASE_URL"),
    )


def add_load_tutorial_data(subparsers):
    parsers = subparsers.add_parser(
        "load-tutorial-data", help="Load data for working with tutorial"
    )
    parsers.set_defaults(which="load-tutorial-data")
    parsers.add_argument(
        "--input",
        help="Path and filename of the input file",
        type=Path,
    )


def existing_python_file(value):
    path = Path(value)
    if not path.exists():
        raise ArgumentTypeError(f"{value} does not exist")
    if not path.suffix == ".py":
        raise ArgumentTypeError(f"{value} is not a Python file")
    return path


if __name__ == "__main__":
    main()
