import sqlalchemy.orm

from databuilder.query_model import (
    AggregateByPatient,
    Function,
    SelectPatientTable,
    SelectTable,
)

from ..lib.util import orm_class_from_schema


def setup(schema, num_patient_tables, num_event_tables):
    base_class = sqlalchemy.orm.declarative_base()

    patient_table_names, patient_classes = _build_orm_classes(
        "p", num_patient_tables, schema, base_class
    )
    event_table_names, event_classes = _build_orm_classes(
        "e", num_event_tables, schema, base_class
    )

    all_patients_query = _build_query(patient_table_names, event_table_names, schema)

    return (
        patient_classes,
        event_classes,
        all_patients_query,
        base_class.metadata,
    )


def _build_orm_classes(prefix, count, schema, base_class):
    names = [f"{prefix}{i}" for i in range(count)]
    classes = [_build_orm_class(name, schema, base_class) for name in names]
    return names, classes


def _build_orm_class(name, schema, base_class):
    class_ = orm_class_from_schema(base_class, name, schema)
    # It's helpful to have the classes available as module properties so that we can
    # copy-paste failing test cases from Hypothesis.
    globals()[name] = class_
    return class_


def _build_query(patient_tables, event_tables, schema):
    clauses = []

    for table in patient_tables:
        clauses.append(
            AggregateByPatient.Exists(
                source=SelectPatientTable(name=table, schema=schema)
            )
        )

    for table in event_tables:
        clauses.append(
            AggregateByPatient.Exists(source=SelectTable(name=table, schema=schema))
        )

    return _join_with_or(clauses)


def _join_with_or(clauses):
    query = clauses[0]
    for clause in clauses[1:]:
        query = Function.Or(query, clause)
    return query
