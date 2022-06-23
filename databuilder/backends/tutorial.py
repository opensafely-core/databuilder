from ..contracts import universal
from ..query_engines.sqlite import SQLiteQueryEngine
from .base import BaseBackend, Column, MappedTable


class TutorialBackend(BaseBackend):
    """Backend for working with data in a tutorial.

    This will allow data to be loaded in from the command-line."""

    backend_id = "tutorial"
    query_engine_class = SQLiteQueryEngine
    patient_join_column = "patient_id"

    patients = MappedTable(
        implements=universal.Patients,
        source="patients",
        columns=dict(
            sex=Column("varchar", source="sex"),
            date_of_birth=Column("date", source="dob"),
            date_of_death=Column("date", source="date_of_death"),
        ),
    )
