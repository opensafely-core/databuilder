from . import types
from .base import Column, PatientIDColumn, TableContract
from .constraints import FirstOfMonthConstraint, NotNullConstraint

__all__ = ["Patients"]


class Patients(TableContract):
    _name = "patients"

    patient_id = PatientIDColumn
    date_of_birth = Column(
        type=types.Date(),
        description=(
            "Patient's year and month of birth, provided in format YYYY-MM-01. "
            "The day will always be the first of the month."
        ),
        constraints=[FirstOfMonthConstraint(), NotNullConstraint()],
    )
    sex = Column(
        type=types.Choice("female", "male", "intersex", "unknown"),
        description="Patient's sex as defined by the options: male, female, intersex, unknown.",
        implementation_notes_to_add_to_description=(
            'Specify how this has been determined, e.g. "sex at birth", or "current sex".'
        ),
        constraints=[NotNullConstraint()],
    )
    date_of_death = Column(
        type=types.Date(),
        description=(
            "Patient's year and month of death, provided in format YYYY-MM-01."
            "The day will always be the first of the month."
        ),
        constraints=[FirstOfMonthConstraint()],
    )
