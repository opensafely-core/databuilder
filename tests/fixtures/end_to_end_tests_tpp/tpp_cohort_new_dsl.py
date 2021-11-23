from cohortextractor.concepts import tables
from cohortextractor.definition import register
from cohortextractor.dsl import Cohort

cohort = Cohort()
cohort.set_population(tables.registrations.exists_for_patient())
events = tables.ClinicalEvents()
cohort.date = events.sort_by(events.date).first_for_patient().select_column(events.date)
cohort.event = (
    events.sort_by(events.code).first_for_patient().select_column(events.code)
)

register(cohort)
