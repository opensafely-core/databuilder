# noqa
from databuilder.definition import register
from databuilder.query_language import Dataset
from databuilder.tables import patients

dataset = Dataset()
year = patients.date_of_birth.year
dataset.set_population(year >= 1900)
dataset.year = year
register(dataset)
