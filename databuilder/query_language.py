"""
This contains just enough of erhQL to be able to extract the following dataset:

    year_of_birth = patients.date_of_birth.year
    dataset = Dataset()
    dataset.set_population(year_of_birth <= 2000)
    dataset.year_of_birth = year_of_birth
"""


from __future__ import annotations

from .query_model import Function, Node, SelectColumn, SelectPatientTable, Value


class Dataset:
    def __init__(self):
        self.variables = {}

    def set_population(self, population: BoolSeries) -> None:
        self.population = population

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Series):
            # TODO: add check that series is patient series not event series
            self.variables[name] = value
        else:
            super().__setattr__(name, value)

    def compile(self) -> dict[str, Node]:  # noqa A003
        return {name: variable.qm_node for name, variable in self.variables.items()}


class Table:
    def __init__(self, qm_node):
        self.qm_node = qm_node

    def __getattribute__(self, item):
        attr = super().__getattribute__(item)
        if isinstance(attr, Column):
            # This is a syntactic convenience to allow columns to be treated as series within dataset definitions.
            return attr.as_series(self)
        else:
            return attr


class PatientTable(Table):
    def __init__(self):
        super().__init__(SelectPatientTable(name=self.__name__))


class Column:
    def __init__(self, name: str, series_type: Series):
        self.name = name
        self.series_type = series_type

    def as_series(self, table):
        return self.series_type(SelectColumn(name=self.name, source=table.qm_node))


class DateColumn(Column):
    def __init__(self, name):
        super().__init__(name, DateSeries)


class Series:
    def __init__(self, qm_node):
        self.qm_node = qm_node


class IdSeries(Series):
    pass


class BoolSeries(Series):
    pass


class IntSeries(Series):
    def __le__(self, other):
        return BoolSeries(Function.LE(lhs=self.qm_node, rhs=Value(other)))


class DateSeries(Series):
    @property
    def year(self):
        return IntSeries(Function.YearFromDate(source=self.qm_node))
