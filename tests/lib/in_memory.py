import contextlib
import operator
import os
import re
from collections import defaultdict
from datetime import date

from databuilder.query_engines.base import BaseQueryEngine
from databuilder.query_model import Position, Value

from .util import iter_flatten

T = True
F = False
N = None


DEBUG = os.environ.get("DEBUG")


class InMemoryQueryEngine(BaseQueryEngine):
    engine = None

    @contextlib.contextmanager
    def execute_query(self):
        name_to_series = {
            "patient_id": PatientSeries(
                {patient: patient for patient in self.all_patients}
            )
        }

        for name, node in self.column_definitions.items():
            ps = self.visit(node)
            name_to_series[name] = ps
            if DEBUG:
                print("-" * 80)
                print(name)
                print("-" * 40)
                print(node)
                print("-" * 40)
                print(ps)

        pf = PatientFrame(name_to_series)
        if DEBUG:
            print(pf)

        records = []
        for patient in self.all_patients:
            if name_to_series["population"][patient]:
                records.append(
                    {
                        col_name: self.extract_value(name_to_series[col_name][patient])
                        for col_name in name_to_series
                        if col_name != "population"
                    }
                )

        yield records

    def extract_value(self, value):
        return value.value if isinstance(value, Value) else value

    @property
    def database(self):
        return self.backend.database_url  # Hack!

    @property
    def tables(self):
        return self.database.tables

    @property
    def all_patients(self):
        return self.database.all_patients

    def visit(self, node):
        visitor = getattr(self, f"visit_{type(node).__name__}")
        return visitor(node)

    def visit_Code(self, node):
        return node.value

    def visit_Value(self, node):
        value = node.value
        if isinstance(value, frozenset):
            return {self.visit(v) for v in value}
        if isinstance(value, str) and re.match(r"\d\d\d\d-\d\d-\d\d", value):
            return date.fromisoformat(value)
        return value

    def visit_Position(self, node):
        return node

    def visit_SelectTable(self, node):
        return self.tables[node.name]

    def visit_SelectPatientTable(self, node):
        return self.tables[node.name]

    def visit_SelectColumn(self, node):
        frame = self.visit(node.source)
        return frame[node.name]

    def visit_Filter(self, node):
        frame = self.visit(node.source)
        event_to_flag = self.visit(node.condition).event_to_value
        return frame.filter_to_event_frame(event_to_flag)

    def visit_Sort(self, node):
        frame = self.visit(node.source)
        patient_to_sorted_events = {
            patient: sorted(group, key=lambda pair: pair[1])
            for patient, group in self.visit(node.sort_by).patient_to_groups().items()
        }
        return frame, patient_to_sorted_events

    def visit_PickOneRowPerPatient(self, node):
        frame, patient_to_sorted_events = self.visit(node.source)
        position = self.visit(node.position)
        ix = {Position.FIRST: 0, Position.LAST: -1}[position]
        patient_to_event = {
            patient: events[ix][0]
            for patient, events in patient_to_sorted_events.items()
        }
        return frame.filter_to_patient_frame(patient_to_event)

    def visit_Exists(self, node):
        patients = self.visit(node.source).patient_to_events.keys()
        return PatientSeries(
            {patient: patient in patients for patient in self.all_patients}
        )

    def visit_Min(self, node):
        assert False

    def visit_Max(self, node):
        assert False

    def visit_Count(self, node):
        frame = self.visit(node.source)
        patient_to_groups = frame.patient_to_groups()
        return PatientSeries(
            {
                patient: len(patient_to_groups.get(patient, []))
                for patient in self.all_patients
            }
        )

    def visit_Sum(self, node):
        frame = self.visit(node.source)
        return PatientSeries(
            {
                patient: sum(value for event, value in group)
                for patient, group in frame.patient_to_groups().items()
            }
        )

    def visit_CombineAsSet(self, node):
        frame = self.visit(node.source)
        return PatientSeries(
            {
                patient: {value for _, value in group}
                for patient, group in frame.patient_to_groups().items()
            }
        )

    def visit_binary_op_with_null(self, node, op):
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        return lhs.binary_op_with_null(op, rhs)

    def visit_EQ(self, node):
        return self.visit_binary_op_with_null(node, operator.eq)

    def visit_NE(self, node):
        return self.visit_binary_op_with_null(node, operator.ne)

    def visit_LT(self, node):
        return self.visit_binary_op_with_null(node, operator.lt)

    def visit_LE(self, node):
        return self.visit_binary_op_with_null(node, operator.le)

    def visit_GT(self, node):
        return self.visit_binary_op_with_null(node, operator.gt)

    def visit_GE(self, node):
        return self.visit_binary_op_with_null(node, operator.ge)

    def visit_And(self, node):
        def op(lhs, rhs):
            return {
                (T, T): T,
                (T, N): N,
                (T, F): F,
                (N, T): N,
                (N, N): N,
                (N, F): F,
                (F, T): F,
                (F, N): F,
                (F, F): F,
            }[lhs, rhs]

        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        return lhs.binary_op(op, rhs)

    def visit_Or(self, node):
        def op(lhs, rhs):
            return {
                (T, T): T,
                (T, N): T,
                (T, F): T,
                (N, T): T,
                (N, N): N,
                (N, F): N,
                (F, T): T,
                (F, N): N,
                (F, F): F,
            }[lhs, rhs]

        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        return lhs.binary_op(op, rhs)

    def visit_Not(self, node):
        def op(value):
            return {
                T: F,
                N: N,
                F: T,
            }[value]

        return self.visit(node.source).unary_op(op)

    def visit_IsNull(self, node):
        def op(value):
            return value is None

        return self.visit(node.source).unary_op(op)

    def visit_Add(self, node):
        assert False

    def visit_Subtract(self, node):
        assert False

    def visit_RoundToFirstOfMonth(self, node):
        assert False

    def visit_RoundToFirstOfYear(self, node):
        assert False

    def visit_DateAdd(self, node):
        assert False

    def visit_DateSubtract(self, node):
        assert False

    def visit_DateDifference(self, node):
        start = self.visit(node.start)
        end = self.visit(node.end)
        units = self.visit(node.units)

        def op(start, end):
            if start is None or end is None:
                return None
            delta = end - start
            if units == "years":
                return int(delta.days / 365.25)
            else:
                assert False, units

        return start.binary_op(op, end)

    def visit_In(self, node):
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        return lhs.binary_op(lambda needle, haystack: needle in haystack, rhs)

    def visit_Categorise(self, node):
        patient_to_value = {}

        for category, category_node in node.categories.items():
            series = self.visit(category_node)
            for patient, value in series.patient_to_value.items():
                if value and patient not in patient_to_value:
                    patient_to_value[patient] = category

        for patient in self.all_patients:
            if patient not in patient_to_value:
                patient_to_value[patient] = node.default

        return PatientSeries(patient_to_value)


class InMemoryDatabase:
    def setup(self, *input_data):
        self.all_patients = set()
        records = {
            "patients": [],
            "practice_registrations": [],
            "clinical_events": [],
            "positive_tests": [],
        }

        def maybe_date(s):
            return date.fromisoformat(s) if s else s

        for item in iter_flatten(input_data):
            self.all_patients.add(item.PatientId)
            if type(item).__name__ == "Patients":
                records["patients"].append(
                    (
                        item.PatientId,
                        item.Height,
                        maybe_date(item.DateOfBirth),
                        item.Sex,
                    )
                )
            elif type(item).__name__ == "RegistrationHistory":
                records["practice_registrations"].append(
                    (
                        item.PatientId,
                        item.StpId,
                        maybe_date(item.StartDate),
                        maybe_date(item.EndDate),
                    )
                )
            elif type(item).__name__ == "CTV3Events":
                records["clinical_events"].append(
                    (
                        item.PatientId,
                        item.EventCode,
                        item.System,
                        maybe_date(item.Date),
                        item.ResultValue,
                    )
                )
            elif type(item).__name__ == "AllTests":
                records["positive_tests"].append(
                    (
                        item.PatientId,
                        item.PositiveResult,
                        maybe_date(item.TestDate),
                    )
                )
            else:
                assert False, item

        self.tables = {
            "patients": PatientFrame.from_records(
                records["patients"],
                ["patient_id", "height", "date_of_birth", "sex"],
            ),
            "practice_registrations": EventFrame.from_records(
                records["practice_registrations"],
                ["patient_id", "stp", "date_start", "date_end"],
            ),
            "clinical_events": EventFrame.from_records(
                records["clinical_events"],
                ["patient_id", "code", "system", "date", "value"],
            ),
            "positive_tests": EventFrame.from_records(
                records["positive_tests"],
                ["patient_id", "result", "test_date"],
            ),
        }

    def host_url(self):
        # Hack!
        return self


class PatientFrame:
    def __init__(self, name_to_series):
        self.name_to_series = name_to_series

    @classmethod
    def from_records(cls, records, col_names):
        if not records:
            name_to_series = {col_name: PatientSeries({}) for col_name in col_names}
        else:
            series_records = list(zip(*records))
            assert col_names[0] == "patient_id"
            patient_ids = series_records[0]
            name_to_series = {
                col_names: PatientSeries(dict(zip(patient_ids, series_record)))
                for col_names, series_record in zip(col_names, series_records)
            }
        return cls(name_to_series)

    def __repr__(self):
        rows = []
        rows.append(" | ".join(col_name.ljust(17) for col_name in self.name_to_series))
        rows.append("-+-".join("-" * 17 for _ in self.name_to_series))
        for patient in sorted(self.patient_id.patient_to_value):
            rows.append(
                " | ".join(
                    str(self.name_to_series[col_name][patient]).ljust(17)
                    for col_name in self.name_to_series
                )
            )

        return "\n".join(rows)

    def __getattr__(self, col_name):
        return self.name_to_series[col_name]

    def __getitem__(self, col_name):
        return self.name_to_series[col_name]


class EventFrame:
    def __init__(self, name_to_series):
        self.name_to_series = name_to_series

    @classmethod
    def from_records(cls, records, col_names):
        if not records:
            name_to_series = {col_name: EventSeries({}, {}) for col_name in col_names}
        else:
            series_records = list(zip(*records))
            assert col_names[0] == "patient_id"
            patient_ids = series_records[0]
            event_to_patient = dict(enumerate(patient_ids))

            patient_to_events = defaultdict(list)
            for event, patient in event_to_patient.items():
                patient_to_events[patient].append(event)
            patient_to_events = dict(patient_to_events)

            name_to_series = {
                col_names: EventSeries(
                    dict(enumerate(series_record)), patient_to_events
                )
                for col_names, series_record in zip(col_names, series_records)
            }

        return cls(name_to_series)

    def __repr__(self):
        rows = []
        rows.append(" | ".join(col_name.ljust(17) for col_name in self.name_to_series))
        rows.append("-+-".join("-" * 17 for _ in self.name_to_series))
        for event in sorted(self.patient_id.event_to_value):
            rows.append(
                " | ".join(
                    str(self.name_to_series[col_name][event]).ljust(17)
                    for col_name in self.name_to_series
                )
            )

        return "\n".join(rows)

    def __getattr__(self, col_names):
        return self.name_to_series[col_names]

    def __getitem__(self, col_name):
        return self.name_to_series[col_name]

    def filter_to_patient_frame(self, patient_to_event):
        new_name_to_series = {}
        for name, series in self.name_to_series.items():
            new_name_to_series[name] = PatientSeries(
                {
                    patient: series.event_to_value[event]
                    for patient, event in patient_to_event.items()
                }
            )
        return PatientFrame(new_name_to_series)

    def filter_to_event_frame(self, event_to_flag):
        new_name_to_series = {}
        for name, series in self.name_to_series.items():
            new_event_to_value = {
                event: series.event_to_value[event]
                for event, flag in event_to_flag.items()
                if flag
            }

            new_name_to_series[name] = EventSeries(
                new_event_to_value,
                series.make_patient_to_events(new_event_to_value),
            )
        return EventFrame(new_name_to_series)


class PatientSeries:
    def __init__(self, patient_to_value):
        self.patient_to_value = patient_to_value

    def __repr__(self):
        return "\n".join(
            f"{patient} | {value}"
            for patient, value in sorted(self.patient_to_value.items())
        )

    def __getitem__(self, patient):
        return self.patient_to_value.get(patient)

    def unary_op(self, fn):
        patients = self.patient_to_value.keys()
        new_patient_to_value = {patient: fn(self[patient]) for patient in patients}
        return PatientSeries(new_patient_to_value)

    def binary_op(self, fn, other):
        if isinstance(other, EventSeries):
            return other.binary_op(fn, self)

        if not isinstance(other, PatientSeries):
            return self.binary_op(fn, self.make_series_from_value(other))

        patients = self.patient_to_value.keys() & other.patient_to_value.keys()
        new_patient_to_value = {
            patient: fn(self[patient], other[patient]) for patient in patients
        }
        return PatientSeries(new_patient_to_value)

    def binary_op_with_null(self, fn, other):
        def fn_with_null(lhs, rhs):
            if lhs is None or rhs is None:
                return None
            return fn(lhs, rhs)

        return self.binary_op(fn_with_null, other)

    def make_series_from_value(self, value):
        new_patient_to_value = {
            patient: value for patient in self.patient_to_value.keys()
        }
        return PatientSeries(new_patient_to_value)


class EventSeries:
    def __init__(self, event_to_value, patient_to_events):
        self.event_to_value = event_to_value
        self.patient_to_events = patient_to_events
        self.event_to_patient = {
            event: patient
            for patient, events in patient_to_events.items()
            for event in events
            if event in event_to_value
        }

    def __repr__(self):
        return "\n".join(
            f"{self.event_to_patient[event]} | {event} | {value}"
            for event, value in sorted(self.event_to_value.items())
        )

    def __getitem__(self, event):
        return self.event_to_value.get(event)

    def unary_op(self, fn):
        events = self.event_to_value.keys()
        new_event_to_value = {event: fn(self[event]) for event in events}
        return EventSeries(new_event_to_value, self.patient_to_events)

    def binary_op(self, fn, other):
        if isinstance(other, PatientSeries):
            return self.binary_op(fn, self.make_series_from_patient_series(other))

        if not isinstance(other, EventSeries):
            return self.binary_op(fn, self.make_series_from_value(other))

        events = self.event_to_value.keys() & other.event_to_value.keys()
        new_event_to_value = {event: fn(self[event], other[event]) for event in events}
        new_patient_to_events = self.make_patient_to_events(new_event_to_value)
        return EventSeries(new_event_to_value, new_patient_to_events)

    def binary_op_with_null(self, fn, other):
        def fn_with_null(lhs, rhs):
            if lhs is None or rhs is None:
                return None
            return fn(lhs, rhs)

        return self.binary_op(fn_with_null, other)

    def make_series_from_value(self, value):
        new_event_to_value = {event: value for event in self.event_to_value.keys()}
        return EventSeries(new_event_to_value, self.patient_to_events)

    def make_series_from_patient_series(self, patient_series):
        patients = patient_series.patient_to_value.keys()
        new_event_to_value = {
            event: patient_series[self.event_to_patient[event]]
            for event in self.event_to_value.keys()
            if self.event_to_patient[event] in patients
        }
        new_patient_to_events = self.make_patient_to_events(new_event_to_value)
        return EventSeries(new_event_to_value, new_patient_to_events)

    def make_patient_to_events(self, event_to_value):
        return {
            patient: [e for e in events if e in event_to_value]
            for patient, events in self.patient_to_events.items()
            if [e for e in events if e in event_to_value]
        }

    def patient_to_groups(self):
        return {
            patient: [(event, self.event_to_value[event]) for event in events]
            for patient, events in self.patient_to_events.items()
        }
