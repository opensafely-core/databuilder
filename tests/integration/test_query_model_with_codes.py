import pytest

from databuilder.codes import CTV3Code, SNOMEDCTCode
from databuilder.query_model import (
    Function,
    SelectColumn,
    SelectTable,
    TableSchema,
    TypeValidationError,
    Value,
)


def test_comparisons_between_codes_are_ok():
    events = SelectTable("events", schema=TableSchema({"code": CTV3Code}))
    code = SelectColumn(events, "code")
    assert Function.EQ(code, Value(CTV3Code("abc")))


def test_attempts_to_mix_coding_systems_are_rejected():
    events = SelectTable("events", schema=TableSchema({"code": CTV3Code}))
    code = SelectColumn(events, "code")
    with pytest.raises(TypeValidationError):
        Function.EQ(code, Value(SNOMEDCTCode("abc")))


def test_attempts_to_mix_codes_and_strings_are_rejected():
    events = SelectTable("events", schema=TableSchema({"code": CTV3Code}))
    code = SelectColumn(events, "code")
    with pytest.raises(TypeValidationError):
        Function.EQ(code, Value("abc"))
