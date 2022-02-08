from .in_memory import EventFrame, EventSeries, PatientSeries


def negate(v):
    if v is None:
        return None
    return -v


def add(v1, v2):
    if v1 is None or v2 is None:
        return None
    return v1 + v2


def test_patient_series_unary_op():
    ps = PatientSeries({1: 111, 2: 222, 3: None})
    out = ps.unary_op(negate)
    assert out.patient_to_value == {1: -111, 2: -222, 3: None}


def test_event_series_unary_op():
    es = EventSeries(
        {101: 111, 102: 112, 201: 222, 202: None},
        {1: [101, 102], 2: [201, 202]},
    )
    out = es.unary_op(negate)
    assert out.event_to_value == {101: -111, 102: -112, 201: -222, 202: None}
    assert out.patient_to_events == {1: [101, 102], 2: [201, 202]}


def test_patient_series_binary_op_value():
    ps = PatientSeries({1: 111, 2: 222, 3: None})
    out = ps.binary_op(add, 1)
    assert out.patient_to_value == {1: 112, 2: 223, 3: None}


def test_patient_series_binary_op_patient_series():
    ps1 = PatientSeries({1: 111, 2: 222, 3: None, 4: 444})
    ps2 = PatientSeries({1: 111, 2: None, 3: 333, 5: 555})
    out = ps1.binary_op(add, ps2)
    assert out.patient_to_value == {1: 222, 2: None, 3: None}


def test_patient_series_binary_op_event_series():
    ps = PatientSeries({1: 111, 2: 222, 3: None, 4: 444})
    es = EventSeries(
        {101: 111, 102: 112, 201: 222, 202: None, 301: 333, 501: 555},
        {1: [101, 102], 2: [201, 202], 3: [301], 5: [501]},
    )
    out = ps.binary_op(add, es)
    assert out.event_to_value == {101: 222, 102: 223, 201: 444, 202: None, 301: None}
    assert out.patient_to_events == {1: [101, 102], 2: [201, 202], 3: [301]}


def test_event_series_binary_op_value():
    es = EventSeries(
        {101: 111, 102: 112, 201: 222, 202: None},
        {1: [101, 102], 2: [201, 202]},
    )
    out = es.binary_op(add, 1)
    assert out.event_to_value == {101: 112, 102: 113, 201: 223, 202: None}
    assert out.patient_to_events == {1: [101, 102], 2: [201, 202]}


def test_event_series_binary_op_patient_series():
    es = EventSeries(
        {101: 111, 102: 112, 201: 222, 202: None, 301: 333, 401: 444},
        {1: [101, 102], 2: [201, 202], 3: [301], 4: [401]},
    )
    ps = PatientSeries({1: 111, 2: 222, 3: None, 5: 555})
    out = es.binary_op(add, ps)
    assert out.event_to_value == {101: 222, 102: 223, 201: 444, 202: None, 301: None}
    assert out.patient_to_events == {1: [101, 102], 2: [201, 202], 3: [301]}


def test_event_series_binary_op_event_series():
    es1 = EventSeries(
        {101: 111, 102: 112, 201: 222, 202: None, 301: 333},
        {1: [101, 102], 2: [201, 202], 3: [301]},
    )
    es2 = EventSeries(
        {101: 111, 102: 112, 201: None, 202: 223, 401: 444},
        {1: [101, 102], 2: [201, 202], 4: [401]},
    )

    out = es1.binary_op(add, es2)
    assert out.event_to_value == {101: 222, 102: 224, 201: None, 202: None}
    assert out.patient_to_events == {1: [101, 102], 2: [201, 202]}


def test_event_series_patient_to_groups():
    es = EventSeries(
        {101: 111, 102: 112, 201: 222, 202: None, 301: 333},
        {1: [101, 102], 2: [201, 202], 3: [301]},
    )
    assert es.patient_to_groups() == {
        1: [(101, 111), (102, 112)],
        2: [(201, 222), (202, None)],
        3: [(301, 333)],
    }


def test_event_frame_filter_to_patient_frame():
    esp = EventSeries(
        {101: 1, 1: 1, 201: 2, 202: 2, 301: 3},
        {1: [101, 102], 2: [201, 202], 3: [301]},
    )
    es = EventSeries(
        {101: 111, 102: 112, 201: 222, 202: None, 301: 333},
        {1: [101, 102], 2: [201, 202], 3: [301]},
    )
    ef = EventFrame({"patient_id": esp, "v": es})
    pf = ef.filter_to_patient_frame({1: 101, 2: 202, 3: 301})
    assert pf["v"].patient_to_value == {1: 111, 2: None, 3: 333}
