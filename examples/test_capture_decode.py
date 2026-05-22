"""Offline tests for the raw-capture + decode pipeline.

Exercises ``capture_format`` (the binary container), ``capture_raw``
(stage 1) and ``decode_capture`` (stage 2) with no hardware: a
``FakeTransport`` pre-loaded with synthesized Init-Mode bytes and Normal-
Mode frames stands in for a STIM300, reusing the demo-traffic builders
from ``examples/demo_transport.py``.

Pytest does not auto-discover this file from the project root (the
project pins ``testpaths = ["tests"]``); run it explicitly::

    pytest examples/test_capture_decode.py -v
"""

import csv
import json
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pystim300 import FakeTransport
from pystim300.configuration import decode_configuration
from pystim300.datagrams import (
    PART_NUMBER_IDS,
    SERIAL_NUMBER_IDS,
    SerialNumberDatagram,
)

from examples.capture_format import (
    iter_records,
    read_header,
    write_header,
    write_record,
)
from examples.capture_raw import run_capture
from examples.decode_capture import _columns, decode_capture
from examples.demo_transport import (
    _build_special_frame,
    _demo_configuration_payload,
    _demo_part_number_payload,
    _demo_serial_number_payload,
    _make_demo_measurement_bytes,
)


def _init_bytes() -> bytes:
    """The Init-Mode datagram sequence the demo device emits at power-up."""
    return (
        _build_special_frame(PART_NUMBER_IDS[0], _demo_part_number_payload())
        + _build_special_frame(SERIAL_NUMBER_IDS[0], _demo_serial_number_payload())
        + _build_special_frame(0xBC, _demo_configuration_payload())
    )


# ---------------------------------------------------------------------------
# capture_format - the binary container
# ---------------------------------------------------------------------------

def test_capture_format_roundtrip(tmp_path):
    path = tmp_path / "fmt.cap"
    records = [(1.0, b"abc"), (1.5, b""), (2.25, b"\x00\x01\x02\x03")]
    with open(path, "wb") as f:
        write_header(f, epoch0=1700000000.5, perf0=10.0,
                     bit_rate=1843200, port="/dev/ttyUSB0")
        for perf, payload in records:
            write_record(f, perf, payload)

    with open(path, "rb") as f:
        header = read_header(f)
        got = list(iter_records(f))

    assert header.version == 1
    assert header.epoch0 == 1700000000.5
    assert header.perf0 == 10.0
    assert header.bit_rate == 1843200
    assert header.port == "/dev/ttyUSB0"
    assert got == records


def test_capture_format_tolerates_truncated_record(tmp_path):
    path = tmp_path / "trunc.cap"
    with open(path, "wb") as f:
        write_header(f, epoch0=1.0, perf0=0.0, bit_rate=0, port="")
        write_record(f, 1.0, b"good")
    # Simulate the writer being killed mid-record.
    with open(path, "ab") as f:
        f.write(b"\x00\x00\x00")

    with open(path, "rb") as f:
        read_header(f)
        got = list(iter_records(f))

    assert got == [(1.0, b"good")]


def test_read_header_rejects_non_capture_file(tmp_path):
    path = tmp_path / "bogus.cap"
    path.write_bytes(b"not a capture file at all")
    with open(path, "rb") as f:
        with pytest.raises(ValueError):
            read_header(f)


# ---------------------------------------------------------------------------
# capture_raw + decode_capture - the full pipeline
# ---------------------------------------------------------------------------

def test_capture_decode_roundtrip(tmp_path):
    n_frames = 60
    max_records = 50
    init = _init_bytes()
    meas = _make_demo_measurement_bytes(n_frames)
    transport = FakeTransport(initial=init + meas)

    cap = tmp_path / "run.cap"
    phase_a, phase_b, total = run_capture(
        transport, cap, bit_rate=1843200, port="fake://demo",
        reset_first=False, max_records=max_records)

    assert phase_b == max_records
    assert phase_a > 0
    assert total > 0

    # The .cap round-trips: header intact, records reproduce the stream.
    with open(cap, "rb") as f:
        header = read_header(f)
        recs = list(iter_records(f))
    assert header.bit_rate == 1843200
    assert header.port == "fake://demo"
    assert len(recs) == phase_a + phase_b
    captured = b"".join(payload for _, payload in recs)
    assert captured == (init + meas)[:len(captured)]

    # Decode to CSV + meta sidecar.
    csv_path = tmp_path / "run.csv"
    summary = decode_capture(cap, csv_path)
    assert summary["rows"] == max_records
    assert summary["resync_events"] == 0
    assert summary["dropped_frames"] == 0

    cfg = decode_configuration(_demo_configuration_payload())
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == _columns(cfg)
    data_rows = rows[1:]
    assert len(data_rows) == max_records

    # host_monotonic is non-decreasing; cluster columns parse as floats.
    monotonics = [float(r[0]) for r in data_rows]
    assert monotonics == sorted(monotonics)
    for r in data_rows:
        for value in r[6:9]:   # gyro_x / gyro_y / gyro_z
            float(value)

    meta = json.loads((tmp_path / "run.csv.meta.json").read_text())
    sn = SerialNumberDatagram.parse(_demo_serial_number_payload())
    assert meta["serial_number"] == sn.serial_number
    assert meta["units"]["gyro"] in ("deg/s", "deg")
    assert meta["units"]["temperature"] == "degC"
    assert meta["columns"] == _columns(cfg)
    assert meta["configuration"]["frame_length_bytes"] == cfg.frame_length()


def test_decode_requires_init_sequence(tmp_path):
    cap = tmp_path / "noinit.cap"
    with open(cap, "wb") as f:
        write_header(f, epoch0=1.0, perf0=0.0, bit_rate=0, port="")
        write_record(f, 1.0, _make_demo_measurement_bytes(5))

    with pytest.raises(ValueError):
        decode_capture(cap, tmp_path / "noinit.csv")
