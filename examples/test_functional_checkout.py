"""Pytest recasting of the functional-checkout demo.

Each ``check_*`` primitive becomes its own test function. A single
module-scoped fixture drives the device through every phase once (Init
capture, measurement window, Service / Utility round-trips, Extended
Error trigger); the individual tests then inspect the captured data.

This is a sibling to ``examples/functional_checkout.py`` - it reuses
that module's synthesized-traffic helpers verbatim so the two demos
exercise identical bytes on the wire. Procedural form prints a summary;
pytest form gives a granular per-check pass/fail surface, free test
selection via ``-k``, and clean skip semantics for absent clusters.

Pytest does not auto-discover this file from the project root (the
project pins ``testpaths = ["tests"]``); run it explicitly::

    pytest examples/test_functional_checkout.py -v

For a single check::

    pytest examples/test_functional_checkout.py -k gravity -v

To enumerate without running::

    pytest examples/test_functional_checkout.py --collect-only -q
"""

import pathlib
import sys
from types import SimpleNamespace

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pystim300 import (
    ExtendedErrorDatagram,
    MemoryAuditor,
    Mode,
    STIM300,
    check_bias_trim_present,
    check_configuration,
    check_extended_error_clean,
    check_frame_rate,
    check_gravity_direction_consistent,
    check_gravity_magnitude,
    check_gyro_quiescent,
    check_inclinometer_gravity,
    check_latency_within,
    check_no_dropped_frames,
    check_parser_clean,
    check_part_number,
    check_serial_number,
    check_service_round_trip,
    check_status_bytes_clean,
    check_temperature_range,
    check_utility_round_trip,
)

# Re-use the synthesized FakeTransport + ExpectedConfiguration from the
# procedural demo so the two checkouts exercise identical bytes on the
# wire. ``functional_checkout`` is importable as a module because its
# ``main()`` is guarded by ``if __name__ == "__main__"``.
from examples.functional_checkout import (
    _build_demo_transport,
    _demo_expected_configuration,
)


MEASUREMENT_COUNT = 200
SAMPLE_RATE_HZ = 2000
SYNTHETIC_DURATION = MEASUREMENT_COUNT / SAMPLE_RATE_HZ

# Fields the demo's ``ExpectedConfiguration`` sets - mirrors the set
# ``check_configuration`` will report on. Drift between this tuple and
# the demo's expected configuration is caught by
# ``test_configuration_field_coverage`` below.
_DEMO_CONFIG_FIELDS = (
    "sample_rate_hz",
    "bit_rate",
    "has_acceleration",
    "has_inclination",
    "has_temperature",
    "has_pps",
    "crlf_termination",
    "bias_trim_at_startup",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def expected():
    """The demo's ``ExpectedConfiguration``."""
    return _demo_expected_configuration()


@pytest.fixture(scope="module")
def checkout_session(expected):
    """Drive the full procedural checkout once; expose the captured data.

    The returned namespace carries:

        * ``client``           - the STIM300 instance after the procedure
        * ``init``             - the InitSequence captured at power-on
        * ``measurements``     - the quiescent-IMU Measurement list
        * ``duration_seconds`` - synthetic duration (FakeTransport is
                                  instant; real harnesses use wall-clock)
        * ``service_audit``    - the audit-event slice for the Service
                                  round-trip
        * ``isn_response``     - the UtilityResponse from ``$isn``
        * ``eed``              - the ExtendedErrorDatagram from ``E``
        * ``auditor``          - the full MemoryAuditor for inspection
    """
    transport = _build_demo_transport(measurement_count=MEASUREMENT_COUNT)
    auditor = MemoryAuditor()
    client = STIM300(transport, audit=auditor, timeout=5.0)
    client.set_mode(Mode.INIT)

    init = client.read_init_sequence(timeout=5.0)
    measurements = list(client.read_measurements(
        limit=MEASUREMENT_COUNT, include_startup=False))

    service_audit_start = len(auditor)
    client.enter_service()
    client.service_command("i d")
    client.exit_service()
    service_audit = auditor.events[service_audit_start:]

    client.enter_utility()
    isn_response = client.utility_command("isn")
    client.exit_utility()

    client.request_extended_error()
    eed = next(
        (r for r in client.read_records(limit=10, timeout=5.0)
         if isinstance(r, ExtendedErrorDatagram)),
        None,
    )

    return SimpleNamespace(
        client=client,
        init=init,
        measurements=measurements,
        duration_seconds=SYNTHETIC_DURATION,
        service_audit=service_audit,
        isn_response=isn_response,
        eed=eed,
        auditor=auditor,
    )


# ---------------------------------------------------------------------------
# Identity & configuration
# ---------------------------------------------------------------------------

def test_part_number(checkout_session, expected):
    r = check_part_number(checkout_session.init.part_number, expected.part_number)
    assert r.passed, r.detail


def test_serial_number(checkout_session, expected):
    r = check_serial_number(
        checkout_session.init.serial_number, expected.serial_numbers)
    assert r.passed, r.detail


@pytest.mark.parametrize("field_name", _DEMO_CONFIG_FIELDS)
def test_configuration_field(checkout_session, expected, field_name):
    results = check_configuration(checkout_session.init.configuration, expected)
    by_name = {r.name: r for r in results}
    target = "configuration.{0}".format(field_name)
    assert target in by_name, "field not checked by check_configuration: {0}".format(target)
    r = by_name[target]
    assert r.passed, r.detail


def test_configuration_field_coverage(checkout_session, expected):
    """Catches drift: every field the demo sets must have a parametrize case."""
    results = check_configuration(checkout_session.init.configuration, expected)
    reported = {r.name for r in results}
    expected_names = {"configuration.{0}".format(f) for f in _DEMO_CONFIG_FIELDS}
    missing = expected_names - reported
    extra = reported - expected_names
    assert not missing and not extra, \
        "parametrize list out of sync; missing={0}, extra={1}".format(missing, extra)


def test_bias_trim_present(checkout_session):
    r = check_bias_trim_present(
        checkout_session.init.bias_trim, checkout_session.init.configuration)
    assert r.passed, r.detail


# ---------------------------------------------------------------------------
# Stream health
# ---------------------------------------------------------------------------

def test_parser_clean(checkout_session):
    assert checkout_session.client.stream_parser is not None
    r = check_parser_clean(checkout_session.client.stream_parser)
    assert r.passed, r.detail


def test_no_dropped_frames(checkout_session):
    r = check_no_dropped_frames(checkout_session.measurements)
    assert r.passed, r.detail


def test_frame_rate(checkout_session, expected):
    cfg = checkout_session.init.configuration
    if not cfg.sample_rate_hz:
        pytest.skip("sample_rate_hz is 0 (external trigger); rate check N/A")
    r = check_frame_rate(
        checkout_session.measurements,
        checkout_session.duration_seconds,
        cfg.sample_rate_hz,
        tolerance_pct=expected.sample_rate_tolerance_pct,
    )
    assert r.passed, r.detail


def test_latency_within(checkout_session, expected):
    if expected.max_latency_us is None:
        pytest.skip("no max_latency_us configured")
    r = check_latency_within(checkout_session.measurements, expected.max_latency_us)
    assert r.passed, r.detail


def test_status_bytes_clean(checkout_session):
    r = check_status_bytes_clean(
        checkout_session.measurements,
        startup_frame_allowance=0,
    )
    assert r.passed, r.detail


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def test_temperature_range(checkout_session, expected):
    if not checkout_session.init.configuration.has_temperature:
        pytest.skip("no temperature cluster in configuration")
    r = check_temperature_range(
        checkout_session.measurements, expected.temp_min_c, expected.temp_max_c)
    assert r.passed, r.detail


def test_gyro_quiescent(checkout_session, expected):
    r = check_gyro_quiescent(
        checkout_session.measurements,
        expected.gyro_max_mean_dps, expected.gyro_max_std_dps)
    assert r.passed, r.detail


def test_gravity_magnitude(checkout_session, expected):
    if not checkout_session.init.configuration.has_acceleration:
        pytest.skip("no acceleration cluster in configuration")
    r = check_gravity_magnitude(
        checkout_session.measurements,
        expected.gravity_magnitude_g, expected.gravity_tolerance_g)
    assert r.passed, r.detail


def test_gravity_direction_consistent(checkout_session, expected):
    if not checkout_session.init.configuration.has_acceleration:
        pytest.skip("no acceleration cluster in configuration")
    r = check_gravity_direction_consistent(
        checkout_session.measurements, expected.gravity_direction_std_g)
    assert r.passed, r.detail


def test_inclinometer_gravity(checkout_session, expected):
    if not checkout_session.init.configuration.has_inclination:
        pytest.skip("no inclination cluster in configuration")
    r = check_inclinometer_gravity(
        checkout_session.measurements,
        expected.gravity_magnitude_g, expected.gravity_tolerance_g)
    assert r.passed, r.detail


# ---------------------------------------------------------------------------
# Service / Utility round-trips
# ---------------------------------------------------------------------------

def test_service_round_trip(checkout_session):
    r = check_service_round_trip(checkout_session.service_audit)
    assert r.passed, r.detail


def test_utility_round_trip(checkout_session):
    r = check_utility_round_trip(
        checkout_session.isn_response, checkout_session.init.serial_number)
    assert r.passed, r.detail


# ---------------------------------------------------------------------------
# Extended Error
# ---------------------------------------------------------------------------

def test_extended_error_clean(checkout_session, expected):
    r = check_extended_error_clean(
        checkout_session.eed, ignore_bits=expected.extended_error_ignore_bits)
    assert r.passed, r.detail
