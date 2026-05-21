"""Functional-checkout demo wiring the pystim300 primitives end-to-end.

Drives ``pystim300.checkout`` against a ``FakeTransport`` pre-loaded with
synthesized Init-Mode bytes, quiescent Normal-Mode frames, Service-Mode
and Utility-Mode responses, and a clean Extended Error datagram. Runs
the full checkout sequence and prints + writes the report.

This is a demonstration of the recommended composition pattern. A real
hardware test harness sits downstream of this repo and:

    1. Controls power to the unit (out of scope here).
    2. Opens a ``SerialTransport`` once the unit has powered up.
    3. Calls ``run_checkout(transport, expected, report_path=...)``.

The structure of ``run_checkout`` below is intended to be copied and
adapted (the harness substitutes its own clock, transport, and
``ExpectedConfiguration``); the individual ``check_*`` primitives from
``pystim300.checkout`` are stable building blocks.

The synthesized-traffic scaffolding lives in ``examples/demo_transport``.

Usage (demo / smoke check)::

    python examples/functional_checkout.py [--out report.json]
"""

import argparse
import pathlib
import sys
import time
from typing import List, Optional

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pystim300 import (
    AuditEvent,
    CheckoutReport,
    ExpectedConfiguration,
    ExtendedErrorDatagram,
    InitSequence,
    Measurement,
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
from pystim300.checkout import CheckResult
from pystim300.transport import Transport

from examples.demo_transport import (
    build_demo_transport,
    demo_expected_configuration,
)


# ---------------------------------------------------------------------------
# Core composition: this is the function downstream harnesses call.
# ---------------------------------------------------------------------------

def run_checkout(transport: Transport,
                  expected: ExpectedConfiguration,
                  *,
                  measurement_count: int = 200,
                  measurement_duration: Optional[float] = None,
                  startup_frame_allowance: int = 0,
                  service_round_trip: bool = True,
                  utility_round_trip: bool = True,
                  read_extended_error: bool = True,
                  init_timeout: float = 5.0,
                  report_path: Optional[pathlib.Path] = None,
                  ) -> CheckoutReport:
    """Run the full functional-checkout procedure against ``transport``.

    The transport is assumed to be connected to a STIM300 that has just
    been powered up. Power control is the caller's responsibility.

    Steps:

        1. Drain Init-Mode datagrams (PN + SN + CFG [+ BT]).
        2. Check identity and configuration against ``expected``.
        3. Collect ``measurement_count`` quiescent frames.
        4. Run stream-health + physics checks on the frames.
        5. Round-trip Service Mode (`i d`) and Utility Mode (`$isn`).
        6. Request Extended Error Information and check it is clean.

    Returns a ``CheckoutReport`` with one ``CheckResult`` per check.
    """
    auditor = MemoryAuditor()
    client = STIM300(transport, audit=auditor, timeout=init_timeout)
    client.set_mode(Mode.INIT)
    checks: List[CheckResult] = []

    # --- 1. Init-Mode capture ----------------------------------------
    init = client.read_init_sequence(timeout=init_timeout)
    checks.append(check_part_number(init.part_number, expected.part_number))
    checks.append(check_serial_number(init.serial_number, expected.serial_numbers))
    checks.extend(check_configuration(init.configuration, expected))
    checks.append(check_bias_trim_present(init.bias_trim, init.configuration))

    # --- 2. Quiescent measurement window ------------------------------
    # ``measurement_duration`` lets the caller substitute a synthetic
    # duration when the transport is not real-time (e.g. ``FakeTransport``
    # in this demo). On real hardware, leave it None to use wall-clock.
    start = time.monotonic()
    measurements: List[Measurement] = list(client.read_measurements(
        limit=measurement_count, include_startup=False))
    duration = (measurement_duration if measurement_duration is not None
                  else time.monotonic() - start)

    # --- 3. Stream + physics checks -----------------------------------
    assert client.stream_parser is not None
    checks.append(check_parser_clean(client.stream_parser))
    checks.append(check_no_dropped_frames(measurements))
    if init.configuration.sample_rate_hz:
        checks.append(check_frame_rate(
            measurements, duration, init.configuration.sample_rate_hz,
            tolerance_pct=expected.sample_rate_tolerance_pct))
    if expected.max_latency_us is not None:
        checks.append(check_latency_within(measurements, expected.max_latency_us))
    checks.append(check_status_bytes_clean(
        measurements, startup_frame_allowance=startup_frame_allowance))
    if init.configuration.has_temperature:
        checks.append(check_temperature_range(
            measurements, expected.temp_min_c, expected.temp_max_c))
    checks.append(check_gyro_quiescent(
        measurements, expected.gyro_max_mean_dps, expected.gyro_max_std_dps))
    if init.configuration.has_acceleration:
        checks.append(check_gravity_magnitude(
            measurements, expected.gravity_magnitude_g, expected.gravity_tolerance_g))
        checks.append(check_gravity_direction_consistent(
            measurements, expected.gravity_direction_std_g))
    if init.configuration.has_inclination:
        checks.append(check_inclinometer_gravity(
            measurements, expected.gravity_magnitude_g, expected.gravity_tolerance_g))

    # --- 4. Service-Mode round-trip -----------------------------------
    if service_round_trip:
        service_start = len(auditor)
        client.enter_service()
        client.service_command("i d")
        client.exit_service()
        checks.append(check_service_round_trip(auditor.events[service_start:]))

    # --- 5. Utility-Mode round-trip -----------------------------------
    if utility_round_trip:
        client.enter_utility()
        isn = client.utility_command("isn")
        client.exit_utility()
        checks.append(check_utility_round_trip(isn, init.serial_number))

    # --- 6. Extended Error -------------------------------------------
    eed: Optional[ExtendedErrorDatagram] = None
    if read_extended_error:
        client.request_extended_error()
        for record in client.read_records(limit=10, timeout=init_timeout):
            if isinstance(record, ExtendedErrorDatagram):
                eed = record
                break
        checks.append(check_extended_error_clean(
            eed, ignore_bits=expected.extended_error_ignore_bits))

    assert client.stream_parser is not None
    report = CheckoutReport(
        checks=tuple(checks),
        init_sequence=init,
        measurement_count=len(measurements),
        duration_seconds=duration,
        parser_resync_events=client.stream_parser.resync_events,
        parser_bytes_discarded=client.stream_parser.bytes_discarded,
        audit=auditor.events,
    )
    print(report.summary())
    if report_path is not None:
        report_path.write_text(report.to_json())
        print()
        print("Report written to {0}".format(report_path))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="Write the JSON report to this path.")
    parser.add_argument("--measurements", type=int, default=200,
                        help="How many quiescent frames to synthesize and collect.")
    args = parser.parse_args(argv)

    transport = build_demo_transport(measurement_count=args.measurements)
    # The FakeTransport delivers all bytes instantly, so use a synthetic
    # duration matching the configured sample rate for the frame-rate
    # check. A real harness omits this argument and uses wall-clock time.
    synthetic_duration = args.measurements / 2000.0
    report = run_checkout(
        transport,
        demo_expected_configuration(),
        measurement_count=args.measurements,
        measurement_duration=synthetic_duration,
        report_path=args.out,
    )
    return 0 if report.passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
