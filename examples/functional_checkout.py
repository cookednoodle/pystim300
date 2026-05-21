"""Functional-checkout demo wiring the pystim300 primitives end-to-end.

Runs the full checkout sequence - Init-Mode capture, a quiescent
measurement window, Service-/Utility-Mode round-trips, and an Extended
Error request - then prints and optionally writes a JSON report.

Two transports are supported:

* **Synthetic demo** (default, no ``--port``): a ``FakeTransport``
  pre-loaded with synthesized device traffic from
  ``examples/demo_transport``. Runs anywhere, no hardware needed.
* **Real hardware** (``--port``): a ``SerialTransport`` talking to an
  actual STIM300. By default the checkout issues a commanded reset
  (``R``) so the unit re-emits its Init-Mode sequence regardless of how
  long it has been running; ``--no-reset`` instead waits for the
  operator to power-cycle the unit.

The structure of ``run_checkout`` is intended to be copied and adapted
by a downstream harness (substituting its own clock, transport, and
``ExpectedConfiguration``); the individual ``check_*`` primitives from
``pystim300.checkout`` are stable building blocks.

Usage::

    # Synthetic demo (no hardware):
    python examples/functional_checkout.py [--out report.json]

    # Real STIM300 on a serial port:
    python examples/functional_checkout.py --port /dev/ttyUSB0 [--no-reset]
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
from pystim300.transport import SerialTransport, Transport

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
                  reset_first: bool = False,
                  init_timeout: float = 5.0,
                  report_path: Optional[pathlib.Path] = None,
                  ) -> CheckoutReport:
    """Run the full functional-checkout procedure against ``transport``.

    The transport is assumed to be connected to a STIM300. With
    ``reset_first`` the checkout issues a commanded reset (``R``) so the
    device re-emits its Init-Mode sequence; otherwise the device is
    assumed to have just been power-cycled. Power control is the
    caller's responsibility.

    Steps:

        1. (Optionally reset, then) drain Init-Mode datagrams
           (PN + SN + CFG [+ BT]).
        2. Check identity and configuration against ``expected``.
        3. Collect ``measurement_count`` quiescent frames.
        4. Run stream-health + physics checks on the frames.
        5. Round-trip Service Mode (`i d`) and Utility Mode (`$isn`).
        6. Request Extended Error Information and check it is clean.

    Returns a ``CheckoutReport`` with one ``CheckResult`` per check.
    """
    auditor = MemoryAuditor()
    client = STIM300(transport, audit=auditor, timeout=init_timeout)
    checks: List[CheckResult] = []

    # --- 1. Init-Mode capture ----------------------------------------
    # ``reset()`` requires NORMAL/UNKNOWN mode; a freshly-constructed
    # client is UNKNOWN, so issue the reset before forcing any mode.
    if reset_first:
        client.reset()
    else:
        client.set_mode(Mode.INIT)
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


def hardware_expected_configuration() -> ExpectedConfiguration:
    """Expected values + thresholds for a real STIM300 on the bench.

    A per-unit, per-environment starting point - edit to match the unit
    under test. Identity fields are left ``None`` so the part/serial
    number are reported but not gated (they are unit-specific); set
    ``serial_numbers`` to enforce a particular unit. Structural fields
    match a standard STIM300 streaming configuration. Physics thresholds
    are loosened from the synthetic-demo values to tolerate real benchtop
    noise, vibration, and ambient temperature.
    """
    return ExpectedConfiguration(
        # Identity - reported, not gated. Set serial_numbers to enforce.
        part_number=None,
        serial_numbers=None,
        # Structural cross-check - standard streaming configuration.
        sample_rate_hz=2000,
        bit_rate=1843200,
        has_acceleration=True,
        has_inclination=True,
        has_temperature=True,
        has_pps=False,
        crlf_termination=False,
        bias_trim_at_startup=False,
        # Stream / framing.
        max_latency_us=2000,
        sample_rate_tolerance_pct=5.0,
        # Physics - loosened for a real, still unit on a bench.
        temp_min_c=0.0,
        temp_max_c=60.0,
        gyro_max_mean_dps=1.0,
        gyro_max_std_dps=2.0,
        gravity_magnitude_g=1.0,
        gravity_tolerance_g=0.1,
        gravity_direction_std_g=0.05,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="Write the JSON report to this path.")
    parser.add_argument("--port", default=None,
                        help="Serial port of a real STIM300 (e.g. /dev/ttyUSB0). "
                             "Omit to run the synthetic demo.")
    parser.add_argument("--bit-rate", type=int, default=1843200,
                        help="Serial bit rate, real hardware only "
                             "(STIM300 factory default 1843200).")
    parser.add_argument("--no-reset", action="store_true",
                        help="Real hardware only: skip the commanded reset and "
                             "wait for the operator to power-cycle the unit.")
    parser.add_argument("--measurements", type=int, default=None,
                        help="Quiescent frames to collect "
                             "(default 200 for the demo, 2000 for real hardware).")
    args = parser.parse_args(argv)

    if args.port is None:
        # --- Synthetic demo ------------------------------------------
        count = args.measurements if args.measurements is not None else 200
        transport = build_demo_transport(measurement_count=count)
        # The FakeTransport delivers all bytes instantly, so use a
        # synthetic duration matching the configured sample rate for the
        # frame-rate check. Real hardware omits this and uses wall-clock.
        report = run_checkout(
            transport,
            demo_expected_configuration(),
            measurement_count=count,
            measurement_duration=count / 2000.0,
            report_path=args.out,
        )
        return 0 if report.passed() else 1

    # --- Real hardware ----------------------------------------------
    count = args.measurements if args.measurements is not None else 2000
    reset_first = not args.no_reset
    if reset_first:
        print("Connecting to {0}; issuing commanded reset (R)...".format(args.port))
    else:
        print("Connecting to {0}; power-cycle the STIM300 now - "
              "waiting for Init-Mode datagrams...".format(args.port))
    with SerialTransport(args.port, bit_rate=args.bit_rate) as transport:
        report = run_checkout(
            transport,
            hardware_expected_configuration(),
            measurement_count=count,
            reset_first=reset_first,
            init_timeout=20.0,
            report_path=args.out,
        )
    return 0 if report.passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
