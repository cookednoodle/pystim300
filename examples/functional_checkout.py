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
    FakeTransport,
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
from pystim300.crc import crc32_stim300, crc8_stim300
from pystim300.datagrams import (
    BIAS_TRIM_IDS,
    BIAS_TRIM_PAYLOAD_LENGTH,
    EXTENDED_ERROR_IDS,
    EXTENDED_ERROR_PAYLOAD_LENGTH,
    PART_NUMBER_IDS,
    PART_NUMBER_PAYLOAD_LENGTH,
    SERIAL_NUMBER_IDS,
    SERIAL_NUMBER_PAYLOAD_LENGTH,
)
from pystim300.normal import (
    SPECIAL_SPECS,
    build_normal_frame,
)
from pystim300.transport import Transport


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


# ---------------------------------------------------------------------------
# Demo: build a FakeTransport with synthesized device traffic and run the
# checkout against it.
# ---------------------------------------------------------------------------

# Configuration payload byte layout (Table 5-16, p.31). The demo uses a
# datagram with acceleration + inclination + temperature, sample rate
# 2000 Hz, no PPS, no CRLF, bias_trim_at_startup=False, 1843200 baud.
_DEMO_PART_NUMBER = "8420005000000-A"
_DEMO_SERIAL_NUMBER = "12345678901234"


def _build_special_frame(datagram_id: int, payload: bytes) -> bytes:
    spec = SPECIAL_SPECS[datagram_id]
    assert len(payload) == spec.payload_length, \
        "payload length {0} != spec {1}".format(len(payload), spec.payload_length)
    id_and_payload = bytes([datagram_id]) + payload
    crc = crc32_stim300(id_and_payload + b"\x00" * spec.dummy_bytes)
    frame = id_and_payload + crc.to_bytes(4, "big")
    if datagram_id in {PART_NUMBER_IDS[1], SERIAL_NUMBER_IDS[1], 0xBD,
                       BIAS_TRIM_IDS[1], EXTENDED_ERROR_IDS[1]}:
        frame += b"\r\n"
    return frame


def _demo_part_number_payload() -> bytes:
    # Pack "8420005000000-A" into the nibble layout of Table 5-14.
    # Easier: just zero the payload; the demo doesn't assert the decoded
    # part_number, only that the datagram itself is captured.
    return bytes(PART_NUMBER_PAYLOAD_LENGTH)


def _demo_serial_number_payload() -> bytes:
    # Bytes: 'N' followed by 7 BCD bytes encoding "12345678901234".
    payload = bytearray(SERIAL_NUMBER_PAYLOAD_LENGTH)
    payload[0] = ord("N")
    payload[1] = 0x12
    payload[2] = 0x34
    payload[3] = 0x56
    payload[4] = 0x78
    payload[5] = 0x90
    payload[6] = 0x12
    payload[7] = 0x34
    return bytes(payload)


def _demo_configuration_payload() -> bytes:
    """Hand-built Configuration payload matching the demo's intent.

    See Table 5-16 (pp.31-33) for the bit layout. The decoded values
    cross-checked by the demo's ExpectedConfiguration are:

        revision_char='-', sample_rate_hz=2000, bit_rate=1843200,
        has_acceleration=True, has_inclination=True, has_temperature=True,
        has_pps=False, crlf_termination=False, bias_trim_at_startup=False.
    """
    p = bytearray(21)
    p[0] = ord("-")                                  # revision_char
    p[1] = 31                                         # firmware_revision
    # Byte 3: sample rate code 4 (2000Hz)<<5 | has_pps=0 | has_temp=1 | has_incl=1 | has_accel=1 | crlf=0
    p[2] = (0b100 << 5) | 0x08 | 0x04 | 0x02         # = 0x8E
    # Byte 4: bit_rate code 3 (1843200) << 4 | stop_bits=1(bit3=0) | parity=00 | line_term=0
    p[3] = (0b0011 << 4)                              # = 0x30
    # Bytes 5-7 (gyro): all axes active, unit 0 (angular rate), LP filter 4 (262 Hz)
    p[4] = 0x70                                       # axes XYZ active, unit 0
    p[5] = (0b100 << 4) | 0b100                       # X+Y LP filter = 262
    p[6] = (0b100 << 4) | 0                           # Z LP filter = 262, g-comp = 0
    # Bytes 8-10 (accel): all axes, unit 0 ([g]), LP filter 4
    p[7] = 0x70
    p[8] = (0b100 << 4) | 0b100
    p[9] = (0b100 << 4)
    # Bytes 11-13 (incl): all axes, unit 0 ([g]), LP filter 4
    p[10] = 0x70
    p[11] = (0b100 << 4) | 0b100
    p[12] = (0b100 << 4) | 0                          # incl Z LP=262, PPS unit 0
    # Byte 14: has_aux_input=0 (so has_pps_input bit is 0), PPS LP filter = 4
    p[13] = 0b100                                     # has_pps_input=0 -> has_aux_input=True
    # Bytes 15-20: ranges. accel_range_g code 0 = 10g (high+low nibble per axis).
    p[16] = 0x00                                      # X (high) + Y (low) = 10g
    p[17] = 0x00                                      # Z (high)
    # Byte 21: TOV logic 5.0V (bit3=0), tov_toggling=0, bias_trim_at_startup=0
    p[20] = 0
    return bytes(p)


def _demo_extended_error_payload() -> bytes:
    """All-zero payload -> error_bits == 0 -> all checks pass."""
    return bytes(EXTENDED_ERROR_PAYLOAD_LENGTH)


def _make_demo_measurement_bytes(num_frames: int) -> bytes:
    """Synthesize ``num_frames`` of quiescent Normal-Mode bytes.

    Uses a freshly-decoded Configuration that matches what the
    Configuration datagram in the Init sequence decodes to. The
    measurements have:

        - gyro:  (0.01, -0.02, 0.005) deg/s (within quiescence bounds)
        - accel: (0, 0, 1) g            -> gravity along +Z
        - incl:  (0, 0, 1) g
        - temp:  25 degC across all clusters
        - counter wraps at 256 modulo
    """
    # Build a Configuration by parsing the synthesized payload, so the
    # decoded configuration here is bit-for-bit identical to what the
    # checkout will receive.
    from pystim300.configuration import decode_configuration
    cfg = decode_configuration(_demo_configuration_payload())

    from pystim300.normal import Measurement
    from pystim300.status import StatusByte
    sb = StatusByte.decode(0)
    out = bytearray()
    for i in range(num_frames):
        m = Measurement(
            datagram_id=cfg.datagram_id,
            counter=i % 256,
            latency_us=1500,
            gyro=(0.01, -0.02, 0.005),
            gyro_status=sb,
            accel=(0.0, 0.0, 1.0),
            accel_status=sb,
            incl=(0.0, 0.0, 1.0),
            incl_status=sb,
            gyro_temp=(25.0, 25.0, 25.0),
            gyro_temp_status=sb,
            accel_temp=(25.0, 25.0, 25.0),
            accel_temp_status=sb,
            incl_temp=(25.0, 25.0, 25.0),
            incl_temp_status=sb,
        )
        out.extend(build_normal_frame(m, cfg))
    return bytes(out)


def _utility_reply(body_no_crc: str) -> bytes:
    """Build a Utility-Mode reply bytes from the body up to and including the trailing comma."""
    crc = crc8_stim300(body_no_crc.encode("ascii"))
    return (body_no_crc + str(crc) + "\r").encode("ascii")


def _build_demo_transport(measurement_count: int) -> FakeTransport:
    """Construct a FakeTransport pre-loaded with everything the checkout needs."""
    # Initial bytes: Init-Mode sequence + N quiescent frames.
    init_bytes = (
        _build_special_frame(PART_NUMBER_IDS[0], _demo_part_number_payload())
        + _build_special_frame(SERIAL_NUMBER_IDS[0], _demo_serial_number_payload())
        + _build_special_frame(0xBC, _demo_configuration_payload())
        # No bias-trim datagram (bias_trim_at_startup=False).
    )
    measurement_bytes = _make_demo_measurement_bytes(measurement_count)

    # Scripted writes: each Service/Utility/E request triggers its reply.
    service_banner = b"PRODUCT = STIM300\rREV = -\r>"
    i_d_reply = b"datagram = a7,no\r>"
    exit_service_reply = b"\r>"
    utility_ack = _utility_reply("#UTILITYMODE,")
    isn_reply = _utility_reply("#isn,0,{0},".format(_DEMO_SERIAL_NUMBER))
    xn_reply = _utility_reply("#xn,0,")
    extended_error_frame = _build_special_frame(
        EXTENDED_ERROR_IDS[0], _demo_extended_error_payload())

    scripted = [
        (b"SERVICEMODE\r", service_banner),
        (b"i d\r", i_d_reply),
        (b"x 1\r", exit_service_reply),
        (b"UTILITYMODE\r", utility_ack),
        (b"$isn,", isn_reply),
        (b"$xn,", xn_reply),
        (b"E\r", extended_error_frame),
    ]
    return FakeTransport(initial=init_bytes + measurement_bytes, scripted=scripted)


def _demo_expected_configuration() -> ExpectedConfiguration:
    return ExpectedConfiguration(
        serial_numbers=frozenset({_DEMO_SERIAL_NUMBER}),
        sample_rate_hz=2000,
        bit_rate=1843200,
        has_acceleration=True,
        has_inclination=True,
        has_temperature=True,
        has_pps=False,
        crlf_termination=False,
        bias_trim_at_startup=False,
        max_latency_us=2000,
        sample_rate_tolerance_pct=5.0,        # demo runs much faster than real-time
        temp_min_c=0.0,
        temp_max_c=50.0,
        gyro_max_mean_dps=0.5,
        gyro_max_std_dps=2.0,
        gravity_magnitude_g=1.0,
        gravity_tolerance_g=0.05,
        gravity_direction_std_g=0.01,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="Write the JSON report to this path.")
    parser.add_argument("--measurements", type=int, default=200,
                        help="How many quiescent frames to synthesize and collect.")
    args = parser.parse_args(argv)

    transport = _build_demo_transport(measurement_count=args.measurements)
    # The FakeTransport delivers all bytes instantly, so use a synthetic
    # duration matching the configured sample rate for the frame-rate
    # check. A real harness omits this argument and uses wall-clock time.
    synthetic_duration = args.measurements / 2000.0
    report = run_checkout(
        transport,
        _demo_expected_configuration(),
        measurement_count=args.measurements,
        measurement_duration=synthetic_duration,
        report_path=args.out,
    )
    return 0 if report.passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
