"""Tests for the functional-checkout primitives in ``pystim300.checkout``."""

import json

import pytest

from pystim300.checkout import (
    CheckResult,
    CheckoutReport,
    ExpectedConfiguration,
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
from pystim300.client import AuditEvent, InitSequence, Mode
from pystim300.datagrams import (
    BIAS_TRIM_PAYLOAD_LENGTH,
    BiasTrimDatagram,
    EXTENDED_ERROR_PAYLOAD_LENGTH,
    ExtendedErrorDatagram,
    PART_NUMBER_PAYLOAD_LENGTH,
    PartNumberDatagram,
    SERIAL_NUMBER_PAYLOAD_LENGTH,
    SerialNumberDatagram,
)
from pystim300.normal import NormalStreamParser
from pystim300.utility import UtilityResponse

from conftest import make_configuration, make_measurement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pn(text: str = "8420005000000-A") -> PartNumberDatagram:
    """A PartNumberDatagram whose decoded part_number is ``text`` (best-effort)."""
    # Reverse-engineering the nibble packing is not worth it for tests; we
    # construct the dataclass directly with the desired string.
    return PartNumberDatagram(
        raw_payload=bytes(PART_NUMBER_PAYLOAD_LENGTH),
        part_number=text,
        revision="A",
    )


def _make_sn(text: str = "12345678901234") -> SerialNumberDatagram:
    return SerialNumberDatagram(
        raw_payload=bytes(SERIAL_NUMBER_PAYLOAD_LENGTH),
        serial_number=text,
    )


def _make_bt() -> BiasTrimDatagram:
    return BiasTrimDatagram(
        raw_payload=bytes(BIAS_TRIM_PAYLOAD_LENGTH),
        gyro_offset=(0.0, 0.0, 0.0),
        accel_offset=(0.0, 0.0, 0.0),
        incl_offset=(0.0, 0.0, 0.0),
        reference_info=0,
        remaining_saves=0,
    )


def _quiescent_measurements(cfg, *, n=200, gravity_axis="z",
                              latency_us=1000, status_raw=0x00):
    """Build a sequence of still-IMU measurements."""
    g_vec = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[gravity_axis]
    out = []
    for i in range(n):
        # Wrap counter at 256 to mimic the device's modular counter.
        out.append(make_measurement(
            cfg,
            gyro=(0.01, -0.02, 0.005),
            accel=g_vec,
            incl=g_vec,
            gyro_temp=(25.0, 25.0, 25.0),
            accel_temp=(25.0, 25.0, 25.0),
            incl_temp=(25.0, 25.0, 25.0),
            pps=0.5,
            counter=i % 256,
            latency_us=latency_us,
            status_raw=status_raw,
        ))
    return out


# ---------------------------------------------------------------------------
# Identity & configuration checks
# ---------------------------------------------------------------------------

class TestIdentityChecks:
    def test_part_number_match(self):
        pn = _make_pn("8420005000000-A")
        result = check_part_number(pn, "8420005000000-A")
        assert result.passed is True
        assert "matches" in result.detail.lower()

    def test_part_number_mismatch(self):
        pn = _make_pn("8420005000000-A")
        result = check_part_number(pn, "OTHER")
        assert result.passed is False
        assert "OTHER" in result.detail

    def test_part_number_no_expected_passes_with_note(self):
        pn = _make_pn("ANYTHING")
        result = check_part_number(pn, None)
        assert result.passed is True
        assert "not checked" in result.detail

    def test_serial_number_in_allowed_set(self):
        sn = _make_sn("12345678901234")
        result = check_serial_number(sn, frozenset({"99999999999999", "12345678901234"}))
        assert result.passed is True

    def test_serial_number_not_in_allowed_set(self):
        sn = _make_sn("12345678901234")
        result = check_serial_number(sn, frozenset({"99999999999999"}))
        assert result.passed is False

    def test_serial_number_empty_allowed_skips(self):
        sn = _make_sn("X")
        assert check_serial_number(sn, None).passed is True
        assert check_serial_number(sn, frozenset()).passed is True


class TestConfigurationChecks:
    def test_all_matching_fields_pass(self):
        cfg = make_configuration(has_acceleration=True, sample_rate_hz=2000)
        expected = ExpectedConfiguration(
            sample_rate_hz=2000,
            has_acceleration=True,
            bit_rate=1843200,
        )
        results = check_configuration(cfg, expected)
        # 3 checked fields (sample_rate, has_accel, bit_rate)
        assert len(results) == 3
        assert all(r.passed for r in results)

    def test_mismatch_reports_failure(self):
        cfg = make_configuration(has_acceleration=True, sample_rate_hz=2000)
        expected = ExpectedConfiguration(sample_rate_hz=1000)
        results = check_configuration(cfg, expected)
        assert len(results) == 1
        assert results[0].passed is False
        assert "1000" in results[0].detail
        assert "2000" in results[0].detail

    def test_no_expected_fields_returns_empty(self):
        cfg = make_configuration()
        results = check_configuration(cfg, ExpectedConfiguration())
        assert results == []

    def test_bias_trim_present_match(self):
        cfg = make_configuration()
        # Default bias_trim_at_startup = False
        assert check_bias_trim_present(None, cfg).passed is True
        assert check_bias_trim_present(_make_bt(), cfg).passed is False

    def test_bias_trim_present_when_expected(self):
        cfg = make_configuration()
        # Mutate via dataclasses.replace alternative
        from dataclasses import replace
        cfg2 = replace(cfg, bias_trim_at_startup=True)
        assert check_bias_trim_present(_make_bt(), cfg2).passed is True
        assert check_bias_trim_present(None, cfg2).passed is False


# ---------------------------------------------------------------------------
# Stream-health checks
# ---------------------------------------------------------------------------

class TestStreamChecks:
    def test_parser_clean_pass(self):
        cfg = make_configuration()
        parser = NormalStreamParser(cfg)
        result = check_parser_clean(parser)
        assert result.passed is True

    def test_parser_clean_fail_when_resyncs(self):
        cfg = make_configuration()
        parser = NormalStreamParser(cfg)
        parser.resync_events = 1
        parser.bytes_discarded = 10
        result = check_parser_clean(parser)
        assert result.passed is False
        assert "1" in result.detail
        assert "10" in result.detail

    def test_no_dropped_frames_pass(self):
        cfg = make_configuration()
        ms = _quiescent_measurements(cfg, n=50)
        assert check_no_dropped_frames(ms).passed is True

    def test_no_dropped_frames_detects_gap(self):
        cfg = make_configuration()
        ms = _quiescent_measurements(cfg, n=10)
        # Drop one frame: skip counter 5.
        del ms[5]
        result = check_no_dropped_frames(ms)
        assert result.passed is False
        assert "gap" in result.detail

    def test_no_dropped_frames_handles_modular_wrap(self):
        cfg = make_configuration()
        ms = _quiescent_measurements(cfg, n=300)   # > 256, wraps once
        assert check_no_dropped_frames(ms).passed is True

    def test_no_dropped_frames_empty_passes(self):
        assert check_no_dropped_frames([]).passed is True
        assert check_no_dropped_frames(
            [_quiescent_measurements(make_configuration(), n=1)[0]]
        ).passed is True

    def test_frame_rate_pass_within_tolerance(self):
        cfg = make_configuration()
        ms = _quiescent_measurements(cfg, n=2000)
        # 2000 frames over exactly 1 s -> 2000 Hz exactly.
        result = check_frame_rate(ms, duration_seconds=1.0, expected_hz=2000)
        assert result.passed is True

    def test_frame_rate_fail_outside_tolerance(self):
        cfg = make_configuration()
        ms = _quiescent_measurements(cfg, n=1800)
        result = check_frame_rate(ms, duration_seconds=1.0, expected_hz=2000)
        assert result.passed is False

    def test_latency_within_pass(self):
        cfg = make_configuration()
        ms = _quiescent_measurements(cfg, n=10, latency_us=1000)
        assert check_latency_within(ms, max_us=2000).passed is True

    def test_latency_within_fail(self):
        cfg = make_configuration()
        ms = _quiescent_measurements(cfg, n=10, latency_us=5000)
        assert check_latency_within(ms, max_us=2000).passed is False

    def test_status_bytes_clean_pass(self):
        cfg = make_configuration()
        ms = _quiescent_measurements(cfg, n=10, status_raw=0x00)
        assert check_status_bytes_clean(ms).passed is True

    def test_status_bytes_clean_fail_on_overload(self):
        cfg = make_configuration()
        ms = _quiescent_measurements(cfg, n=10, status_raw=0x10)  # overload bit
        result = check_status_bytes_clean(ms)
        assert result.passed is False
        assert "overload" in result.detail.lower() or "0x10" in result.detail

    def test_status_bytes_startup_allowance(self):
        cfg = make_configuration()
        # Three startup frames followed by clean frames.
        startup = _quiescent_measurements(cfg, n=3, status_raw=0x40)
        clean = _quiescent_measurements(cfg, n=10, status_raw=0x00)
        ms = startup + clean
        # No allowance -> fails because frame 0 has startup bit.
        assert check_status_bytes_clean(ms, startup_frame_allowance=0).passed is False
        # Allow 3 startup frames -> passes.
        assert check_status_bytes_clean(ms, startup_frame_allowance=3).passed is True


class TestPhysicsChecks:
    def test_temperature_range_pass(self):
        cfg = make_configuration(has_temperature=True, has_acceleration=True)
        ms = _quiescent_measurements(cfg, n=10)   # 25 degC default
        assert check_temperature_range(ms, min_c=0.0, max_c=50.0).passed is True

    def test_temperature_range_fail_low(self):
        cfg = make_configuration(has_temperature=True, has_acceleration=True)
        ms = _quiescent_measurements(cfg, n=10)
        assert check_temperature_range(ms, min_c=30.0, max_c=50.0).passed is False

    def test_temperature_range_no_temp_data_passes(self):
        cfg = make_configuration()   # no temperature cluster
        ms = _quiescent_measurements(cfg, n=10)
        result = check_temperature_range(ms, min_c=0.0, max_c=50.0)
        assert result.passed is True
        assert "no temperature" in result.detail

    def test_gyro_quiescent_pass(self):
        cfg = make_configuration()
        ms = _quiescent_measurements(cfg, n=50)
        assert check_gyro_quiescent(ms, max_mean_dps=0.5, max_std_dps=2.0).passed is True

    def test_gyro_quiescent_fail_on_high_mean(self):
        cfg = make_configuration()
        ms = _quiescent_measurements(cfg, n=10)
        # Mutate one measurement to inject a high mean
        from dataclasses import replace
        ms = [replace(m, gyro=(5.0, 0.0, 0.0)) for m in ms]
        assert check_gyro_quiescent(ms, max_mean_dps=0.5, max_std_dps=10.0).passed is False

    def test_gyro_quiescent_fail_on_high_std(self):
        cfg = make_configuration()
        ms = _quiescent_measurements(cfg, n=10)
        from dataclasses import replace
        # Alternate gyro between +5 and -5 for high std but zero mean.
        ms = [replace(m, gyro=(5.0 if i % 2 == 0 else -5.0, 0.0, 0.0))
              for i, m in enumerate(ms)]
        assert check_gyro_quiescent(ms, max_mean_dps=1.0, max_std_dps=2.0).passed is False

    def test_gravity_magnitude_pass(self):
        cfg = make_configuration(has_acceleration=True)
        ms = _quiescent_measurements(cfg, n=10)   # accel = (0,0,1) -> mag 1
        assert check_gravity_magnitude(ms, expected_g=1.0, tolerance_g=0.05).passed is True

    def test_gravity_magnitude_fail(self):
        cfg = make_configuration(has_acceleration=True)
        ms = _quiescent_measurements(cfg, n=10)
        from dataclasses import replace
        ms = [replace(m, accel=(0.0, 0.0, 2.0)) for m in ms]   # 2g instead of 1g
        assert check_gravity_magnitude(ms, expected_g=1.0, tolerance_g=0.05).passed is False

    def test_gravity_magnitude_no_accel_passes(self):
        cfg = make_configuration()   # no accel cluster
        ms = _quiescent_measurements(cfg, n=10)
        result = check_gravity_magnitude(ms, expected_g=1.0, tolerance_g=0.05)
        assert result.passed is True
        assert "no accel" in result.detail.lower()

    def test_gravity_direction_consistent_pass(self):
        cfg = make_configuration(has_acceleration=True)
        ms = _quiescent_measurements(cfg, n=20)
        assert check_gravity_direction_consistent(ms, max_std_g=0.01).passed is True

    def test_gravity_direction_consistent_fail(self):
        cfg = make_configuration(has_acceleration=True)
        ms = _quiescent_measurements(cfg, n=20)
        from dataclasses import replace
        # Alternate the z-axis between 1.0 and 0.5 -> high std
        ms = [replace(m, accel=(0.0, 0.0, 1.0 if i % 2 == 0 else 0.5))
              for i, m in enumerate(ms)]
        assert check_gravity_direction_consistent(ms, max_std_g=0.01).passed is False

    def test_inclinometer_gravity_pass(self):
        cfg = make_configuration(has_inclination=True)
        ms = _quiescent_measurements(cfg, n=10)
        assert check_inclinometer_gravity(ms, expected_g=1.0, tolerance_g=0.05).passed is True

    def test_inclinometer_gravity_no_data_passes(self):
        cfg = make_configuration()
        ms = _quiescent_measurements(cfg, n=10)
        result = check_inclinometer_gravity(ms, expected_g=1.0, tolerance_g=0.05)
        assert result.passed is True
        assert "no inclinometer" in result.detail.lower()


# ---------------------------------------------------------------------------
# Service / Utility round-trip
# ---------------------------------------------------------------------------

class TestRoundTripChecks:
    def test_service_round_trip_pass(self):
        events = [
            AuditEvent(Mode.NORMAL, "tx", b"SERVICEMODE\r", 0.0),
            AuditEvent(Mode.SERVICE, "rx", b"BANNER\r>", 0.1),
            AuditEvent(Mode.SERVICE, "tx", b"i d\r", 0.2),
            AuditEvent(Mode.SERVICE, "rx", b"datagram=A7\r>", 0.3),
        ]
        result = check_service_round_trip(events)
        assert result.passed is True

    def test_service_round_trip_fail_when_no_prompt(self):
        events = [
            AuditEvent(Mode.NORMAL, "tx", b"SERVICEMODE\r", 0.0),
            AuditEvent(Mode.SERVICE, "rx", b"garbage", 0.1),
        ]
        result = check_service_round_trip(events)
        assert result.passed is False

    def test_service_round_trip_fail_when_no_events(self):
        assert check_service_round_trip([]).passed is False

    def test_utility_round_trip_pass(self):
        response = UtilityResponse(command="isn", status=0,
                                     fields=("12345678901234",), raw=b"")
        sn = _make_sn("12345678901234")
        assert check_utility_round_trip(response, sn).passed is True

    def test_utility_round_trip_fail_on_status(self):
        response = UtilityResponse(command="isn", status=2, fields=(), raw=b"")
        sn = _make_sn("X")
        assert check_utility_round_trip(response, sn).passed is False

    def test_utility_round_trip_fail_on_serial_mismatch(self):
        response = UtilityResponse(command="isn", status=0,
                                     fields=("99999999999999",), raw=b"")
        sn = _make_sn("12345678901234")
        result = check_utility_round_trip(response, sn)
        assert result.passed is False
        assert "12345678901234" in result.detail


class TestExtendedErrorCheck:
    def test_clean(self):
        eed = ExtendedErrorDatagram(
            raw_payload=bytes(EXTENDED_ERROR_PAYLOAD_LENGTH),
            error_bits=0, flags=frozenset())
        assert check_extended_error_clean(eed).passed is True

    def test_failure_on_set_bit(self):
        eed = ExtendedErrorDatagram(
            raw_payload=bytes(EXTENDED_ERROR_PAYLOAD_LENGTH),
            error_bits=(1 << 104), flags=frozenset({"accel_x_overload"}))
        assert check_extended_error_clean(eed).passed is False

    def test_ignore_bits(self):
        eed = ExtendedErrorDatagram(
            raw_payload=bytes(EXTENDED_ERROR_PAYLOAD_LENGTH),
            error_bits=(1 << 16), flags=frozenset({"startup_phase_active"}))
        # Ignoring bit 16 -> passes.
        result = check_extended_error_clean(eed, ignore_bits=frozenset({16}))
        assert result.passed is True

    def test_missing_datagram_fails(self):
        assert check_extended_error_clean(None).passed is False


# ---------------------------------------------------------------------------
# CheckoutReport
# ---------------------------------------------------------------------------

class TestCheckoutReport:
    def test_passed_when_all_checks_pass(self):
        report = CheckoutReport(checks=(
            CheckResult(name="a", passed=True, detail="ok"),
            CheckResult(name="b", passed=True, detail="ok"),
        ))
        assert report.passed() is True

    def test_fails_when_any_check_fails(self):
        report = CheckoutReport(checks=(
            CheckResult(name="a", passed=True, detail="ok"),
            CheckResult(name="b", passed=False, detail="bad"),
        ))
        assert report.passed() is False

    def test_summary_includes_pass_fail_marks(self):
        report = CheckoutReport(checks=(
            CheckResult(name="x", passed=True, detail="all good"),
            CheckResult(name="y", passed=False, detail="oh no"),
        ))
        s = report.summary()
        assert "FAIL" in s
        assert "[PASS] x" in s
        assert "[FAIL] y" in s
        assert "1/2 checks passed" in s

    def test_to_json_is_valid_json(self):
        report = CheckoutReport(checks=(
            CheckResult(name="x", passed=True, detail="ok", measured=42),
        ))
        decoded = json.loads(report.to_json())
        assert decoded["passed"] is True
        assert decoded["checks"][0]["name"] == "x"
        assert decoded["checks"][0]["measured"] == 42

    def test_to_dict_handles_bytes_and_init_sequence(self):
        cfg = make_configuration()
        init = InitSequence(
            part_number=_make_pn(),
            serial_number=_make_sn(),
            configuration=cfg,
            bias_trim=None,
            raw=b"\xde\xad\xbe\xef",
        )
        report = CheckoutReport(
            checks=(CheckResult(name="x", passed=True, detail="ok"),),
            init_sequence=init,
            measurement_count=100,
            duration_seconds=1.0,
        )
        d = report.to_dict()
        # Round-trips through json without raising.
        s = json.dumps(d, default=str)
        assert "deadbeef" in s


# ---------------------------------------------------------------------------
# End-to-end integration: run_checkout against the demo FakeTransport
# ---------------------------------------------------------------------------

class TestEndToEndCheckout:
    """Exercise the full ``run_checkout`` composition from the demo module.

    These tests load ``examples/functional_checkout.py`` as a module and
    run its synthesized FakeTransport flow. Verifies the primitives
    compose cleanly and that single-point mutations to the synthesized
    data flip exactly the expected checks.
    """

    def _load_demo(self):
        import importlib.util
        import pathlib as _p
        path = _p.Path(__file__).resolve().parents[1] / "examples" / "functional_checkout.py"
        spec = importlib.util.spec_from_file_location("demo_checkout", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_clean_run_all_pass(self):
        demo = self._load_demo()
        transport = demo.build_demo_transport(measurement_count=50)
        report = demo.run_checkout(
            transport,
            demo.demo_expected_configuration(),
            measurement_count=50,
            measurement_duration=50 / 2000.0,
        )
        assert report.passed(), report.summary()

    def test_serial_mismatch_fails_only_serial_check(self):
        demo = self._load_demo()
        transport = demo.build_demo_transport(measurement_count=50)
        expected = demo.demo_expected_configuration()
        # Replace the allowed serial set with a non-matching one.
        from dataclasses import replace
        bad_expected = replace(expected, serial_numbers=frozenset({"99999999999999"}))
        report = demo.run_checkout(
            transport, bad_expected,
            measurement_count=50,
            measurement_duration=50 / 2000.0,
        )
        # The serial_number check fails; everything else still passes.
        failures = [c for c in report.checks if not c.passed]
        assert len(failures) == 1
        assert failures[0].name == "serial_number"
        assert not report.passed()

    def test_configuration_mismatch_fails_only_that_field(self):
        demo = self._load_demo()
        transport = demo.build_demo_transport(measurement_count=50)
        expected = demo.demo_expected_configuration()
        from dataclasses import replace
        bad = replace(expected, sample_rate_hz=1000)   # demo synthesizes 2000
        report = demo.run_checkout(
            transport, bad,
            measurement_count=50,
            measurement_duration=50 / 2000.0,
        )
        failures = [c for c in report.checks if not c.passed]
        assert len(failures) == 1
        assert failures[0].name == "configuration.sample_rate_hz"

    def test_writes_json_report(self, tmp_path):
        demo = self._load_demo()
        transport = demo.build_demo_transport(measurement_count=20)
        out = tmp_path / "report.json"
        report = demo.run_checkout(
            transport,
            demo.demo_expected_configuration(),
            measurement_count=20,
            measurement_duration=20 / 2000.0,
            report_path=out,
        )
        assert out.exists()
        decoded = json.loads(out.read_text())
        assert decoded["passed"] is True
        assert decoded["measurement_count"] == 20
        assert len(decoded["checks"]) == len(report.checks)
