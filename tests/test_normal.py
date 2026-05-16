"""Tests for the Normal-Mode datagram catalogue, builder, and per-frame parser."""

import pytest

from pystim300.crc import crc32_stim300
from pystim300.normal import (
    NORMAL_SPECS,
    Measurement,
    build_normal_frame,
    parse_normal_payload,
)
from pystim300.status import StatusByte

from conftest import ALL_CLUSTER_COMBOS, make_configuration, make_measurement


class TestNormalFrameSpecs:
    def test_all_16_ids_present(self):
        expected = {0x90, 0x91, 0x92, 0x93, 0x94, 0xA5, 0xA6, 0xA7,
                    0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7}
        assert set(NORMAL_SPECS) == expected

    def test_payload_lengths_against_table_5_12(self):
        # Table 5-12 (p.28): "Number of transmitted bytes" including 4-byte CRC,
        # excluding CR+LF. So expected = 1 (ID) + payload + 4.
        expected_bytes = {
            0x90: 18, 0x91: 28, 0x92: 28, 0x93: 38,
            0x94: 25, 0xA5: 42, 0xA6: 42, 0xA7: 59,
            0xF0: 22, 0xF1: 32, 0xF2: 32, 0xF3: 42,
            0xF4: 29, 0xF5: 46, 0xF6: 46, 0xF7: 63,
        }
        for dgid, total in expected_bytes.items():
            spec = NORMAL_SPECS[dgid]
            assert 1 + spec.payload_length + 4 == total, "ID 0x{0:02X}".format(dgid)

    def test_dummy_bytes_align_to_4(self):
        # Table 5-22 (p.37): dummy padding brings (ID + payload) to a 4-byte
        # multiple before CRC computation.
        for spec in NORMAL_SPECS.values():
            assert (1 + spec.payload_length + spec.dummy_bytes) % 4 == 0

    @pytest.mark.parametrize("dgid, expected_dummy", [
        (0x90, 2), (0x91, 0), (0x92, 0), (0x93, 2),
        (0x94, 3), (0xA5, 2), (0xA6, 2), (0xA7, 1),
        (0xF0, 2), (0xF1, 0), (0xF2, 0), (0xF3, 2),
        (0xF4, 3), (0xF5, 2), (0xF6, 2), (0xF7, 1),
    ])
    def test_dummy_byte_counts_match_table_5_22(self, dgid, expected_dummy):
        assert NORMAL_SPECS[dgid].dummy_bytes == expected_dummy


class TestRoundTripAllIds:
    @pytest.mark.parametrize("has_pps, has_temp, has_incl, has_accel", ALL_CLUSTER_COMBOS)
    @pytest.mark.parametrize("crlf", [False, True])
    def test_build_and_parse(self, has_pps, has_temp, has_incl, has_accel, crlf, vec3_equal):
        cfg = make_configuration(
            has_pps=has_pps,
            has_temperature=has_temp,
            has_inclination=has_incl,
            has_acceleration=has_accel,
            crlf=crlf,
        )
        meas = make_measurement(cfg)
        wire = build_normal_frame(meas, cfg)

        spec = NORMAL_SPECS[cfg.datagram_id]
        assert wire[0] == cfg.datagram_id
        expected_len = spec.frame_length(crlf=crlf)
        assert len(wire) == expected_len

        # CRC should validate.
        id_and_payload = wire[:1 + spec.payload_length]
        crc_bytes = wire[1 + spec.payload_length:1 + spec.payload_length + 4]
        expected = crc32_stim300(id_and_payload + b"\x00" * spec.dummy_bytes)
        assert int.from_bytes(crc_bytes, "big") == expected

        if crlf:
            assert wire[-2:] == b"\r\n"

        # Parse the payload back into a Measurement and check equivalence.
        payload = id_and_payload[1:]
        parsed = parse_normal_payload(payload, spec, cfg)
        assert parsed.datagram_id == meas.datagram_id
        assert parsed.counter == meas.counter
        assert parsed.latency_us == meas.latency_us
        assert vec3_equal(parsed.gyro, meas.gyro)
        assert parsed.gyro_status.raw == meas.gyro_status.raw
        assert vec3_equal(parsed.accel, meas.accel)
        assert vec3_equal(parsed.incl, meas.incl)
        assert vec3_equal(parsed.gyro_temp, meas.gyro_temp, tol=1e-2)
        assert vec3_equal(parsed.accel_temp, meas.accel_temp, tol=1e-2)
        assert vec3_equal(parsed.incl_temp, meas.incl_temp, tol=1e-2)
        if cfg.has_pps:
            assert parsed.pps == pytest.approx(meas.pps, abs=1e-6)
        else:
            assert parsed.pps is None


class TestStatusPropagation:
    def test_status_round_trip(self):
        cfg = make_configuration(has_acceleration=True)
        meas = make_measurement(cfg, status_raw=0x14)  # overload on Z
        wire = build_normal_frame(meas, cfg)
        spec = NORMAL_SPECS[cfg.datagram_id]
        payload = wire[1:1 + spec.payload_length]
        parsed = parse_normal_payload(payload, spec, cfg)
        assert parsed.gyro_status.raw == 0x14
        assert parsed.gyro_status.overload
        assert parsed.gyro_status.axis_z
        assert parsed.accel_status.raw == 0x14


class TestPayloadValidation:
    def test_payload_length_mismatch_raises(self):
        cfg = make_configuration()
        spec = NORMAL_SPECS[cfg.datagram_id]
        with pytest.raises(Exception):
            parse_normal_payload(b"\x00" * (spec.payload_length - 1), spec, cfg)


class TestUnitDispatch:
    def test_gyro_incremental_angle_round_trip(self, vec3_equal):
        cfg = make_configuration(gyro_output_unit=0b0001)  # INCREMENTAL_ANGLE
        # Smaller numbers because the LSB is much finer.
        meas = make_measurement(cfg, gyro=(0.001, -0.002, 0.0005))
        wire = build_normal_frame(meas, cfg)
        spec = NORMAL_SPECS[cfg.datagram_id]
        parsed = parse_normal_payload(wire[1:1 + spec.payload_length], spec, cfg)
        assert vec3_equal(parsed.gyro, meas.gyro, tol=1e-6)

    def test_pps_time_since_round_trip(self):
        cfg = make_configuration(has_pps=True, pps_output_unit=0b0010)  # TIME_SINCE_0
        meas = make_measurement(cfg, pps=12345.0)
        wire = build_normal_frame(meas, cfg)
        spec = NORMAL_SPECS[cfg.datagram_id]
        parsed = parse_normal_payload(wire[1:1 + spec.payload_length], spec, cfg)
        assert parsed.pps == 12345.0
