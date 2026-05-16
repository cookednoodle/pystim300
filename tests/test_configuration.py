"""Tests for the Configuration datagram decoder (Table 5-16, pp.31-33)."""

import pytest

from pystim300.configuration import (
    CONFIGURATION_PAYLOAD_LENGTH,
    compute_datagram_id,
    decode_configuration,
)


def _empty_payload() -> bytearray:
    """Build a 21-byte payload of zeros (a syntactically valid baseline)."""
    return bytearray(CONFIGURATION_PAYLOAD_LENGTH)


class TestDatagramIdLookup:
    @pytest.mark.parametrize("has_pps, has_temp, has_incl, has_accel, expected", [
        (False, False, False, False, 0x90),
        (False, False, False, True,  0x91),
        (False, False, True,  False, 0x92),
        (False, False, True,  True,  0x93),
        (False, True,  False, False, 0x94),
        (False, True,  False, True,  0xA5),
        (False, True,  True,  False, 0xA6),
        (False, True,  True,  True,  0xA7),
        (True,  False, False, False, 0xF0),
        (True,  False, False, True,  0xF1),
        (True,  False, True,  False, 0xF2),
        (True,  False, True,  True,  0xF3),
        (True,  True,  False, False, 0xF4),
        (True,  True,  False, True,  0xF5),
        (True,  True,  True,  False, 0xF6),
        (True,  True,  True,  True,  0xF7),
    ])
    def test_table_5_21(self, has_pps, has_temp, has_incl, has_accel, expected):
        assert compute_datagram_id(
            has_pps=has_pps,
            has_temperature=has_temp,
            has_inclination=has_incl,
            has_acceleration=has_accel,
        ) == expected


class TestDecodeConfiguration:
    def test_payload_length_validation(self):
        with pytest.raises(ValueError):
            decode_configuration(b"\x00" * 20)
        with pytest.raises(ValueError):
            decode_configuration(b"\x00" * 22)

    def test_all_zero_baseline(self):
        cfg = decode_configuration(bytes(CONFIGURATION_PAYLOAD_LENGTH))
        # All-zero defaults: 125 Hz, no clusters, no CRLF, 374400 bps, 1 stop, no parity.
        assert cfg.sample_rate_hz == 125
        assert cfg.has_pps is False
        assert cfg.has_temperature is False
        assert cfg.has_inclination is False
        assert cfg.has_acceleration is False
        assert cfg.crlf_termination is False
        assert cfg.bit_rate == 374400
        assert cfg.stop_bits == 1
        assert cfg.parity == "none"
        assert cfg.line_termination is False
        assert cfg.datagram_id == 0x90
        assert cfg.raw_payload == bytes(CONFIGURATION_PAYLOAD_LENGTH)

    def test_sample_rate_codes(self):
        cases = [
            (0b000, 125),
            (0b001, 250),
            (0b010, 500),
            (0b011, 1000),
            (0b100, 2000),
            (0b101, 0),  # External trigger
        ]
        for code, expected in cases:
            payload = _empty_payload()
            payload[2] = (code << 5)
            cfg = decode_configuration(bytes(payload))
            assert cfg.sample_rate_hz == expected, "code {0:03b}".format(code)

    def test_cluster_flags_and_derived_id(self):
        payload = _empty_payload()
        # Byte 3 bits 4..1 + CRLF (bit 0): PPS=1, temp=1, incl=1, accel=1, CRLF=1
        payload[2] = 0b00011111
        cfg = decode_configuration(bytes(payload))
        assert cfg.has_pps and cfg.has_temperature and cfg.has_inclination and cfg.has_acceleration
        assert cfg.crlf_termination is True
        assert cfg.datagram_id == 0xF7

    def test_bit_rate_codes(self):
        for code, expected in [(0b0000, 374400), (0b0001, 460800),
                                (0b0010, 921600), (0b0011, 1843200),
                                (0b1111, 0)]:
            payload = _empty_payload()
            payload[3] = code << 4
            cfg = decode_configuration(bytes(payload))
            assert cfg.bit_rate == expected, "code {0:04b}".format(code)

    def test_parity_codes(self):
        for code, expected in [(0b00, "none"), (0b01, "even"), (0b10, "odd")]:
            payload = _empty_payload()
            payload[3] = code << 1
            cfg = decode_configuration(bytes(payload))
            assert cfg.parity == expected, "code {0:02b}".format(code)

    def test_stop_bits(self):
        payload = _empty_payload()
        payload[3] = 0x08  # bit 3 = stop bits 2
        assert decode_configuration(bytes(payload)).stop_bits == 2

    def test_gyro_axes_and_lp_filter(self):
        payload = _empty_payload()
        # Byte 5: X+Y+Z active (bits 6,5,4); output unit 0001 (incremental angle)
        payload[4] = 0b01110001
        # Byte 6: X = 262Hz (100), Y = 33Hz (001)
        payload[5] = (0b100 << 4) | 0b001
        # Byte 7: Z = 66Hz (010), g-comp 0000
        payload[6] = (0b010 << 4)
        cfg = decode_configuration(bytes(payload))
        assert cfg.gyro_axes_active == (True, True, True)
        assert cfg.gyro_output_unit == 0b0001
        assert cfg.gyro_lp_filter_hz == (262, 33, 66)

    def test_accel_range_codes(self):
        for code, expected in [(0b0000, 10), (0b0011, 5),
                                (0b0100, 30), (0b0110, 80)]:
            payload = _empty_payload()
            payload[16] = (code << 4) | code  # X and Y both
            payload[17] = code << 4            # Z
            cfg = decode_configuration(bytes(payload))
            assert cfg.accel_range_g == (expected, expected, expected), "code {0:04b}".format(code)

    def test_raw_payload_preserved(self):
        payload = bytes(range(CONFIGURATION_PAYLOAD_LENGTH))
        cfg = decode_configuration(payload)
        assert cfg.raw_payload == payload

    def test_revision_and_firmware(self):
        payload = _empty_payload()
        payload[0] = ord("E")
        payload[1] = 31
        cfg = decode_configuration(bytes(payload))
        assert cfg.revision_char == "E"
        assert cfg.firmware_revision == 31
