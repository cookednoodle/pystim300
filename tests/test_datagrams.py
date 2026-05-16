"""Tests for the Init-Mode datagram dataclasses."""

import pytest

from pystim300.datagrams import (
    BIAS_TRIM_PAYLOAD_LENGTH,
    EXTENDED_ERROR_PAYLOAD_LENGTH,
    PART_NUMBER_PAYLOAD_LENGTH,
    SERIAL_NUMBER_PAYLOAD_LENGTH,
    BiasTrimDatagram,
    ExtendedErrorDatagram,
    PartNumberDatagram,
    SerialNumberDatagram,
)
from pystim300.scaling import (
    encode_accel_g,
    encode_gyro_angular_rate,
    encode_incl_g,
    encode_uint16,
)


class TestPartNumber:
    def _payload(self) -> bytes:
        # Construct a payload representing "12345-67890-1234" with revision 'B'.
        # Table 5-13/5-14 (p.29):
        #   byte 0 low nibble = digit 1
        #   byte 1 high/low = digits 2/3
        #   byte 2 high/low = digits 4/5
        #   byte 3 = '-' (0x2D)
        #   byte 4 high/low = digits 6/7
        #   byte 5 high/low = digits 8/9
        #   byte 6 high/low = digits 10/11
        #   byte 7 = '-' (0x2D)
        #   byte 8 high = P12 (digit 13 low nibble), byte 8 low = P13 (digit 14)
        #   byte 9 high = P14 (digit 15 low), byte 9 low = P15 (digit 15 high)
        #   byte 10 high = P16 (digit 13 high), byte 10 low = P17 (digit 12)
        #   bytes 11-13 reserved (0)
        #   byte 14 = revision ASCII
        b = bytearray(15)
        b[0] = 0x01            # digit 1 = 1
        b[1] = (0x02 << 4) | 0x03  # digits 2, 3
        b[2] = (0x04 << 4) | 0x05  # digits 4, 5
        b[3] = 0x2D
        b[4] = (0x06 << 4) | 0x07
        b[5] = (0x08 << 4) | 0x09
        b[6] = (0x00 << 4) | 0x01  # digits 10, 11 = 0, 1
        b[7] = 0x2D
        # digit 12 = P17 = 1
        # digit 13 = P12 + P16*16 = 2 + 0*16 = 2 -> set P12 = 2, P16 = 0
        # digit 14 = P13 = 3
        # digit 15 = P14 + P15*16 = 4 + 0*16 = 4 -> set P14 = 4, P15 = 0
        b[8] = (0x02 << 4) | 0x03   # P12, P13
        b[9] = (0x04 << 4) | 0x00   # P14, P15
        b[10] = (0x00 << 4) | 0x01  # P16, P17
        b[14] = ord("B")
        return bytes(b)

    def test_round_trip(self):
        payload = self._payload()
        pn = PartNumberDatagram.parse(payload)
        assert pn.part_number == "12345-678901-1234"
        assert pn.revision == "B"
        assert pn.raw_payload == payload

    def test_wrong_length(self):
        with pytest.raises(ValueError):
            PartNumberDatagram.parse(b"\x00" * (PART_NUMBER_PAYLOAD_LENGTH - 1))

    def test_hex_digits_render_as_uppercase(self):
        b = bytearray(15)
        b[0] = 0x0A   # digit 1 = 0xA -> 'A'
        b[14] = ord("-")
        pn = PartNumberDatagram.parse(bytes(b))
        assert pn.part_number.startswith("A0000")
        assert pn.revision == "-"


class TestSerialNumber:
    def test_round_trip(self):
        b = bytearray(SERIAL_NUMBER_PAYLOAD_LENGTH)
        b[0] = ord("N")
        # 14 BCD digits = "12345678901234"
        digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
        for i in range(7):
            b[1 + i] = (digits[2 * i] << 4) | digits[2 * i + 1]
        sn = SerialNumberDatagram.parse(bytes(b))
        assert sn.serial_number == "12345678901234"
        assert sn.raw_payload == bytes(b)

    def test_wrong_length(self):
        with pytest.raises(ValueError):
            SerialNumberDatagram.parse(b"\x00" * (SERIAL_NUMBER_PAYLOAD_LENGTH - 1))


class TestBiasTrim:
    def test_round_trip(self):
        gyro = (1.25, -2.5, 0.5)        # deg/s
        accel = (0.01, -0.02, 0.005)    # g
        incl = (0.1, -0.05, 0.025)      # g
        b = bytearray()
        for v in gyro:
            b += encode_gyro_angular_rate(v)
        for v in accel:
            b += encode_accel_g(v, 10)
        for v in incl:
            b += encode_incl_g(v)
        b += (0xDEADBEEF).to_bytes(4, "big")           # reference_info
        b += encode_uint16(7)                          # remaining_saves
        b += b"\x00\x00"                                # reserved
        assert len(b) == BIAS_TRIM_PAYLOAD_LENGTH

        bt = BiasTrimDatagram.parse(bytes(b))
        assert bt.gyro_offset == pytest.approx(gyro, rel=1e-4, abs=1e-4)
        assert bt.accel_offset == pytest.approx(accel, rel=1e-4, abs=1e-5)
        assert bt.incl_offset == pytest.approx(incl, rel=1e-4, abs=1e-6)
        assert bt.reference_info == 0xDEADBEEF
        assert bt.remaining_saves == 7
        assert bt.raw_payload == bytes(b)

    def test_wrong_length(self):
        with pytest.raises(ValueError):
            BiasTrimDatagram.parse(b"\x00" * (BIAS_TRIM_PAYLOAD_LENGTH - 1))


class TestExtendedError:
    def test_all_zero(self):
        eed = ExtendedErrorDatagram.parse(b"\x00" * EXTENDED_ERROR_PAYLOAD_LENGTH)
        assert eed.error_bits == 0
        assert eed.flags == frozenset()
        for i in range(128):
            assert eed.bit(i) is False

    def test_startup_phase_bit(self):
        # E16 -> bit 16 of the 128-bit value.
        payload = bytearray(EXTENDED_ERROR_PAYLOAD_LENGTH)
        value = 1 << 16
        payload[:] = value.to_bytes(EXTENDED_ERROR_PAYLOAD_LENGTH, "big")
        eed = ExtendedErrorDatagram.parse(bytes(payload))
        assert eed.bit(16) is True
        assert "startup_phase_active" in eed.flags

    def test_multiple_bits(self):
        # E20 (supply voltage error) + E104 (accel X overload)
        value = (1 << 20) | (1 << 104)
        payload = value.to_bytes(EXTENDED_ERROR_PAYLOAD_LENGTH, "big")
        eed = ExtendedErrorDatagram.parse(payload)
        assert "supply_voltage_error" in eed.flags
        assert "accel_x_overload" in eed.flags
        assert eed.bit(20) and eed.bit(104)

    def test_unnamed_bits_still_accessible(self):
        # E125 isn't named (For future use), but bit() should still work.
        value = 1 << 125
        payload = value.to_bytes(EXTENDED_ERROR_PAYLOAD_LENGTH, "big")
        eed = ExtendedErrorDatagram.parse(payload)
        assert eed.bit(125) is True
        assert eed.flags == frozenset()

    def test_bit_out_of_range(self):
        eed = ExtendedErrorDatagram.parse(b"\x00" * EXTENDED_ERROR_PAYLOAD_LENGTH)
        with pytest.raises(ValueError):
            eed.bit(-1)
        with pytest.raises(ValueError):
            eed.bit(128)

    def test_wrong_length(self):
        with pytest.raises(ValueError):
            ExtendedErrorDatagram.parse(b"\x00" * (EXTENDED_ERROR_PAYLOAD_LENGTH - 1))
