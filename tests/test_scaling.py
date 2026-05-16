"""Tests for the scaling decoders/encoders (§7.5.2.2.x equations)."""

import pytest

from pystim300 import scaling


class TestInt24:
    def test_zero(self):
        assert scaling.decode_int24(b"\x00\x00\x00") == 0

    def test_positive(self):
        assert scaling.decode_int24(b"\x00\x00\x01") == 1
        assert scaling.decode_int24(b"\x00\x00\xff") == 0xFF
        assert scaling.decode_int24(b"\x7f\xff\xff") == 0x7FFFFF

    def test_negative(self):
        assert scaling.decode_int24(b"\xff\xff\xff") == -1
        assert scaling.decode_int24(b"\xff\xff\x00") == -256
        assert scaling.decode_int24(b"\x80\x00\x00") == -0x800000

    @pytest.mark.parametrize("v", [0, 1, -1, 100, -100, 0x7FFFFF, -0x800000, 1234, -567890])
    def test_round_trip(self, v):
        assert scaling.decode_int24(scaling.encode_int24(v)) == v

    def test_overflow(self):
        with pytest.raises(ValueError):
            scaling.encode_int24(0x800000)
        with pytest.raises(ValueError):
            scaling.encode_int24(-0x800001)

    def test_bad_length(self):
        with pytest.raises(ValueError):
            scaling.decode_int24(b"\x00\x00")
        with pytest.raises(ValueError):
            scaling.decode_int24(b"\x00\x00\x00\x00")


class TestInt16:
    @pytest.mark.parametrize("v", [0, 1, -1, 100, -100, 0x7FFF, -0x8000])
    def test_round_trip(self, v):
        assert scaling.decode_int16(scaling.encode_int16(v)) == v

    def test_overflow(self):
        with pytest.raises(ValueError):
            scaling.encode_int16(0x8000)


class TestUint(object):
    def test_uint16(self):
        assert scaling.decode_uint16(b"\x00\x00") == 0
        assert scaling.decode_uint16(b"\xff\xff") == 0xFFFF
        assert scaling.decode_uint16(b"\x01\x02") == 0x0102

    def test_uint24(self):
        assert scaling.decode_uint24(b"\x00\x00\x00") == 0
        assert scaling.decode_uint24(b"\xff\xff\xff") == 0xFFFFFF
        assert scaling.decode_uint24(b"\x12\x34\x56") == 0x123456


class TestGyro:
    def test_angular_rate_round_trip(self):
        for v in [0.0, 1.0, -1.0, 50.0, -50.0, 100.5, -123.456]:
            encoded = scaling.encode_gyro_angular_rate(v)
            decoded = scaling.decode_gyro_angular_rate(encoded)
            # 24-bit at /2**14 -> LSB ~ 6.1e-5 deg/s
            assert abs(decoded - v) < 1e-4

    def test_angular_rate_lsb_magnitude(self):
        # 1 LSB = 1 / 2**14 deg/s
        assert scaling.decode_gyro_angular_rate(b"\x00\x00\x01") == pytest.approx(1.0 / (1 << 14))

    def test_incremental_angle_lsb_magnitude(self):
        assert scaling.decode_gyro_incremental_angle(b"\x00\x00\x01") == pytest.approx(1.0 / (1 << 21))


class TestAccel:
    @pytest.mark.parametrize("range_g, expected_lsb", [
        (5, 1.0 / (1 << 20)),
        (10, 1.0 / (1 << 19)),
        (30, 1.0 / (1 << 18)),
        (80, 1.0 / (1 << 16)),
    ])
    def test_g_lsb_per_range(self, range_g, expected_lsb):
        assert scaling.decode_accel_g(b"\x00\x00\x01", range_g) == pytest.approx(expected_lsb)

    @pytest.mark.parametrize("range_g", [5, 10, 30, 80])
    def test_g_round_trip(self, range_g):
        for v in [0.0, 0.5, -0.5, 1.0, -2.0]:
            encoded = scaling.encode_accel_g(v, range_g)
            decoded = scaling.decode_accel_g(encoded, range_g)
            assert abs(decoded - v) < 1e-3

    def test_bad_range(self):
        with pytest.raises(ValueError):
            scaling.decode_accel_g(b"\x00\x00\x01", 42)

    def test_incremental_velocity_lsb(self):
        assert scaling.decode_accel_incremental_velocity(b"\x00\x00\x01", 10) == \
            pytest.approx(1.0 / (1 << 22))


class TestIncl:
    def test_g_lsb(self):
        # Eq. 5: divisor 2**22
        assert scaling.decode_incl_g(b"\x00\x00\x01") == pytest.approx(1.0 / (1 << 22))

    def test_g_round_trip(self):
        for v in [0.0, 0.1, -0.1, 1.0, -1.0]:
            encoded = scaling.encode_incl_g(v)
            decoded = scaling.decode_incl_g(encoded)
            assert abs(decoded - v) < 1e-5

    def test_incremental_velocity_lsb(self):
        # Eq. 6: divisor 2**25
        assert scaling.decode_incl_incremental_velocity(b"\x00\x00\x01") == pytest.approx(1.0 / (1 << 25))


class TestPps:
    def test_time_since_detection(self):
        # Eq. 7: raw int (us)
        assert scaling.decode_pps_time_since(b"\x00\x00\x01") == 1
        assert scaling.decode_pps_time_since(b"\xff\xff\xff") == -1
        assert scaling.decode_pps_time_since(b"\x00\x0f\xa0") == 4000

    def test_filtered(self):
        # Eq. 8: unsigned / 2**22
        assert scaling.decode_pps_filtered(b"\x00\x00\x00") == 0.0
        assert scaling.decode_pps_filtered(b"\x40\x00\x00") == pytest.approx(1.0)


class TestTemperature:
    def test_zero(self):
        assert scaling.decode_temperature(b"\x00\x00") == 0.0

    def test_one_lsb(self):
        # Eq. 9: divisor 2**8 = 1/256 degC per LSB
        assert scaling.decode_temperature(b"\x00\x01") == pytest.approx(1.0 / 256.0)

    def test_round_trip(self):
        for v in [0.0, 25.0, -10.0, 85.0, -40.0]:
            encoded = scaling.encode_temperature(v)
            decoded = scaling.decode_temperature(encoded)
            assert abs(decoded - v) < 1.0 / 256.0


class TestLatency:
    def test_us(self):
        assert scaling.decode_latency_us(b"\x00\x00") == 0
        assert scaling.decode_latency_us(b"\x03\xe8") == 1000
        assert scaling.decode_latency_us(b"\xff\xff") == 0xFFFF
