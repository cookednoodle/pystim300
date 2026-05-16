"""Scaling helpers for STIM300 measurement fields.

Every measurement field on the wire is a fixed-width big-endian signed (or,
in one case, unsigned) integer. Conversion to physical units is documented
in §7.5.2.2.2 through 7.5.2.2.19. This module wraps those equations as
small pure functions; ``normal.py`` calls them while parsing a datagram.

Gyro:                       §7.5.2.2.2 to 7.5.2.2.5
Accelerometer:              §7.5.2.2.7 to 7.5.2.2.10
Inclinometer:               §7.5.2.2.11 to 7.5.2.2.14
PPS:                        §7.5.2.2.15 to 7.5.2.2.16
Temperature:                §7.5.2.2.17
Counter / Latency:          §7.5.2.2.18 / 7.5.2.2.19

Encoders (``encode_*``) are the exact inverses of the decoders and exist so
tests can build wire-shaped bytes from a known physical value and round-trip
through the parser.
"""

from typing import Tuple

Vec3 = Tuple[float, float, float]

# Accelerometer divisor tables. Equation 3 (acceleration [g], p.52) and
# Equation 4 (incremental velocity [m/s/sample], p.52); both keyed on the
# configured accelerometer range.
_ACCEL_G_DIVISOR = {       # Equation 3
    5: 2 ** 20,
    10: 2 ** 19,
    30: 2 ** 18,
    80: 2 ** 16,
}
_ACCEL_IV_DIVISOR = {      # Equation 4
    5: 2 ** 23,
    10: 2 ** 22,
    30: 2 ** 21,
    80: 2 ** 19,
}

_INT24_MAX = (1 << 23) - 1
_INT24_MIN = -(1 << 23)
_INT16_MAX = (1 << 15) - 1
_INT16_MIN = -(1 << 15)


def decode_int24(data: bytes) -> int:
    """Decode 3 big-endian bytes as a signed two's-complement integer."""
    if len(data) != 3:
        raise ValueError("int24 needs exactly 3 bytes, got {0}".format(len(data)))
    value = (data[0] << 16) | (data[1] << 8) | data[2]
    if value & 0x800000:
        value -= 1 << 24
    return value


def decode_uint24(data: bytes) -> int:
    """Decode 3 big-endian bytes as an unsigned integer (used by Filtered PPS)."""
    if len(data) != 3:
        raise ValueError("uint24 needs exactly 3 bytes, got {0}".format(len(data)))
    return (data[0] << 16) | (data[1] << 8) | data[2]


def decode_int16(data: bytes) -> int:
    """Decode 2 big-endian bytes as a signed two's-complement integer."""
    if len(data) != 2:
        raise ValueError("int16 needs exactly 2 bytes, got {0}".format(len(data)))
    value = (data[0] << 8) | data[1]
    if value & 0x8000:
        value -= 1 << 16
    return value


def decode_uint16(data: bytes) -> int:
    """Decode 2 big-endian bytes as an unsigned integer (used by Latency)."""
    if len(data) != 2:
        raise ValueError("uint16 needs exactly 2 bytes, got {0}".format(len(data)))
    return (data[0] << 8) | data[1]


def encode_int24(value: int) -> bytes:
    """Inverse of ``decode_int24``. Raises ``ValueError`` on overflow."""
    if not _INT24_MIN <= value <= _INT24_MAX:
        raise ValueError("int24 out of range: {0}".format(value))
    if value < 0:
        value += 1 << 24
    return bytes(((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF))


def encode_uint24(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFF:
        raise ValueError("uint24 out of range: {0}".format(value))
    return bytes(((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF))


def encode_int16(value: int) -> bytes:
    if not _INT16_MIN <= value <= _INT16_MAX:
        raise ValueError("int16 out of range: {0}".format(value))
    if value < 0:
        value += 1 << 16
    return bytes(((value >> 8) & 0xFF, value & 0xFF))


def encode_uint16(value: int) -> bytes:
    if not 0 <= value <= 0xFFFF:
        raise ValueError("uint16 out of range: {0}".format(value))
    return bytes(((value >> 8) & 0xFF, value & 0xFF))


def decode_gyro_angular_rate(data: bytes) -> float:
    """Eq. 1, §7.5.2.2.2 (p.49). Output unit: [deg/s]. Divisor 2**14."""
    return decode_int24(data) / float(1 << 14)


def decode_gyro_incremental_angle(data: bytes) -> float:
    """Eq. 2, §7.5.2.2.3 (p.49). Output unit: [deg/sample]. Divisor 2**21.

    Also used for ``integrated angle`` (§7.5.2.2.5).
    """
    return decode_int24(data) / float(1 << 21)


def decode_accel_g(data: bytes, range_g: int) -> float:
    """Eq. 3, §7.5.2.2.7 (p.52). Output unit: [g]. Range-dependent divisor.

    Also used for ``average acceleration`` (§7.5.2.2.9).
    """
    if range_g not in _ACCEL_G_DIVISOR:
        raise ValueError("accelerometer range must be one of {0}, got {1}".format(
            sorted(_ACCEL_G_DIVISOR), range_g))
    return decode_int24(data) / float(_ACCEL_G_DIVISOR[range_g])


def decode_accel_incremental_velocity(data: bytes, range_g: int) -> float:
    """Eq. 4, §7.5.2.2.8 (p.52). Output unit: [m/s/sample]. Range-dependent.

    Also used for ``integrated velocity`` (§7.5.2.2.10).
    """
    if range_g not in _ACCEL_IV_DIVISOR:
        raise ValueError("accelerometer range must be one of {0}, got {1}".format(
            sorted(_ACCEL_IV_DIVISOR), range_g))
    return decode_int24(data) / float(_ACCEL_IV_DIVISOR[range_g])


def decode_incl_g(data: bytes) -> float:
    """Eq. 5, §7.5.2.2.11 (p.53). Output unit: [g]. Divisor 2**22.

    Also used for inclinometer ``average acceleration`` (§7.5.2.2.13).
    """
    return decode_int24(data) / float(1 << 22)


def decode_incl_incremental_velocity(data: bytes) -> float:
    """Eq. 6, §7.5.2.2.12 (p.54). Output unit: [m/s/sample]. Divisor 2**25.

    Also used for inclinometer ``integrated velocity`` (§7.5.2.2.14).
    """
    return decode_int24(data) / float(1 << 25)


def decode_pps_time_since(data: bytes) -> int:
    """Eq. 7, §7.5.2.2.15 (p.54). PPS ``Time since detection``: [us]. Divisor 1."""
    return decode_int24(data)


def decode_pps_filtered(data: bytes) -> float:
    """Eq. 8, §7.5.2.2.16 (p.54). PPS ``Filtered`` value, unsigned, range [0, 1]."""
    return decode_uint24(data) / float(1 << 22)


def decode_temperature(data: bytes) -> float:
    """Eq. 9, §7.5.2.2.17 (p.55). Output unit: [degC]. 16-bit signed, divisor 2**8."""
    return decode_int16(data) / float(1 << 8)


def decode_latency_us(data: bytes) -> int:
    """Eq. 10, §7.5.2.2.19 (p.55). Output unit: [us]. 16-bit unsigned, no scaling."""
    return decode_uint16(data)


def encode_gyro_angular_rate(value: float) -> bytes:
    return encode_int24(int(round(value * (1 << 14))))


def encode_gyro_incremental_angle(value: float) -> bytes:
    return encode_int24(int(round(value * (1 << 21))))


def encode_accel_g(value: float, range_g: int) -> bytes:
    if range_g not in _ACCEL_G_DIVISOR:
        raise ValueError("accelerometer range must be one of {0}, got {1}".format(
            sorted(_ACCEL_G_DIVISOR), range_g))
    return encode_int24(int(round(value * _ACCEL_G_DIVISOR[range_g])))


def encode_accel_incremental_velocity(value: float, range_g: int) -> bytes:
    if range_g not in _ACCEL_IV_DIVISOR:
        raise ValueError("accelerometer range must be one of {0}, got {1}".format(
            sorted(_ACCEL_IV_DIVISOR), range_g))
    return encode_int24(int(round(value * _ACCEL_IV_DIVISOR[range_g])))


def encode_incl_g(value: float) -> bytes:
    return encode_int24(int(round(value * (1 << 22))))


def encode_incl_incremental_velocity(value: float) -> bytes:
    return encode_int24(int(round(value * (1 << 25))))


def encode_pps_time_since(value: int) -> bytes:
    return encode_int24(value)


def encode_pps_filtered(value: float) -> bytes:
    return encode_uint24(int(round(value * (1 << 22))))


def encode_temperature(value: float) -> bytes:
    return encode_int16(int(round(value * (1 << 8))))


def encode_latency_us(value: int) -> bytes:
    return encode_uint16(value)
