"""Init-Mode and on-demand datagrams.

Five datagrams besides the Configuration datagram (handled in
``configuration.py``):

* **Part Number** (0xB1 / 0xB3) - Table 5-13 (p.29) + Table 5-14 (p.29)
* **Serial Number** (0xB5 / 0xB7) - Table 5-15 (p.30)
* **Bias Trim Offset** (0xD1 / 0xD2) - Table 5-17 (p.34)
* **Extended Error Information** (0xBE / 0xBF) - Table 5-18 / 5-19 (pp.34-35)

The Configuration datagram lives in ``configuration.py`` because the parser
needs its decoded fields to drive Normal-Mode framing.

Every dataclass carries:

* the **physical** decoded fields (``Vec3`` triples, ints, strings, ...)
* a ``raw_payload: bytes`` snapshot of the wire bytes between the identifier
  and the CRC, so downstream code can decode fields not yet surfaced.

Each datagram type also exposes a ``parse(payload)`` classmethod that
expects the payload bytes (i.e. the bytes between the identifier and the
CRC; framing strips the identifier, CRC and optional CR+LF before calling).
"""

from dataclasses import dataclass, field
from typing import FrozenSet, Tuple

from pystim300.scaling import (
    Vec3,
    decode_accel_g,
    decode_gyro_angular_rate,
    decode_incl_g,
    decode_int24,
    decode_uint16,
)

# Payload lengths (bytes between identifier and CRC) for each Init-Mode datagram.
PART_NUMBER_PAYLOAD_LENGTH = 15          # Table 5-13
SERIAL_NUMBER_PAYLOAD_LENGTH = 15        # Table 5-15
BIAS_TRIM_PAYLOAD_LENGTH = 35            # Table 5-17
EXTENDED_ERROR_PAYLOAD_LENGTH = 16       # Table 5-18

# Identifier bytes (low-nibble bit 1 distinguishes the CRLF variant).
PART_NUMBER_IDS = (0xB1, 0xB3)
SERIAL_NUMBER_IDS = (0xB5, 0xB7)
BIAS_TRIM_IDS = (0xD1, 0xD2)
EXTENDED_ERROR_IDS = (0xBE, 0xBF)


def _nibble_char(n: int) -> str:
    """Convert a 4-bit value to its ASCII rendering per Table 5-14 (p.29)."""
    if 0 <= n <= 9:
        return chr(n + 48)   # '0'-'9'
    if 10 <= n <= 15:
        return chr(n + 55)   # 'A'-'F'
    # Two-nibble combined digits (digits 13 and 15) may overflow; fall back
    # to '?' to keep the string printable rather than raising mid-parse.
    return "?"


@dataclass(frozen=True)
class PartNumberDatagram:
    """Decoded Part Number datagram (IDs 0xB1 / 0xB3).

    See Table 5-13 (p.29) for the byte layout and Table 5-14 (p.29) for the
    nibble-to-ASCII conversion. The textual ``part_number`` follows the form
    ``DDDDD-DDDDDD-DDDD`` plus a trailing revision character.
    """

    raw_payload: bytes
    part_number: str
    revision: str

    @classmethod
    def parse(cls, payload: bytes) -> "PartNumberDatagram":
        if len(payload) != PART_NUMBER_PAYLOAD_LENGTH:
            raise ValueError("Part Number payload must be {0} bytes, got {1}".format(
                PART_NUMBER_PAYLOAD_LENGTH, len(payload)))
        b = payload
        # Table 5-14, p.29 - nibble locations
        d1 = b[0] & 0x0F
        d2 = (b[1] >> 4) & 0x0F
        d3 = b[1] & 0x0F
        d4 = (b[2] >> 4) & 0x0F
        d5 = b[2] & 0x0F
        # b[3] is fixed ASCII '-' (0x2D)
        d6 = (b[4] >> 4) & 0x0F
        d7 = b[4] & 0x0F
        d8 = (b[5] >> 4) & 0x0F
        d9 = b[5] & 0x0F
        d10 = (b[6] >> 4) & 0x0F
        d11 = b[6] & 0x0F
        # b[7] is fixed ASCII '-' (0x2D)
        # Digits 12, 13, 15 are constructed from two nibbles (Table 5-14)
        p12 = (b[8] >> 4) & 0x0F
        p13 = b[8] & 0x0F
        p14 = (b[9] >> 4) & 0x0F
        p15 = b[9] & 0x0F
        p16 = (b[10] >> 4) & 0x0F
        p17 = b[10] & 0x0F
        digit_12 = p17                          # Table 5-14
        digit_13 = p12 + (p16 << 4)             # Table 5-14
        digit_14 = p13                          # Table 5-14
        digit_15 = p14 + (p15 << 4)             # Table 5-14

        part_number = (
            _nibble_char(d1) + _nibble_char(d2) + _nibble_char(d3)
            + _nibble_char(d4) + _nibble_char(d5)
            + "-"
            + _nibble_char(d6) + _nibble_char(d7) + _nibble_char(d8)
            + _nibble_char(d9) + _nibble_char(d10) + _nibble_char(d11)
            + "-"
            + _nibble_char(digit_12) + _nibble_char(digit_13)
            + _nibble_char(digit_14) + _nibble_char(digit_15)
        )
        revision = chr(b[14])   # Table 5-13 byte 15
        return cls(raw_payload=bytes(payload), part_number=part_number, revision=revision)


@dataclass(frozen=True)
class SerialNumberDatagram:
    """Decoded Serial Number datagram (IDs 0xB5 / 0xB7).

    See Table 5-15 (p.30). The number is a 14-digit BCD field preceded by
    the ASCII character ``N`` and followed by reserved bytes.
    """

    raw_payload: bytes
    serial_number: str

    @classmethod
    def parse(cls, payload: bytes) -> "SerialNumberDatagram":
        if len(payload) != SERIAL_NUMBER_PAYLOAD_LENGTH:
            raise ValueError("Serial Number payload must be {0} bytes, got {1}".format(
                SERIAL_NUMBER_PAYLOAD_LENGTH, len(payload)))
        # Byte 0 of the payload is ASCII 'N' (0x4E) per Table 5-15.
        # Bytes 1..7 contain 14 BCD digits (high nibble, low nibble each).
        digits = []
        for byte in payload[1:8]:
            digits.append(_nibble_char((byte >> 4) & 0x0F))
            digits.append(_nibble_char(byte & 0x0F))
        return cls(raw_payload=bytes(payload), serial_number="".join(digits))


@dataclass(frozen=True)
class BiasTrimDatagram:
    """Decoded Bias Trim Offset datagram (IDs 0xD1 / 0xD2).

    See Table 5-17 (p.34). Gyro offsets are always [deg/s], accelerometer and
    inclinometer offsets are always [g], regardless of the configured Normal-
    Mode output unit (Eq. 11, p.56).
    """

    raw_payload: bytes
    gyro_offset: Vec3            # deg/s
    accel_offset: Vec3           # g
    incl_offset: Vec3            # g
    reference_info: int          # uint32, opaque
    remaining_saves: int         # uint16

    @classmethod
    def parse(cls, payload: bytes) -> "BiasTrimDatagram":
        if len(payload) != BIAS_TRIM_PAYLOAD_LENGTH:
            raise ValueError("Bias Trim payload must be {0} bytes, got {1}".format(
                BIAS_TRIM_PAYLOAD_LENGTH, len(payload)))
        b = payload
        # Gyro - Eq. 1 form, divisor 2**14 (deg/s)
        gyro = (
            decode_gyro_angular_rate(b[0:3]),
            decode_gyro_angular_rate(b[3:6]),
            decode_gyro_angular_rate(b[6:9]),
        )
        # Accel - Eq. 3 form. Bias trim is always [g]; we use the 10g divisor
        # since that's the canonical [g] LSB; range scaling is irrelevant for
        # the bias trim (the device encodes physical units directly).
        accel = (
            decode_accel_g(b[9:12], 10),
            decode_accel_g(b[12:15], 10),
            decode_accel_g(b[15:18], 10),
        )
        # Inclinometer - Eq. 5 form, fixed divisor 2**22 (g)
        incl = (
            decode_incl_g(b[18:21]),
            decode_incl_g(b[21:24]),
            decode_incl_g(b[24:27]),
        )
        reference_info = int.from_bytes(b[27:31], byteorder="big", signed=False)
        remaining_saves = decode_uint16(b[31:33])
        return cls(
            raw_payload=bytes(payload),
            gyro_offset=gyro,
            accel_offset=accel,
            incl_offset=incl,
            reference_info=reference_info,
            remaining_saves=remaining_saves,
        )


@dataclass(frozen=True)
class ExtendedErrorDatagram:
    """Decoded Extended Error Information datagram (IDs 0xBE / 0xBF).

    See Table 5-18 (p.34) for the byte layout and Table 5-19 (p.35) for the
    per-bit meaning. The 128 error bits are exposed both as a raw integer
    (``error_bits``, MSB-first concatenation of bytes 1..16) and as a set of
    human-readable flag names (``flags``); the latter only includes bits the
    library has been taught to name.
    """

    raw_payload: bytes
    error_bits: int              # 128-bit integer; bit i set iff Ei == 1
    flags: FrozenSet[str]

    @classmethod
    def parse(cls, payload: bytes) -> "ExtendedErrorDatagram":
        if len(payload) != EXTENDED_ERROR_PAYLOAD_LENGTH:
            raise ValueError("Extended Error payload must be {0} bytes, got {1}".format(
                EXTENDED_ERROR_PAYLOAD_LENGTH, len(payload)))
        # Bytes 1..16 of Table 5-18 are E127..E0, MSB first.
        value = int.from_bytes(payload, byteorder="big", signed=False)
        named_flags = set()
        for bit, name in _EXTENDED_ERROR_NAMES.items():
            if value & (1 << bit):
                named_flags.add(name)
        return cls(
            raw_payload=bytes(payload),
            error_bits=value,
            flags=frozenset(named_flags),
        )

    def bit(self, index: int) -> bool:
        """Return the value of error bit ``E<index>`` (0 <= index <= 127)."""
        if not 0 <= index <= 127:
            raise ValueError("Extended Error bit index out of range: {0}".format(index))
        return bool(self.error_bits & (1 << index))


# Selected named bits from Table 5-19 (p.35). Only the most actionable bits
# are named here; less common bits are still accessible via ``bit(n)``.
_EXTENDED_ERROR_NAMES = {
    16: "startup_phase_active",
    17: "reference_voltage_1_error",
    18: "reference_voltage_2_error",
    19: "reference_voltage_3_error",
    20: "supply_voltage_error",
    21: "regulated_voltage_1_error",
    22: "regulated_voltage_2_error",
    23: "regulated_voltage_3_error",
    56: "ram_check_error",
    57: "flash_check_error",
    58: "internal_dac_error",
    59: "supply_overvoltage",
    68: "uart_unable_to_transmit",
    85: "self_test_not_running",
    97: "uc_temperature_failure",
    104: "accel_x_overload",
    105: "accel_y_overload",
    106: "accel_z_overload",
    101: "gyro_x_overload",
    102: "gyro_y_overload",
    103: "gyro_z_overload",
    107: "incl_x_overload",
    108: "incl_y_overload",
    109: "incl_z_overload",
    111: "reference_voltage_4_error",
    112: "pps_time_overflow",
}
