"""Configuration datagram decoder.

The Configuration datagram (IDs 0xBC / 0xBD) is emitted in Init Mode and on
demand via the Normal-Mode ``C`` command. It encodes every user-configurable
parameter of the unit in 21 bit-field bytes. See §5.5.3 and Table 5-16
(pp.31-33).

The parser needs three things from this datagram to interpret subsequent
Normal-Mode frames correctly:

* Which clusters are present (gyro / accel / incl / temp / PPS) -> the
  Normal-Mode datagram identifier (Table 5-21).
* Whether CR+LF terminates each Normal-Mode frame -> frame length on the wire.
* The accelerometer range per axis -> the divisor in the scaling equations.

Everything else (gyro/accel/incl output unit codes, LP filter cutoffs, TOV
logic, parity, etc.) is decoded too, but the parser proper only consults the
three items above. The full payload is retained on the dataclass as
``raw_payload`` so downstream code can reach fields that are not (yet)
surfaced as named attributes.
"""

from dataclasses import dataclass, field
from typing import Tuple

# Sample rate code -> Hz (byte 3 bits 7..5). Code 5 means external trigger
# (no fixed rate); the value 0 in ``sample_rate_hz`` is the sentinel.
_SAMPLE_RATE = {
    0b000: 125,
    0b001: 250,
    0b010: 500,
    0b011: 1000,
    0b100: 2000,
    0b101: 0,  # External trigger; see §9 sample rate sub-section.
}

# Bit-rate code -> bits/s (byte 4 bits 7..4). Code 0b1111 is "user-defined"
# (see §9.5); reported as 0 for sentinel.
_BIT_RATE = {
    0b0000: 374400,
    0b0001: 460800,
    0b0010: 921600,
    0b0011: 1843200,
    0b1111: 0,
}

# LP filter code -> Hz (3 bits per axis; same encoding across gyro/accel/incl
# and PPS Y). Table 5-16, e.g. byte 6 for gyro X/Y.
_LP_FILTER_HZ = {
    0b000: 16,
    0b001: 33,
    0b010: 66,
    0b011: 131,
    0b100: 262,
}

# Accelerometer range nibble -> g. Table 5-16, bytes 17 / 18.
_ACCEL_RANGE_G = {
    0b0000: 10,
    0b0011: 5,
    0b0100: 30,
    0b0110: 80,
}

# Datagram identifier lookup keyed on (has_pps, has_temp, has_incl, has_accel).
# Table 5-21 (p.37); pure rate datagram is 0x90 and each cluster bit sets a
# specific identifier.
_DATAGRAM_ID = {
    (False, False, False, False): 0x90,
    (False, False, False, True):  0x91,
    (False, False, True,  False): 0x92,
    (False, False, True,  True):  0x93,
    (False, True,  False, False): 0x94,
    (False, True,  False, True):  0xA5,
    (False, True,  True,  False): 0xA6,
    (False, True,  True,  True):  0xA7,
    (True,  False, False, False): 0xF0,
    (True,  False, False, True):  0xF1,
    (True,  False, True,  False): 0xF2,
    (True,  False, True,  True):  0xF3,
    (True,  True,  False, False): 0xF4,
    (True,  True,  False, True):  0xF5,
    (True,  True,  True,  False): 0xF6,
    (True,  True,  True,  True):  0xF7,
}

CONFIGURATION_PAYLOAD_LENGTH = 21  # Bytes 1..21 of Table 5-16.


@dataclass(frozen=True)
class Configuration:
    """Decoded STIM300 configuration.

    Source: Configuration datagram (IDs 0xBC / 0xBD), Table 5-16 (pp.31-33).
    """

    raw_payload: bytes

    # Byte 1
    revision_char: str
    # Byte 2
    firmware_revision: int

    # Byte 3
    sample_rate_hz: int          # 125/250/500/1000/2000, or 0 for ExtTrig
    has_pps: bool
    has_temperature: bool
    has_inclination: bool
    has_acceleration: bool
    crlf_termination: bool

    # Byte 4
    bit_rate: int                # bps; 0 for "user-defined"
    stop_bits: int               # 1 or 2
    parity: str                  # 'none' | 'odd' | 'even'
    line_termination: bool

    # Bytes 5-7: gyro
    gyro_axes_active: Tuple[bool, bool, bool]
    gyro_output_unit: int        # raw 4-bit code; see Table 5-16 byte 5
    gyro_lp_filter_hz: Tuple[int, int, int]
    gyro_g_compensation: int     # raw 4-bit code; see Table 5-16 byte 7

    # Bytes 8-10: accelerometer
    accel_axes_active: Tuple[bool, bool, bool]
    accel_output_unit: int       # raw 4-bit code; see Table 5-16 byte 8
    accel_lp_filter_hz: Tuple[int, int, int]

    # Bytes 11-13: inclinometer
    incl_axes_active: Tuple[bool, bool, bool]
    incl_output_unit: int        # raw 4-bit code; see Table 5-16 byte 11
    incl_lp_filter_hz: Tuple[int, int, int]

    # Bytes 13-14: PPS
    pps_output_unit: int         # raw 4-bit code; see Table 5-16 byte 13
    pps_lp_filter_hz: int
    has_aux_input: bool
    has_pps_input: bool

    # Bytes 15-20: ranges (gyro always 400 deg/s; incl always 1.7 g on
    # current revision, included for completeness).
    gyro_range_dps: Tuple[int, int, int] = (400, 400, 400)
    accel_range_g: Tuple[int, int, int] = (10, 10, 10)
    incl_range_g: Tuple[float, float, float] = (1.7, 1.7, 1.7)

    # Bytes 21
    tov_logic_v: float = 5.0
    tov_toggling: bool = False
    bias_trim_at_startup: bool = False

    # Derived
    datagram_id: int = field(default=0x90)

    def frame_length(self) -> int:
        """Total bytes per Normal-Mode frame on the wire including CRC and CR/LF.

        Re-computed from ``datagram_id`` and ``crlf_termination`` via the
        Normal-Mode frame catalogue (lives in ``normal.py``); kept as a method
        rather than a stored field to avoid a circular import.
        """
        from pystim300.normal import frame_length_for  # local import; cycle-safe
        return frame_length_for(self.datagram_id, self.crlf_termination)


def decode_configuration(payload: bytes) -> Configuration:
    """Decode the Configuration datagram payload (bytes 1..21 of Table 5-16).

    ``payload`` is the 21 bytes following the identifier byte (0xBC / 0xBD).
    The identifier itself, the CRC, and any trailing CR+LF are stripped by
    the framing layer before this function is called.
    """
    if len(payload) != CONFIGURATION_PAYLOAD_LENGTH:
        raise ValueError("Configuration payload must be {0} bytes, got {1}".format(
            CONFIGURATION_PAYLOAD_LENGTH, len(payload)))

    b = payload
    revision_char = chr(b[0])                            # Table 5-16, byte 1, p.31
    firmware_revision = b[1]                              # Table 5-16, byte 2, p.31

    # Byte 3 — sample rate (bits 7..5), cluster flags (bits 4..1), CRLF (bit 0)
    sample_code = (b[2] >> 5) & 0x07
    sample_rate_hz = _SAMPLE_RATE.get(sample_code, 0)
    has_pps = bool(b[2] & 0x10)                           # Table 5-16, byte 3 bit 4
    has_temperature = bool(b[2] & 0x08)                   # bit 3
    has_inclination = bool(b[2] & 0x04)                   # bit 2
    has_acceleration = bool(b[2] & 0x02)                  # bit 1
    crlf_termination = bool(b[2] & 0x01)                  # bit 0

    # Byte 4 — bit-rate (bits 7..4), stop bits (bit 3), parity (bits 2..1),
    # line term (bit 0)
    bit_rate_code = (b[3] >> 4) & 0x0F
    bit_rate = _BIT_RATE.get(bit_rate_code, 0)
    stop_bits = 2 if (b[3] & 0x08) else 1                 # bit 3
    parity_code = (b[3] >> 1) & 0x03                      # bits 2..1
    parity = {0b00: "none", 0b01: "even", 0b10: "odd"}.get(parity_code, "none")
    line_termination = bool(b[3] & 0x01)                  # bit 0

    # Bytes 5-7 — gyro
    gyro_axes_active = (
        bool(b[4] & 0x40),                                # X axis, bit 6
        bool(b[4] & 0x20),                                # Y axis, bit 5
        bool(b[4] & 0x10),                                # Z axis, bit 4
    )
    gyro_output_unit = b[4] & 0x0F                        # bits 3..0
    gyro_lp_filter_hz = (
        _LP_FILTER_HZ.get((b[5] >> 4) & 0x07, 16),        # X axis, byte 6 bits 6..4
        _LP_FILTER_HZ.get(b[5] & 0x07, 16),               # Y axis, byte 6 bits 2..0
        _LP_FILTER_HZ.get((b[6] >> 4) & 0x07, 16),        # Z axis, byte 7 bits 6..4
    )
    gyro_g_compensation = b[6] & 0x0F                     # byte 7 bits 3..0

    # Bytes 8-10 — accelerometer
    accel_axes_active = (
        bool(b[7] & 0x40),
        bool(b[7] & 0x20),
        bool(b[7] & 0x10),
    )
    accel_output_unit = b[7] & 0x0F
    accel_lp_filter_hz = (
        _LP_FILTER_HZ.get((b[8] >> 4) & 0x07, 16),        # X axis, byte 9
        _LP_FILTER_HZ.get(b[8] & 0x07, 16),               # Y axis, byte 9
        _LP_FILTER_HZ.get((b[9] >> 4) & 0x07, 16),        # Z axis, byte 10
    )

    # Bytes 11-13 — inclinometer
    incl_axes_active = (
        bool(b[10] & 0x40),
        bool(b[10] & 0x20),
        bool(b[10] & 0x10),
    )
    incl_output_unit = b[10] & 0x0F
    incl_lp_filter_hz = (
        _LP_FILTER_HZ.get((b[11] >> 4) & 0x07, 16),       # X axis, byte 12
        _LP_FILTER_HZ.get(b[11] & 0x07, 16),              # Y axis, byte 12
        _LP_FILTER_HZ.get((b[12] >> 4) & 0x07, 16),       # Z axis, byte 13
    )

    # Byte 13 / 14 — PPS
    pps_output_unit = b[12] & 0x0F                        # byte 13 bits 3..0
    has_pps_input = bool(b[13] & 0x08)                    # byte 14 bit 3
    has_aux_input = not has_pps_input
    pps_lp_filter_hz = _LP_FILTER_HZ.get(b[13] & 0x07, 16)  # byte 14 bits 2..0

    # Bytes 15-20 — ranges. Gyro is always 400 dps and inclinometer is always
    # 1.7 g on the current revision, but the bytes are decoded for completeness
    # and to surface any future range options.
    accel_range_g = (
        _ACCEL_RANGE_G.get((b[16] >> 4) & 0x0F, 10),      # X axis, byte 17 high nibble
        _ACCEL_RANGE_G.get(b[16] & 0x0F, 10),             # Y axis, byte 17 low nibble
        _ACCEL_RANGE_G.get((b[17] >> 4) & 0x0F, 10),      # Z axis, byte 18 high nibble
    )

    # Byte 21 — TOV logic + toggling + bias trim transmission
    tov_logic_v = 3.3 if (b[20] & 0x08) else 5.0          # bit 3
    tov_toggling = bool(b[20] & 0x04)                     # bit 2
    bias_trim_at_startup = bool(b[20] & 0x02)             # bit 1

    datagram_id = _DATAGRAM_ID[(has_pps, has_temperature, has_inclination, has_acceleration)]

    return Configuration(
        raw_payload=bytes(payload),
        revision_char=revision_char,
        firmware_revision=firmware_revision,
        sample_rate_hz=sample_rate_hz,
        has_pps=has_pps,
        has_temperature=has_temperature,
        has_inclination=has_inclination,
        has_acceleration=has_acceleration,
        crlf_termination=crlf_termination,
        bit_rate=bit_rate,
        stop_bits=stop_bits,
        parity=parity,
        line_termination=line_termination,
        gyro_axes_active=gyro_axes_active,
        gyro_output_unit=gyro_output_unit,
        gyro_lp_filter_hz=gyro_lp_filter_hz,
        gyro_g_compensation=gyro_g_compensation,
        accel_axes_active=accel_axes_active,
        accel_output_unit=accel_output_unit,
        accel_lp_filter_hz=accel_lp_filter_hz,
        incl_axes_active=incl_axes_active,
        incl_output_unit=incl_output_unit,
        incl_lp_filter_hz=incl_lp_filter_hz,
        pps_output_unit=pps_output_unit,
        pps_lp_filter_hz=pps_lp_filter_hz,
        has_aux_input=has_aux_input,
        has_pps_input=has_pps_input,
        accel_range_g=accel_range_g,
        tov_logic_v=tov_logic_v,
        tov_toggling=tov_toggling,
        bias_trim_at_startup=bias_trim_at_startup,
        datagram_id=datagram_id,
    )


def compute_datagram_id(*, has_pps: bool, has_temperature: bool,
                         has_inclination: bool, has_acceleration: bool) -> int:
    """Map cluster presence flags to the Normal-Mode datagram identifier.

    Table 5-21 (p.37).
    """
    return _DATAGRAM_ID[(has_pps, has_temperature, has_inclination, has_acceleration)]
