"""Normal-Mode datagram catalogue, builder, parser, and streaming framer.

References:
  - §5.5.6 Normal-Mode datagram structure (p.36)
  - §5.5.7 CRC (p.37)
  - Table 5-20 Full datagram layout (p.36)
  - Table 5-21 Datagram identifiers (p.37)
  - Table 5-22 CRC dummy-byte counts (p.37)
  - Table 5-23 Status byte (p.38) - via ``status.py``
  - §7.5.2 Normal Mode (p.45)
  - §7.5.2.2.x Conversion equations (pp.48-55) - via ``scaling.py``
  - §8 Normal-Mode commands (pp.58-59)

A Normal-Mode datagram on the wire is::

    [ID][payload .................][CRC32 big-endian][\\r\\n if CRLF]

The CRC is computed over ``ID || payload || 0x00 * dummy`` where the
dummy padding is determined by Table 5-22 to align the input to 4-byte
boundaries. The dummy bytes are NOT transmitted.

The streaming parser tolerates short reads (any chunk size), special-
datagram interleaving (Part Number / Configuration / etc. emitted in
response to Normal-Mode commands), and CRC failures (resyncs by walking
one byte forward and re-attempting framing).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from pystim300.configuration import Configuration
from pystim300.crc import crc32_stim300
from pystim300.datagrams import (
    BIAS_TRIM_IDS,
    BIAS_TRIM_PAYLOAD_LENGTH,
    BiasTrimDatagram,
    EXTENDED_ERROR_IDS,
    EXTENDED_ERROR_PAYLOAD_LENGTH,
    ExtendedErrorDatagram,
    PART_NUMBER_IDS,
    PART_NUMBER_PAYLOAD_LENGTH,
    PartNumberDatagram,
    SERIAL_NUMBER_IDS,
    SERIAL_NUMBER_PAYLOAD_LENGTH,
    SerialNumberDatagram,
)
from pystim300.exceptions import CrcError, ProtocolError
from pystim300.configuration import decode_configuration as _decode_configuration_payload
from pystim300.scaling import (
    Vec3,
    decode_accel_g,
    decode_accel_incremental_velocity,
    decode_gyro_angular_rate,
    decode_gyro_incremental_angle,
    decode_incl_g,
    decode_incl_incremental_velocity,
    decode_latency_us,
    decode_pps_filtered,
    decode_pps_time_since,
    decode_temperature,
    encode_accel_g,
    encode_accel_incremental_velocity,
    encode_gyro_angular_rate,
    encode_gyro_incremental_angle,
    encode_incl_g,
    encode_incl_incremental_velocity,
    encode_latency_us,
    encode_pps_filtered,
    encode_pps_time_since,
    encode_temperature,
)
from pystim300.status import StatusByte


# Output unit codes (Table 5-16). Gyro unit values, low nibble of byte 5.
GYRO_UNIT_ANGULAR_RATE = 0b0000
GYRO_UNIT_INCREMENTAL_ANGLE = 0b0001
GYRO_UNIT_AVERAGE_RATE = 0b0010
GYRO_UNIT_INTEGRATED_ANGLE = 0b0011
GYRO_UNIT_ANGULAR_RATE_DELAYED = 0b1000
GYRO_UNIT_INCREMENTAL_ANGLE_DELAYED = 0b1001
GYRO_UNIT_AVERAGE_RATE_DELAYED = 0b1010
GYRO_UNIT_INTEGRATED_ANGLE_DELAYED = 0b1011

# Accelerometer / inclinometer unit values, low nibble of bytes 8 / 11.
ACCEL_UNIT_ACCELERATION = 0b0000
ACCEL_UNIT_INCREMENTAL_VELOCITY = 0b0001
ACCEL_UNIT_AVERAGE_ACCELERATION = 0b0010
ACCEL_UNIT_INTEGRATED_VELOCITY_GS = 0b0011
ACCEL_UNIT_INTEGRATED_VELOCITY_MS = 0b0100

# PPS unit values, low nibble of byte 13.
PPS_UNIT_FILTERED = 0b0000
PPS_UNIT_FILTERED_DELAYED = 0b0001
PPS_UNIT_TIME_SINCE_0 = 0b0010
PPS_UNIT_TIME_SINCE_1 = 0b0011


@dataclass(frozen=True)
class NormalFrameSpec:
    """Per-ID layout description for a Normal-Mode datagram.

    The fields are derived from Table 5-21 (cluster presence) and Table
    5-22 (dummy-byte count). ``payload_length`` is the number of bytes
    transmitted between the identifier and the 4-byte CRC.
    """

    datagram_id: int
    has_accel: bool
    has_incl: bool
    has_temp: bool
    has_pps: bool
    payload_length: int
    dummy_bytes: int

    def frame_length(self, *, crlf: bool) -> int:
        """Total bytes on the wire including ID, CRC, and optional CR+LF."""
        return 1 + self.payload_length + 4 + (2 if crlf else 0)


def _build_normal_specs() -> Dict[int, NormalFrameSpec]:
    """Build the Normal-Mode frame catalogue keyed by datagram ID.

    Derived directly from Table 5-21 (cluster presence) and Table 5-22
    (dummy-byte counts), with payload lengths computed from the cluster
    sizes documented in Table 5-20.
    """
    # Cluster sizes from Table 5-20 (p.36):
    #   gyro cluster:   3*3 + 1 status = 10 bytes  (always present)
    #   accel cluster:  3*3 + 1 status = 10 bytes
    #   incl cluster:   3*3 + 1 status = 10 bytes
    #   temp triple:    3*2 + 1 status =  7 bytes  (one per present cluster)
    #   PPS:            3   + 1 status =  4 bytes
    #   trailer:        1 counter + 2 latency =  3 bytes
    gyro_bytes = 10
    cluster_bytes = 10
    temp_bytes = 7
    pps_bytes = 4
    trailer_bytes = 3

    specs: Dict[int, NormalFrameSpec] = {}
    # Cluster presence triples per Table 5-21 (gyro is always present).
    # Tuple order: (has_pps, has_temp, has_incl, has_accel) -> datagram ID
    table_5_21 = {
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
    for (has_pps, has_temp, has_incl, has_accel), dgid in table_5_21.items():
        payload = gyro_bytes + trailer_bytes
        cluster_count = 1  # gyro
        if has_accel:
            payload += cluster_bytes
            cluster_count += 1
        if has_incl:
            payload += cluster_bytes
            cluster_count += 1
        if has_temp:
            payload += temp_bytes * cluster_count
        if has_pps:
            payload += pps_bytes
        # Pre-CRC bytes including the ID = 1 + payload. Dummy bytes pad to
        # the next 4-byte boundary (Table 5-22).
        pre_crc = 1 + payload
        dummy = (-pre_crc) % 4
        specs[dgid] = NormalFrameSpec(
            datagram_id=dgid,
            has_accel=has_accel,
            has_incl=has_incl,
            has_temp=has_temp,
            has_pps=has_pps,
            payload_length=payload,
            dummy_bytes=dummy,
        )
    return specs


NORMAL_SPECS: Dict[int, NormalFrameSpec] = _build_normal_specs()


# ----------------------------------------------------------------------------
# Init / on-demand datagram specs (interleaved into the Normal-Mode stream
# when the user issues N/I/C/T/E commands).
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class _SpecialFrameSpec:
    """Frame spec for an Init-Mode datagram that can appear mid-stream."""
    datagram_id: int
    payload_length: int
    dummy_bytes: int


SPECIAL_SPECS: Dict[int, _SpecialFrameSpec] = {
    PART_NUMBER_IDS[0]:    _SpecialFrameSpec(PART_NUMBER_IDS[0], PART_NUMBER_PAYLOAD_LENGTH, 0),
    PART_NUMBER_IDS[1]:    _SpecialFrameSpec(PART_NUMBER_IDS[1], PART_NUMBER_PAYLOAD_LENGTH, 0),
    SERIAL_NUMBER_IDS[0]:  _SpecialFrameSpec(SERIAL_NUMBER_IDS[0], SERIAL_NUMBER_PAYLOAD_LENGTH, 0),
    SERIAL_NUMBER_IDS[1]:  _SpecialFrameSpec(SERIAL_NUMBER_IDS[1], SERIAL_NUMBER_PAYLOAD_LENGTH, 0),
    # Configuration is 22 bytes pre-CRC (1 ID + 21 payload) so dummy is 2 (Table 5-22).
    0xBC:                  _SpecialFrameSpec(0xBC, 21, 2),
    0xBD:                  _SpecialFrameSpec(0xBD, 21, 2),
    BIAS_TRIM_IDS[0]:      _SpecialFrameSpec(BIAS_TRIM_IDS[0], BIAS_TRIM_PAYLOAD_LENGTH, 0),
    BIAS_TRIM_IDS[1]:      _SpecialFrameSpec(BIAS_TRIM_IDS[1], BIAS_TRIM_PAYLOAD_LENGTH, 0),
    EXTENDED_ERROR_IDS[0]: _SpecialFrameSpec(EXTENDED_ERROR_IDS[0], EXTENDED_ERROR_PAYLOAD_LENGTH, 3),
    EXTENDED_ERROR_IDS[1]: _SpecialFrameSpec(EXTENDED_ERROR_IDS[1], EXTENDED_ERROR_PAYLOAD_LENGTH, 3),
}

# IDs whose presence on the wire implies CR+LF termination follows the CRC.
_CRLF_SPECIAL_IDS = {PART_NUMBER_IDS[1], SERIAL_NUMBER_IDS[1], 0xBD,
                     BIAS_TRIM_IDS[1], EXTENDED_ERROR_IDS[1]}


def frame_length_for(datagram_id: int, crlf: bool) -> int:
    """Return the total wire length of a Normal-Mode frame for a given ID.

    Used by ``Configuration.frame_length()``; lives here so the catalogue
    has a single owner.
    """
    if datagram_id not in NORMAL_SPECS:
        raise ValueError("unknown Normal-Mode datagram ID 0x{0:02X}".format(datagram_id))
    return NORMAL_SPECS[datagram_id].frame_length(crlf=crlf)


# ----------------------------------------------------------------------------
# Measurement dataclass and unit dispatchers
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Measurement:
    """A single Normal-Mode measurement record.

    Cluster fields are populated only when the configured datagram ID
    includes that cluster (Table 5-21). The numeric units depend on the
    configured output units:

    * gyro:  ``[deg/s]`` for ANGULAR_RATE / AVERAGE; ``[deg]`` for
      INCREMENTAL_ANGLE / INTEGRATED_ANGLE.
    * accel: ``[g]`` for ACCELERATION / AVERAGE; ``[m/s/sample]`` for
      INCREMENTAL / INTEGRATED VELOCITY (or ``[gs]`` for INTEGRATED_GS).
    * incl:  same options as accel.
    * temperature: always ``[degC]``.
    * pps:   ``[us]`` for TIME_SINCE_DETECTION_*; unitless [0, 1] for
      FILTERED; ``None`` otherwise.
    * latency_us: always microseconds (Eq. 10).
    """

    datagram_id: int
    counter: int                 # 0..255 (Â§7.5.2.2.18)
    latency_us: int              # 0..65535
    gyro: Vec3
    gyro_status: StatusByte
    accel: Optional[Vec3] = None
    accel_status: Optional[StatusByte] = None
    incl: Optional[Vec3] = None
    incl_status: Optional[StatusByte] = None
    gyro_temp: Optional[Vec3] = None
    gyro_temp_status: Optional[StatusByte] = None
    accel_temp: Optional[Vec3] = None
    accel_temp_status: Optional[StatusByte] = None
    incl_temp: Optional[Vec3] = None
    incl_temp_status: Optional[StatusByte] = None
    pps: Optional[float] = None
    pps_status: Optional[StatusByte] = None


def _gyro_decoder(unit_code: int) -> Callable[[bytes], float]:
    # Bit 3 of the unit code is the "delayed" marker; bits 1..0 select the form.
    base = unit_code & 0b0011
    if base in (0b00, 0b10):    # ANGULAR_RATE or AVERAGE_RATE
        return decode_gyro_angular_rate
    return decode_gyro_incremental_angle  # INCREMENTAL_ANGLE or INTEGRATED_ANGLE


def _gyro_encoder(unit_code: int) -> Callable[[float], bytes]:
    base = unit_code & 0b0011
    if base in (0b00, 0b10):
        return encode_gyro_angular_rate
    return encode_gyro_incremental_angle


def _accel_decoder(unit_code: int, range_g: int) -> Callable[[bytes], float]:
    if unit_code in (ACCEL_UNIT_ACCELERATION, ACCEL_UNIT_AVERAGE_ACCELERATION):
        return lambda data: decode_accel_g(data, range_g)
    # INCREMENTAL_VELOCITY, INTEGRATED_VELOCITY_GS, INTEGRATED_VELOCITY_MS
    return lambda data: decode_accel_incremental_velocity(data, range_g)


def _accel_encoder(unit_code: int, range_g: int) -> Callable[[float], bytes]:
    if unit_code in (ACCEL_UNIT_ACCELERATION, ACCEL_UNIT_AVERAGE_ACCELERATION):
        return lambda v: encode_accel_g(v, range_g)
    return lambda v: encode_accel_incremental_velocity(v, range_g)


def _incl_decoder(unit_code: int) -> Callable[[bytes], float]:
    if unit_code in (ACCEL_UNIT_ACCELERATION, ACCEL_UNIT_AVERAGE_ACCELERATION):
        return decode_incl_g
    return decode_incl_incremental_velocity


def _incl_encoder(unit_code: int) -> Callable[[float], bytes]:
    if unit_code in (ACCEL_UNIT_ACCELERATION, ACCEL_UNIT_AVERAGE_ACCELERATION):
        return encode_incl_g
    return encode_incl_incremental_velocity


def _pps_decoder(unit_code: int) -> Callable[[bytes], float]:
    if unit_code in (PPS_UNIT_FILTERED, PPS_UNIT_FILTERED_DELAYED):
        return decode_pps_filtered
    # Both TIME_SINCE_* variants return ints (microseconds); cast to float.
    return lambda data: float(decode_pps_time_since(data))


def _pps_encoder(unit_code: int) -> Callable[[float], bytes]:
    if unit_code in (PPS_UNIT_FILTERED, PPS_UNIT_FILTERED_DELAYED):
        return encode_pps_filtered
    return lambda v: encode_pps_time_since(int(round(v)))


def _decode_vec3(data: bytes, decoder: Callable[[bytes], float]) -> Vec3:
    return (decoder(data[0:3]), decoder(data[3:6]), decoder(data[6:9]))


def _decode_temp_triple(data: bytes) -> Vec3:
    return (decode_temperature(data[0:2]),
            decode_temperature(data[2:4]),
            decode_temperature(data[4:6]))


# ----------------------------------------------------------------------------
# Frame builder + parser (per-frame; the streaming wrapper sits on top)
# ----------------------------------------------------------------------------

def build_normal_frame(measurement: Measurement, configuration: Configuration) -> bytes:
    """Build the complete wire bytes for a Normal-Mode datagram.

    Includes the ID byte, payload, 4-byte big-endian CRC, and CR+LF if the
    configuration has line termination enabled. Used by tests to construct
    round-trip-safe byte sequences from a known Measurement.
    """
    spec = NORMAL_SPECS[configuration.datagram_id]
    payload = _build_payload(measurement, spec, configuration)
    frame = bytes([spec.datagram_id]) + payload
    padded = frame + b"\x00" * spec.dummy_bytes
    crc = crc32_stim300(padded)
    frame = frame + crc.to_bytes(4, "big")
    if configuration.crlf_termination:
        frame = frame + b"\r\n"
    return frame


def _build_payload(measurement: Measurement, spec: NormalFrameSpec,
                    configuration: Configuration) -> bytes:
    range_g = configuration.accel_range_g[0]  # the parser uses X range uniformly
    gyro_enc = _gyro_encoder(configuration.gyro_output_unit)
    accel_enc = _accel_encoder(configuration.accel_output_unit, range_g)
    incl_enc = _incl_encoder(configuration.incl_output_unit)
    pps_enc = _pps_encoder(configuration.pps_output_unit)

    buf = bytearray()
    # gyro cluster
    for v in measurement.gyro:
        buf += gyro_enc(v)
    buf.append(measurement.gyro_status.raw)
    if spec.has_accel:
        if measurement.accel is None or measurement.accel_status is None:
            raise ProtocolError("datagram 0x{0:02X} requires accel cluster".format(spec.datagram_id))
        for v in measurement.accel:
            buf += accel_enc(v)
        buf.append(measurement.accel_status.raw)
    if spec.has_incl:
        if measurement.incl is None or measurement.incl_status is None:
            raise ProtocolError("datagram 0x{0:02X} requires incl cluster".format(spec.datagram_id))
        for v in measurement.incl:
            buf += incl_enc(v)
        buf.append(measurement.incl_status.raw)
    if spec.has_temp:
        # One temperature triple per present cluster (Â§5.5.6 final paragraph).
        for triple, status in [
            (measurement.gyro_temp, measurement.gyro_temp_status),
            (measurement.accel_temp if spec.has_accel else None,
             measurement.accel_temp_status if spec.has_accel else None),
            (measurement.incl_temp if spec.has_incl else None,
             measurement.incl_temp_status if spec.has_incl else None),
        ]:
            if triple is None:
                continue
            for v in triple:
                buf += encode_temperature(v)
            buf.append(status.raw if status is not None else 0)
    if spec.has_pps:
        if measurement.pps is None or measurement.pps_status is None:
            raise ProtocolError("datagram 0x{0:02X} requires PPS".format(spec.datagram_id))
        buf += pps_enc(measurement.pps)
        buf.append(measurement.pps_status.raw)
    buf.append(measurement.counter & 0xFF)
    buf += encode_latency_us(measurement.latency_us)
    return bytes(buf)


def parse_normal_payload(payload: bytes, spec: NormalFrameSpec,
                          configuration: Configuration) -> Measurement:
    """Parse the payload (bytes between ID and CRC) into a Measurement."""
    if len(payload) != spec.payload_length:
        raise ProtocolError("payload length mismatch for ID 0x{0:02X}: expected {1}, got {2}".format(
            spec.datagram_id, spec.payload_length, len(payload)))
    range_g = configuration.accel_range_g[0]
    gyro_dec = _gyro_decoder(configuration.gyro_output_unit)
    accel_dec = _accel_decoder(configuration.accel_output_unit, range_g)
    incl_dec = _incl_decoder(configuration.incl_output_unit)
    pps_dec = _pps_decoder(configuration.pps_output_unit)

    cursor = 0
    gyro = _decode_vec3(payload[cursor:cursor + 9], gyro_dec)
    cursor += 9
    gyro_status = StatusByte.decode(payload[cursor])
    cursor += 1

    accel: Optional[Vec3] = None
    accel_status: Optional[StatusByte] = None
    if spec.has_accel:
        accel = _decode_vec3(payload[cursor:cursor + 9], accel_dec)
        cursor += 9
        accel_status = StatusByte.decode(payload[cursor])
        cursor += 1

    incl: Optional[Vec3] = None
    incl_status: Optional[StatusByte] = None
    if spec.has_incl:
        incl = _decode_vec3(payload[cursor:cursor + 9], incl_dec)
        cursor += 9
        incl_status = StatusByte.decode(payload[cursor])
        cursor += 1

    gyro_temp: Optional[Vec3] = None
    gyro_temp_status: Optional[StatusByte] = None
    accel_temp: Optional[Vec3] = None
    accel_temp_status: Optional[StatusByte] = None
    incl_temp: Optional[Vec3] = None
    incl_temp_status: Optional[StatusByte] = None
    if spec.has_temp:
        # Always: gyro temp
        gyro_temp = _decode_temp_triple(payload[cursor:cursor + 6])
        cursor += 6
        gyro_temp_status = StatusByte.decode(payload[cursor])
        cursor += 1
        if spec.has_accel:
            accel_temp = _decode_temp_triple(payload[cursor:cursor + 6])
            cursor += 6
            accel_temp_status = StatusByte.decode(payload[cursor])
            cursor += 1
        if spec.has_incl:
            incl_temp = _decode_temp_triple(payload[cursor:cursor + 6])
            cursor += 6
            incl_temp_status = StatusByte.decode(payload[cursor])
            cursor += 1

    pps: Optional[float] = None
    pps_status: Optional[StatusByte] = None
    if spec.has_pps:
        pps = pps_dec(payload[cursor:cursor + 3])
        cursor += 3
        pps_status = StatusByte.decode(payload[cursor])
        cursor += 1

    counter = payload[cursor]
    cursor += 1
    latency_us = decode_latency_us(payload[cursor:cursor + 2])
    cursor += 2

    if cursor != spec.payload_length:
        raise ProtocolError("trailing bytes after parse: cursor={0}, expected={1}".format(
            cursor, spec.payload_length))

    return Measurement(
        datagram_id=spec.datagram_id,
        counter=counter,
        latency_us=latency_us,
        gyro=gyro,
        gyro_status=gyro_status,
        accel=accel,
        accel_status=accel_status,
        incl=incl,
        incl_status=incl_status,
        gyro_temp=gyro_temp,
        gyro_temp_status=gyro_temp_status,
        accel_temp=accel_temp,
        accel_temp_status=accel_temp_status,
        incl_temp=incl_temp,
        incl_temp_status=incl_temp_status,
        pps=pps,
        pps_status=pps_status,
    )


# ----------------------------------------------------------------------------
# Streaming framer
# ----------------------------------------------------------------------------

# A parsed record from the stream is one of:
#   Measurement
#   PartNumberDatagram, SerialNumberDatagram, ConfigurationDatagram,
#   BiasTrimDatagram, ExtendedErrorDatagram
# All are returned through the same iterator from NormalStreamParser.feed().


def _validate_crc(payload_with_id: bytes, dummy: int, crc_bytes: bytes) -> bool:
    """True iff the CRC of ``payload_with_id + dummy*0x00`` matches ``crc_bytes``."""
    expected = crc32_stim300(payload_with_id + b"\x00" * dummy)
    actual = int.from_bytes(crc_bytes, "big")
    return expected == actual


@dataclass(frozen=True)
class _CandidateMatch:
    """Internal: the layout for one decode attempt at the head of the buffer."""
    datagram_id: int
    payload_length: int
    dummy_bytes: int
    crlf: bool
    is_special: bool

    @property
    def total_length(self) -> int:
        return 1 + self.payload_length + 4 + (2 if self.crlf else 0)


def _candidates_for(head: int, configuration: Configuration) -> List[_CandidateMatch]:
    """Return all plausible frame layouts that could start with ``head``.

    The expected Normal-Mode ID is always offered first (with the configured
    CRLF flag). Any Init / on-demand special ID (interleaved by N/I/C/T/E
    commands) is offered too, in both CRLF and non-CRLF variants if the
    datasheet defines both.
    """
    cands: List[_CandidateMatch] = []
    if head == configuration.datagram_id:
        spec = NORMAL_SPECS[head]
        cands.append(_CandidateMatch(
            datagram_id=head,
            payload_length=spec.payload_length,
            dummy_bytes=spec.dummy_bytes,
            crlf=configuration.crlf_termination,
            is_special=False,
        ))
    if head in SPECIAL_SPECS:
        spec = SPECIAL_SPECS[head]
        cands.append(_CandidateMatch(
            datagram_id=head,
            payload_length=spec.payload_length,
            dummy_bytes=spec.dummy_bytes,
            crlf=head in _CRLF_SPECIAL_IDS,
            is_special=True,
        ))
    return cands


def _parse_special(datagram_id: int, payload: bytes):
    """Dispatch a special-datagram payload to its decoder."""
    if datagram_id in PART_NUMBER_IDS:
        return PartNumberDatagram.parse(payload)
    if datagram_id in SERIAL_NUMBER_IDS:
        return SerialNumberDatagram.parse(payload)
    if datagram_id in (0xBC, 0xBD):
        return _decode_configuration_payload(payload)
    if datagram_id in BIAS_TRIM_IDS:
        return BiasTrimDatagram.parse(payload)
    if datagram_id in EXTENDED_ERROR_IDS:
        return ExtendedErrorDatagram.parse(payload)
    raise ProtocolError("no parser for special ID 0x{0:02X}".format(datagram_id))


class NormalStreamParser:
    """Pull-based streaming parser for Normal-Mode byte streams.

    Usage::

        parser = NormalStreamParser(configuration)
        for chunk in transport_reads:
            for record in parser.feed(chunk):
                handle(record)

    The parser owns a small ``bytearray`` that accumulates between calls.
    It yields ``Measurement`` records for Normal-Mode frames matching the
    configured ID, and the corresponding special-datagram dataclasses
    (PartNumber, SerialNumber, Configuration, BiasTrim, ExtendedError) for
    Init / on-demand frames that the device interleaves into the stream.

    On a frame whose CRC fails the parser advances one byte and re-attempts
    framing at the new head until a valid frame is found. The number of
    resync events (and bytes dropped) is exposed via ``resync_events`` /
    ``bytes_discarded`` for monitoring.
    """

    def __init__(self, configuration: Configuration) -> None:
        self._configuration = configuration
        self._buffer = bytearray()
        self.resync_events = 0
        self.bytes_discarded = 0

    @property
    def configuration(self) -> Configuration:
        return self._configuration

    def update_configuration(self, configuration: Configuration) -> None:
        """Swap the active configuration mid-stream.

        Used after a successful Service-Mode reconfiguration. The buffer is
        kept; the next frame attempt will use the new datagram ID.
        """
        self._configuration = configuration

    def feed(self, data: bytes) -> Iterator[object]:
        """Append ``data`` to the internal buffer and yield any complete records."""
        if data:
            self._buffer.extend(data)
        while True:
            record = self._try_parse_head()
            if record is _NEED_MORE:
                return
            if record is None:
                # CRC failed at head; resync (drop one byte).
                self._buffer.pop(0)
                self.bytes_discarded += 1
                self.resync_events += 1
                continue
            yield record

    def _try_parse_head(self):
        """Attempt to decode one frame starting at buffer[0].

        Returns:
            ``_NEED_MORE``: not enough bytes; caller should wait.
            ``None``: head byte is unrecognized or CRC fails; caller resyncs.
            a parsed record: a valid frame was consumed.
        """
        if not self._buffer:
            return _NEED_MORE
        head = self._buffer[0]
        candidates = _candidates_for(head, self._configuration)
        if not candidates:
            return None
        for cand in candidates:
            if len(self._buffer) < cand.total_length:
                # Not enough bytes for this candidate. If *all* candidates are
                # in this state, we need more data; otherwise try the next.
                continue
            # ID + payload + CRC validates?
            id_and_payload = bytes(self._buffer[:1 + cand.payload_length])
            crc_bytes = bytes(self._buffer[1 + cand.payload_length:
                                            1 + cand.payload_length + 4])
            if not _validate_crc(id_and_payload, cand.dummy_bytes, crc_bytes):
                continue
            if cand.crlf:
                tail = bytes(self._buffer[cand.total_length - 2:cand.total_length])
                if tail != b"\r\n":
                    continue
            # Accept this candidate; build the record.
            payload = id_and_payload[1:]
            if cand.is_special:
                record = _parse_special(cand.datagram_id, payload)
            else:
                spec = NORMAL_SPECS[cand.datagram_id]
                record = parse_normal_payload(payload, spec, self._configuration)
            del self._buffer[:cand.total_length]
            return record
        # No candidate validated. If any candidate is incomplete, wait for more
        # data; otherwise the head must be wrong - signal resync.
        if any(len(self._buffer) < c.total_length for c in candidates):
            return _NEED_MORE
        return None


_NEED_MORE = object()
