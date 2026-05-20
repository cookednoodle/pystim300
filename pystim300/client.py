"""High-level STIM300 client.

Owns the transport, tracks the active device mode, manages the streaming
parser for Normal-Mode and the request/response loops for Service-Mode
and Utility-Mode, and fires audit events for every byte of ASCII I/O.

The codec layers are imported but never call back into this module - they
are pure functions and dataclasses. The client is the single place where
bytes flow on and off the wire.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, List, Optional, Tuple

from pystim300 import service as _service
from pystim300 import utility as _utility
from pystim300.configuration import Configuration
from pystim300.datagrams import (
    BiasTrimDatagram,
    PartNumberDatagram,
    SerialNumberDatagram,
)
from pystim300.exceptions import ModeError, ProtocolError, TimeoutError
from pystim300.normal import Measurement, NormalStreamParser
from pystim300.service import ServiceResponse
from pystim300.transport import Transport
from pystim300.utility import UtilityResponse


class Mode(Enum):
    """Device mode tracked by the client.

    ``UNKNOWN`` is the initial state until the client has either entered a
    specific mode or been told via ``set_mode``. Methods that require a
    specific mode raise ``ModeError`` if the tracked mode doesn't match.
    """

    UNKNOWN = "unknown"
    INIT = "init"
    NORMAL = "normal"
    SERVICE = "service"
    UTILITY = "utility"


@dataclass(frozen=True)
class AuditEvent:
    """One byte-level event from the Service or Utility ASCII protocols.

    Fires for every ``write`` (``direction="tx"``) and every complete
    response (``direction="rx"``). Normal-Mode binary I/O does NOT emit
    audit events - its volume is far too high and the per-frame
    ``Measurement`` records already provide the inspectable trace.
    """

    mode: Mode
    direction: str        # "tx" or "rx"
    payload: bytes
    timestamp: float


AuditCallback = Callable[[AuditEvent], None]


@dataclass(frozen=True)
class InitSequence:
    """The Init-Mode datagram sequence captured immediately after power-on.

    The STIM300 emits its Part Number, Serial Number, and Configuration
    datagrams within ~1 s of power-up, followed by the Bias Trim datagram
    iff ``bias_trim_at_startup`` is set in the configuration (§7.5.1, p.45).
    ``raw`` is the full byte stream that was drained from the transport
    during capture, retained for auditing.
    """

    part_number: PartNumberDatagram
    serial_number: SerialNumberDatagram
    configuration: Configuration
    bias_trim: Optional[BiasTrimDatagram]
    raw: bytes


class MemoryAuditor:
    """Ready-made auditor that collects ``AuditEvent`` records in memory.

    Bounded by ``max_events`` (oldest events drop). Pass an instance as
    the ``audit`` argument when constructing ``STIM300``; inspect via
    ``events`` after the session.
    """

    def __init__(self, max_events: Optional[int] = None) -> None:
        self._max = max_events
        self._events: List[AuditEvent] = []

    def __call__(self, event: AuditEvent) -> None:
        self._events.append(event)
        if self._max is not None and len(self._events) > self._max:
            del self._events[: len(self._events) - self._max]

    @property
    def events(self) -> Tuple[AuditEvent, ...]:
        return tuple(self._events)

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


_READ_CHUNK = 4096


class STIM300:
    """High-level driver for the Safran STIM300 IMU.

    Construct with a ``Transport`` (typically ``SerialTransport`` for real
    hardware, ``FakeTransport`` for tests). Optionally provide an
    ``audit`` callback to receive Service / Utility I/O events, and a
    ``configuration`` if the device's current configuration is already
    known (otherwise call ``read_configuration`` once after entering
    Normal Mode).
    """

    def __init__(
        self,
        transport: Transport,
        *,
        audit: Optional[AuditCallback] = None,
        configuration: Optional[Configuration] = None,
        timeout: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._transport = transport
        self._audit = audit
        self._timeout = timeout
        self._clock = clock
        self._mode = Mode.UNKNOWN
        self._configuration = configuration
        self._normal_parser: Optional[NormalStreamParser] = None
        if configuration is not None:
            self._normal_parser = NormalStreamParser(configuration)
            self._mode = Mode.NORMAL

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def configuration(self) -> Optional[Configuration]:
        return self._configuration

    @property
    def stream_parser(self) -> Optional[NormalStreamParser]:
        """The streaming parser used for Normal-Mode reads (None until configured).

        Public so the functional-checkout primitives can read its
        ``resync_events`` and ``bytes_discarded`` counters without poking
        a private attribute.
        """
        return self._normal_parser

    def set_configuration(self, configuration: Configuration) -> None:
        """Tell the client what the device's current configuration is.

        Used when the caller already knows the configuration (e.g. set via
        Service Mode in a previous session). The Normal-Mode parser is
        re-keyed accordingly.
        """
        self._configuration = configuration
        if self._normal_parser is None:
            self._normal_parser = NormalStreamParser(configuration)
        else:
            self._normal_parser.update_configuration(configuration)

    def set_mode(self, mode: Mode) -> None:
        """Force the tracked mode (advanced; usually managed by transitions)."""
        self._mode = mode

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "STIM300":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_audit(self, mode: Mode, direction: str, payload: bytes) -> None:
        if self._audit is None:
            return
        self._audit(AuditEvent(
            mode=mode,
            direction=direction,
            payload=payload,
            timestamp=self._clock(),
        ))

    def _write(self, mode: Mode, data: bytes) -> None:
        self._transport.write(data)
        self._emit_audit(mode, "tx", data)

    def _read_until(self, sentinel_finder: Callable[[bytes], int],
                     mode: Mode, *, timeout: Optional[float] = None) -> bytes:
        """Read from the transport until ``sentinel_finder`` returns >= 0.

        ``sentinel_finder(buffer)`` is invoked with the accumulated buffer
        after each read and returns the index of the first byte past the
        completion marker, or -1 if not yet present. Raises
        ``TimeoutError`` if no bytes arrive within ``timeout`` seconds AND
        the buffer is still incomplete.
        """
        deadline = None
        if timeout is None:
            timeout = self._timeout
        if timeout is not None:
            deadline = self._clock() + timeout
        buffer = bytearray(self._carryover)
        self._carryover = b""
        while True:
            end = sentinel_finder(bytes(buffer))
            if end >= 0:
                consumed = bytes(buffer[:end])
                # Anything past ``end`` is leftover for the next read; the
                # service/utility protocols never deliver more than one
                # response at a time, so this is rare but we preserve it
                # in a small carry-over buffer.
                self._carryover = bytes(buffer[end:])
                self._emit_audit(mode, "rx", consumed)
                return consumed
            remaining = None
            if deadline is not None:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise TimeoutError(
                        "no complete response within {0}s; got {1!r}".format(
                            timeout, bytes(buffer)))
            chunk = self._transport.read(_READ_CHUNK, timeout=remaining)
            if not chunk:
                if deadline is None:
                    # No timeout requested but the transport returned empty:
                    # treat it as end-of-stream / no progress possible.
                    raise TimeoutError("transport returned no data and no timeout was set")
                continue
            buffer.extend(chunk)

    _carryover: bytes = b""

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def enter_service(self, *, timeout: Optional[float] = None) -> ServiceResponse:
        """Send ``SERVICEMODE\\r``; read the entry banner; switch to Service mode.

        Per §7.5.3 (p.55) the device replies with its full configuration
        banner (same content as the ``i`` command, Figure 9-21). The
        banner ends with the ``\\r>`` prompt, which is the framing marker
        for all subsequent Service-Mode reads.
        """
        if self._mode not in (Mode.NORMAL, Mode.UNKNOWN):
            raise ModeError("enter_service requires NORMAL mode, currently {0}".format(self._mode))
        self._carryover = b""
        self._write(Mode.NORMAL, _service.SERVICEMODE_ENTRY)
        raw = self._read_until(_service.find_response_end, Mode.SERVICE, timeout=timeout)
        self._mode = Mode.SERVICE
        return _service.parse_response(raw, "SERVICEMODE")

    def exit_service(self, *, to_init: bool = False,
                      timeout: Optional[float] = None) -> ServiceResponse:
        """Send ``x N\\r`` (Normal) or ``x I\\r`` (Init); switch mode.

        Unlike every other Service-Mode command the EXIT response is not
        terminated by the ``\\r>`` prompt - the device leaves Service
        Mode and resumes Normal- (or Init-) Mode traffic (§9.14, Figures
        9-49 / 9-50). On success the device emits ``SYSTEM RETURNING TO
        <mode> MODE.``; a rejected parameter instead yields an
        ``E<nnn>`` error and the device stays in Service Mode (so the
        ``CommandError`` is raised before the tracked mode changes).
        """
        if self._mode != Mode.SERVICE:
            raise ModeError("exit_service requires SERVICE mode, currently {0}".format(self._mode))
        cmd = _service.cmd_exit(to_normal=not to_init)
        self._write(Mode.SERVICE, _service.encode_command(cmd))
        raw = self._read_until(_service.find_exit_response_end, Mode.SERVICE,
                                 timeout=timeout)
        response = _service.parse_exit_response(raw, cmd)
        _service.raise_for_error(response)
        self._mode = Mode.INIT if to_init else Mode.NORMAL
        if self._normal_parser is not None and self._configuration is not None:
            # Re-key so the next stream chunk starts fresh.
            self._normal_parser.update_configuration(self._configuration)
        return response

    def service_command(self, command: str, *,
                          timeout: Optional[float] = None) -> ServiceResponse:
        """Send a Service-Mode command and return its response.

        Raises ``CommandError`` if the device reports ``E<nnn>``.
        """
        if self._mode != Mode.SERVICE:
            raise ModeError("service_command requires SERVICE mode, currently {0}".format(self._mode))
        wire = _service.encode_command(command)
        self._write(Mode.SERVICE, wire)
        raw = self._read_until(_service.find_response_end, Mode.SERVICE, timeout=timeout)
        response = _service.parse_response(raw, command)
        _service.raise_for_error(response)
        return response

    def enter_utility(self, *, timeout: Optional[float] = None) -> UtilityResponse:
        """Send ``UTILITYMODE\\r``; read the ``#UTILITYMODE,234\\r`` acknowledgement."""
        if self._mode not in (Mode.NORMAL, Mode.UNKNOWN):
            raise ModeError("enter_utility requires NORMAL mode, currently {0}".format(self._mode))
        self._carryover = b""
        self._write(Mode.NORMAL, _utility.UTILITYMODE_ENTRY)
        raw = self._read_until(_utility.find_response_end, Mode.UTILITY, timeout=timeout)
        response = _utility.parse_response(raw)
        self._mode = Mode.UTILITY
        return response

    def exit_utility(self, *, timeout: Optional[float] = None) -> UtilityResponse:
        """Send ``$xn,<crc>\\r``; switch back to Normal mode."""
        if self._mode != Mode.UTILITY:
            raise ModeError("exit_utility requires UTILITY mode, currently {0}".format(self._mode))
        response = self.utility_command_raw(_utility.cmd_exit_to_normal(),
                                              command_name="xn", timeout=timeout)
        self._mode = Mode.NORMAL
        return response

    def utility_command(self, command: str, *params: object,
                          timeout: Optional[float] = None) -> UtilityResponse:
        """Send a Utility-Mode command and return its parsed response.

        Raises ``CommandError`` if the device reports a non-zero status.
        """
        if self._mode != Mode.UTILITY:
            raise ModeError("utility_command requires UTILITY mode, currently {0}".format(self._mode))
        wire = _utility.encode_command(command, *params)
        return self.utility_command_raw(wire, command_name=command, timeout=timeout)

    def utility_command_raw(self, wire: bytes, *, command_name: str,
                              timeout: Optional[float] = None) -> UtilityResponse:
        """Send pre-encoded Utility-Mode wire bytes; parse + raise on error.

        Used internally for ``$xn`` (which must be sent regardless of the
        tracked mode flag during exit handling).
        """
        self._write(self._mode, wire)
        raw = self._read_until(_utility.find_response_end, self._mode, timeout=timeout)
        response = _utility.parse_response(raw)
        _utility.raise_for_error(response)
        return response

    # ------------------------------------------------------------------
    # Normal-Mode commands (single-byte triggers; responses interleave
    # into the Measurement stream as special datagrams).
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Send the Normal-Mode ``R\\r`` command (full device reset).

        After this the device emits the Init-Mode datagram sequence
        before Normal-Mode resumes. The client transitions to
        ``Mode.INIT``; call ``set_mode(Mode.NORMAL)`` after the first
        post-reset Measurement is seen.
        """
        if self._mode not in (Mode.NORMAL, Mode.UNKNOWN):
            raise ModeError("reset requires NORMAL mode, currently {0}".format(self._mode))
        self._transport.write(b"R\r")
        self._mode = Mode.INIT

    def request_part_number(self) -> None:
        self._normal_trigger(b"N\r")

    def request_serial_number(self) -> None:
        self._normal_trigger(b"I\r")

    def request_configuration(self) -> None:
        self._normal_trigger(b"C\r")

    def request_bias_trim(self) -> None:
        self._normal_trigger(b"T\r")

    def request_extended_error(self) -> None:
        self._normal_trigger(b"E\r")

    def _normal_trigger(self, wire: bytes) -> None:
        if self._mode not in (Mode.NORMAL, Mode.UNKNOWN):
            raise ModeError("Normal-Mode command requires NORMAL mode, currently {0}".format(self._mode))
        self._transport.write(wire)

    # ------------------------------------------------------------------
    # Init-Mode capture
    # ------------------------------------------------------------------

    def read_init_sequence(self, *, timeout: float = 10.0) -> "InitSequence":
        """Drain Init-Mode datagrams from the transport into an ``InitSequence``.

        Run this immediately after the device powers up (and after the
        transport is opened). The STIM300 emits Part Number, Serial
        Number, and Configuration datagrams within ~1 s of power-up; if
        ``bias_trim_at_startup`` is set in the Configuration, a Bias Trim
        datagram follows. The method also calls ``set_configuration`` so
        the client is ready for Normal-Mode reads afterwards.

        Raises ``TimeoutError`` if the full sequence does not arrive
        within ``timeout`` seconds; the partial set of datagrams that
        *did* arrive (if any) is attached to the exception's ``args``
        tuple for diagnosis.
        """
        if self._mode not in (Mode.INIT, Mode.UNKNOWN, Mode.NORMAL):
            raise ModeError("read_init_sequence requires INIT/UNKNOWN/NORMAL mode, "
                            "currently {0}".format(self._mode))
        parser = NormalStreamParser(configuration=None)
        deadline = self._clock() + timeout
        captured_raw = bytearray()
        part_number: Optional[PartNumberDatagram] = None
        serial_number: Optional[SerialNumberDatagram] = None
        configuration: Optional[Configuration] = None
        bias_trim: Optional[BiasTrimDatagram] = None

        def _complete() -> bool:
            if part_number is None or serial_number is None or configuration is None:
                return False
            if configuration.bias_trim_at_startup and bias_trim is None:
                return False
            return True

        # Bytes left over from a prior read (e.g. the Init datagrams that
        # arrive right after exit_service(to_init=True)) must be consumed
        # before reading from the transport, or the first datagrams would
        # be lost.
        pending = self._carryover
        self._carryover = b""

        complete = False
        while not complete:
            if pending:
                chunk = pending
                pending = b""
            else:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    self._raise_init_timeout(timeout, bytes(captured_raw),
                                              part_number, serial_number,
                                              configuration, bias_trim)
                chunk = self._transport.read(_READ_CHUNK, timeout=remaining)
                if not chunk:
                    if self._clock() >= deadline:
                        self._raise_init_timeout(timeout, bytes(captured_raw),
                                                  part_number, serial_number,
                                                  configuration, bias_trim)
                    continue
            captured_raw.extend(chunk)
            for record in parser.feed(chunk):
                if isinstance(record, PartNumberDatagram):
                    part_number = record
                elif isinstance(record, SerialNumberDatagram):
                    serial_number = record
                elif isinstance(record, Configuration):
                    configuration = record
                    # Once we know the configuration, key the streaming
                    # parser used for subsequent read_records calls.
                    self.set_configuration(record)
                elif isinstance(record, BiasTrimDatagram):
                    bias_trim = record
                # Other record types are ignored during Init capture.
                if _complete():
                    complete = True
                    break
            if complete:
                # Anything the temp parser hasn't consumed yet is real
                # Normal-Mode data that arrived in the same read as the
                # tail of the Init sequence. Hand it to the client's
                # carry-over so the next read_records call sees it.
                remaining_buf = bytes(parser._buffer)
                if remaining_buf:
                    self._carryover = remaining_buf + self._carryover

        assert part_number is not None
        assert serial_number is not None
        assert configuration is not None
        self._mode = Mode.NORMAL
        return InitSequence(
            part_number=part_number,
            serial_number=serial_number,
            configuration=configuration,
            bias_trim=bias_trim,
            raw=bytes(captured_raw),
        )

    @staticmethod
    def _raise_init_timeout(timeout: float, raw: bytes,
                              pn: Optional[PartNumberDatagram],
                              sn: Optional[SerialNumberDatagram],
                              cfg: Optional[Configuration],
                              bt: Optional[BiasTrimDatagram]) -> None:
        missing = []
        if pn is None:
            missing.append("PartNumber")
        if sn is None:
            missing.append("SerialNumber")
        if cfg is None:
            missing.append("Configuration")
        elif cfg.bias_trim_at_startup and bt is None:
            missing.append("BiasTrim")
        raise TimeoutError(
            "Init-Mode sequence incomplete after {0}s; missing: {1}; got {2} raw bytes".format(
                timeout, ", ".join(missing) or "(none)", len(raw)),
            raw, pn, sn, cfg, bt,
        )

    # ------------------------------------------------------------------
    # Normal-Mode streaming
    # ------------------------------------------------------------------

    def read_records(self, limit: Optional[int] = None,
                       *, timeout: Optional[float] = None) -> Iterator[object]:
        """Iterate over decoded records from the Normal-Mode byte stream.

        Yields ``Measurement`` instances for Normal-Mode frames and the
        corresponding ``PartNumberDatagram`` / ``SerialNumberDatagram`` /
        ``Configuration`` / ``BiasTrimDatagram`` / ``ExtendedErrorDatagram``
        for interleaved special datagrams.

        ``limit`` bounds the total number of records yielded. ``timeout``
        bounds each underlying read; an empty read raises ``TimeoutError``.
        """
        if self._normal_parser is None or self._configuration is None:
            raise ModeError("read_records requires a known configuration; "
                            "call set_configuration first or read Init-Mode datagrams")
        if self._mode not in (Mode.NORMAL, Mode.INIT, Mode.UNKNOWN):
            raise ModeError("read_records requires NORMAL/INIT mode, currently {0}".format(self._mode))
        deadline = None
        if timeout is None:
            timeout = self._timeout
        yielded = 0
        # First, drain any carryover from a previous Service/Utility read.
        if self._carryover:
            for record in self._normal_parser.feed(self._carryover):
                yield record
                yielded += 1
                if limit is not None and yielded >= limit:
                    self._carryover = b""
                    return
            self._carryover = b""
        while limit is None or yielded < limit:
            if timeout is not None:
                deadline = self._clock() + timeout
                remaining = timeout
            else:
                remaining = None
            chunk = self._transport.read(_READ_CHUNK, timeout=remaining)
            if not chunk:
                if deadline is not None and self._clock() >= deadline:
                    raise TimeoutError("no data within {0}s".format(timeout))
                continue
            for record in self._normal_parser.feed(chunk):
                yield record
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    def read_measurements(self, limit: Optional[int] = None,
                            *, timeout: Optional[float] = None,
                            include_startup: bool = True) -> Iterator[Measurement]:
        """Filter ``read_records`` to ``Measurement`` instances only.

        If ``include_startup`` is False, measurements with the ``startup``
        bit in the gyro status are dropped (per §7.5.2.1, valid output
        begins ~0.5 s after Init->Normal).
        """
        for record in self.read_records(limit=None, timeout=timeout):
            if not isinstance(record, Measurement):
                continue
            if not include_startup and record.gyro_status.startup:
                continue
            yield record
            if limit is not None:
                limit -= 1
                if limit <= 0:
                    return
