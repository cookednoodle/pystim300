"""Utility-Mode protocol (pure encode/decode; I/O lives on the client).

Utility Mode is a machine-to-machine ASCII protocol with per-message
CRC-8. Each command is::

    $<cmd>[,<param>...],<crc8>\\r

Each response is::

    #<cmd>,<status>[,<data>...],<crc8>\\r

The CRC-8 is computed over the ASCII bytes of the preceding string,
**including** the leading ``$`` or ``#`` and **including** the final
comma before the CRC field; it is rendered as decimal ASCII (e.g.
``"28"`` for value 28). See §10.2.3 (p.99).

Status codes (§10.2.4, Table 10-2, p.100):
  0 = OK; 1..8 = various errors (mapped to ``CommandError`` codes).

The Utility-Mode entry acknowledgement is a special-format response::

    #UTILITYMODE,<crc8>\\r

(no status field). See §10.1 (p.99).

This module is I/O-free.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from pystim300.crc import crc8_stim300
from pystim300.exceptions import CommandError, CrcError, ProtocolError

UTILITYMODE_ENTRY = b"UTILITYMODE\r"
UTILITY_TERMINATOR = b"\r"

# Status-code descriptions per Table 10-2 (p.100).
STATUS_DESCRIPTIONS = {
    0: "OK",
    1: "Invalid command",
    2: "Incorrect CRC",
    3: "Unknown command",
    4: "Incorrect number of parameters",
    5: "Invalid parameter(s)",
    6: "Exceeded maximum number of saves",
    7: "Error during save",
    8: "Requested change(s) reduced due to violation of min/max bias trim offsets",
}


@dataclass(frozen=True)
class UtilityResponse:
    """Decoded Utility-Mode response.

    Attributes:
        command: The command name echoed by the device (e.g. ``"isn"``).
            Empty string when the device could not identify the command
            (status 1 / 3 - Figure 10-3, p.100).
        status: Status code per Table 10-2; ``0`` is success.
        fields: Extra return values (strings) between status and CRC.
        raw: The exact bytes received, including the trailing CR - for
            auditing and inspection.
    """

    command: str
    status: int
    fields: Tuple[str, ...]
    raw: bytes


def encode_command(command: str, *params: object) -> bytes:
    """Build the wire bytes for a Utility-Mode command.

    ``command`` is the bare name without the leading ``$`` (e.g.
    ``"isn"`` or ``"sm"``). ``params`` are stringified with ``str()``;
    floats can be passed directly. Per §10.2.1 (p.99) the result is
    ``$<cmd>[,<param>...],<crc8>\\r`` with the CRC over everything up
    to and including the final comma.
    """
    if not command or "," in command or "\r" in command or "\n" in command:
        raise ValueError("invalid Utility-Mode command name: {0!r}".format(command))
    parts = ["$" + command]
    for p in params:
        parts.append(str(p))
    body = ",".join(parts) + ","
    crc = crc8_stim300(body.encode("ascii"))
    return (body + str(crc)).encode("ascii") + UTILITY_TERMINATOR


def find_response_end(buffer: bytes) -> int:
    """Return the index of the first byte AFTER a complete response.

    Utility-Mode responses are terminated by a single CR (§10.2.2). Returns
    -1 if no terminator has arrived yet.
    """
    idx = buffer.find(UTILITY_TERMINATOR)
    if idx < 0:
        return -1
    return idx + len(UTILITY_TERMINATOR)


def parse_response(raw: bytes) -> UtilityResponse:
    """Parse the bytes of one complete Utility-Mode response.

    Validates the CRC-8 against the embedded checksum and raises
    ``CrcError`` if it does not match. Does **not** raise on a non-zero
    status code; callers use ``raise_for_error`` for that so they can
    inspect the response first if needed.
    """
    if not raw.endswith(UTILITY_TERMINATOR):
        raise ProtocolError("Utility response missing CR: {0!r}".format(raw))
    body = raw[:-len(UTILITY_TERMINATOR)]
    if not body.startswith(b"#"):
        raise ProtocolError("Utility response must start with '#': {0!r}".format(raw))
    # Split off the trailing CRC.
    last_comma = body.rfind(b",")
    if last_comma < 0:
        raise ProtocolError("Utility response has no CRC field: {0!r}".format(raw))
    crc_for = body[:last_comma + 1]  # includes the trailing comma
    crc_str = body[last_comma + 1:]
    try:
        actual_crc = int(crc_str)
    except ValueError as exc:
        raise ProtocolError(
            "Utility response CRC field is not an integer: {0!r}".format(crc_str)
        ) from exc
    expected_crc = crc8_stim300(crc_for)
    if expected_crc != actual_crc:
        raise CrcError(
            "Utility CRC mismatch: expected {0}, got {1}".format(expected_crc, actual_crc),
            expected=expected_crc, actual=actual_crc,
        )

    # Decode the body fields.
    text = body.decode("ascii")
    parts = text.split(",")
    # parts[0] starts with '#'; everything between is field data; last is CRC.
    command = parts[0][1:]
    payload = parts[1:-1]

    # Entry acknowledgement: #UTILITYMODE,<crc> - no status field.
    if command == "UTILITYMODE" and len(payload) == 0:
        return UtilityResponse(command=command, status=0, fields=(), raw=bytes(raw))

    if not payload:
        raise ProtocolError("Utility response has no status: {0!r}".format(raw))
    try:
        status = int(payload[0])
    except ValueError as exc:
        raise ProtocolError(
            "Utility response status field is not an integer: {0!r}".format(payload[0])
        ) from exc
    return UtilityResponse(
        command=command,
        status=status,
        fields=tuple(payload[1:]),
        raw=bytes(raw),
    )


def raise_for_error(response: UtilityResponse) -> UtilityResponse:
    """Raise ``CommandError`` if ``response.status != 0``; else return it."""
    if response.status == 0:
        return response
    description = STATUS_DESCRIPTIONS.get(response.status, "unknown")
    raise CommandError(
        "Utility-Mode status {0}: {1}".format(response.status, description),
        code=response.status,
        command=response.command,
        raw=response.raw,
    )


# ---------------------------------------------------------------------------
# Convenience builders for the most-used commands.
# Full catalogue is in Table 10-1 (pp.98-99).
# ---------------------------------------------------------------------------

def cmd_information(name: str) -> bytes:
    """Build any ``$i<name>`` general-query command (§10.3.x)."""
    if not name.startswith("i"):
        raise ValueError("information command must start with 'i', got {0!r}".format(name))
    return encode_command(name)


def cmd_exit_to_normal() -> bytes:
    """Build the ``$xn`` command - leave Utility Mode (§10.3.24)."""
    return encode_command("xn")


def cmd_save() -> bytes:
    """Build the ``$save`` command - persist current config to flash (§10.3.23)."""
    return encode_command("save")


def cmd_set_sample_rate(code: int) -> bytes:
    """Build the ``$sm`` command - set sample rate (§10.4.11).

    Allowed values match the Service-Mode ``m`` command: 0=125 Hz,
    1=250 Hz, 2=500 Hz, 3=1000 Hz, 4=2000 Hz, 5=ExtTrig.
    """
    if not 0 <= code <= 5:
        raise ValueError("sample rate code must be 0..5, got {0}".format(code))
    return encode_command("sm", code)


def cmd_set_datagram(datagram_id: int, *, crlf: bool = False) -> bytes:
    """Build the ``$sd`` command - set Normal-Mode datagram format (§10.4.4).

    ``datagram_id`` is a Table 5-21 ID (rendered as the hex form the device
    expects). ``crlf`` toggles CR+LF termination.
    """
    return encode_command("sd", "{0:02x}".format(datagram_id), "1" if crlf else "0")
