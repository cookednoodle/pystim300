"""Service-Mode protocol (pure encode/decode; I/O lives on the client).

Service Mode is a human-readable ASCII protocol intended for
terminal-based interaction or scripted reconfiguration. Each command
is a short lowercase string terminated with CR (0x0D); the device
echoes nothing for the command itself, then emits the multi-line
response and finally a ``\\r>`` prompt to signal "ready for next
command". Errors come back as ``E<nnn> <description>\\r``.

References:
  - §7.5.3 Service Mode entry/exit (p.55)
  - §8 Normal-Mode commands including ``SERVICEMODE`` (pp.58-59)
  - §9 Service-Mode command catalogue (pp.62-90)
  - Figure 8-1 example SERVICEMODE entry banner (p.61)

This module is I/O-free. ``ServiceModeProtocol`` (in ``client.py``)
owns the transport, manages the read loop, and fires audit events for
inspectable Service-Mode I/O.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from pystim300.exceptions import CommandError, ProtocolError

SERVICE_PROMPT = b"\r>"
SERVICE_TERMINATOR = b"\r"
SERVICEMODE_ENTRY = b"SERVICEMODE\r"

# Tail of the "SYSTEM RETURNING TO <mode> MODE." confirmation line that the
# x (EXIT) command emits instead of the usual "\r>" prompt (Figures 9-49 /
# 9-50, p.87).
SERVICE_EXIT_MARKER = b"MODE."


@dataclass(frozen=True)
class ServiceResponse:
    """Decoded Service-Mode response.

    Attributes:
        command: The command string sent to the device (e.g. ``"i d"`` or
            ``"m 4"``), without trailing CR.
        lines: Response lines split on CR (LF is not used by Service Mode).
            The trailing prompt ``">"`` is **not** included.
        raw: The exact bytes received from the device, including the final
            ``\\r>`` prompt - for auditing and inspection.
    """

    command: str
    lines: Tuple[str, ...]
    raw: bytes


def encode_command(command: str) -> bytes:
    """Build the wire bytes for a Service-Mode command.

    ``command`` is the human-readable form (e.g. ``"i d"`` or ``"m 4"``);
    a trailing CR is appended per §9 (pp.62-90). The empty string sends
    just a CR, which the device uses to redisplay the prompt.
    """
    if "\r" in command or "\n" in command:
        raise ValueError("Service-Mode command must not contain CR/LF: {0!r}".format(command))
    return command.encode("ascii") + SERVICE_TERMINATOR


def find_response_end(buffer: bytes) -> int:
    """Return the index of the first byte AFTER a complete response.

    A complete response ends with the ``\\r>`` prompt (§7.5.3). Returns
    -1 if the prompt has not yet arrived. Used by the I/O loop to know
    when to stop reading.
    """
    idx = buffer.find(SERVICE_PROMPT)
    if idx < 0:
        return -1
    return idx + len(SERVICE_PROMPT)


def parse_response(raw: bytes, command: str) -> ServiceResponse:
    """Parse the full received bytes (including final ``\\r>``) into a response.

    Raises ``ProtocolError`` if ``raw`` does not end with the prompt.
    """
    if not raw.endswith(SERVICE_PROMPT):
        raise ProtocolError("Service-Mode response missing trailing prompt: {0!r}".format(raw))
    # Strip the trailing "\r>" prompt; split remaining on CR.
    body = raw[:-len(SERVICE_PROMPT)]
    # Body may be empty (just a prompt re-display); split keeps a single
    # empty string in that case which we trim.
    text = body.decode("ascii", errors="replace")
    parts = text.split("\r")
    # Drop a single leading empty (sometimes the device emits an extra CR).
    if parts and parts[0] == "":
        parts = parts[1:]
    # Drop a trailing empty (CR immediately before the prompt).
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return ServiceResponse(command=command, lines=tuple(parts), raw=bytes(raw))


def find_exit_response_end(buffer: bytes) -> int:
    """Return the index of the first byte AFTER a complete x (EXIT) response.

    The EXIT command is special: it does **not** end with the ``\\r>``
    prompt, because the device leaves Service Mode. Two outcomes:

    * **Success** - the device emits ``SYSTEM RETURNING TO <mode> MODE.``
      followed by ``\\r`` and then resumes Normal- (or Init-) Mode
      traffic (Figures 9-49 / 9-50, p.87). Located via the ``MODE.``
      marker and the next ``\\r``.
    * **Rejected parameter** - the device stays in Service Mode, emits an
      ``E<nnn>`` error line and re-issues the usual ``\\r>`` prompt.

    Returns the index just past whichever terminator appears first, or
    -1 if neither has arrived yet. The success marker is checked first:
    the error response contains no ``MODE.``, and on success the marker
    matches inside the confirmation line before any binary datagrams
    (which could contain stray ``\\r>`` bytes) arrive.
    """
    marker = buffer.find(SERVICE_EXIT_MARKER)
    if marker >= 0:
        cr = buffer.find(b"\r", marker)
        if cr >= 0:
            return cr + 1
    prompt = buffer.find(SERVICE_PROMPT)
    if prompt >= 0:
        return prompt + len(SERVICE_PROMPT)
    return -1


def parse_exit_response(raw: bytes, command: str) -> ServiceResponse:
    """Parse an x (EXIT) command response into a ``ServiceResponse``.

    Unlike :func:`parse_response` this does not require a trailing
    ``\\r>`` prompt - the EXIT response has none. ``raw`` is split on CR
    and non-empty lines are kept; a rejected-parameter ``E<nnn>`` line is
    preserved so :func:`detect_error` / :func:`raise_for_error` can
    surface it.
    """
    text = raw.decode("ascii", errors="replace")
    lines = tuple(part for part in text.split("\r") if part.strip())
    return ServiceResponse(command=command, lines=lines, raw=bytes(raw))


def detect_error(response: ServiceResponse) -> Optional[Tuple[int, str]]:
    """Return ``(code, message)`` if ``response`` is an ``E<nnn>`` error, else None.

    Service Mode reports errors as ``E001`` through ``E007`` plus a textual
    description on the same line (§9.0). Codes are decimal, three digits.
    """
    for line in response.lines:
        stripped = line.strip()
        if (len(stripped) >= 4 and stripped[0] == "E"
                and stripped[1:4].isdigit()):
            code = int(stripped[1:4])
            message = stripped[4:].strip()
            return code, message
    return None


def raise_for_error(response: ServiceResponse) -> ServiceResponse:
    """Raise ``CommandError`` if ``response`` carries an error code; else return it."""
    err = detect_error(response)
    if err is not None:
        code, message = err
        raise CommandError(
            "Service-Mode E{0:03d}: {1}".format(code, message),
            code=code,
            command=response.command,
            raw=response.raw,
        )
    return response


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def cmd_information(sub: Optional[str] = None) -> str:
    """Build an ``i`` command, optionally with a sub-command letter.

    Sub-commands per §9.1 (p.62): ``a`` (acc), ``b`` (bias trim), ``d``
    (datagram), ``e`` (extended error), ``f`` (filter), ``g`` (gyro),
    ``h`` (system config), ``m`` (sample rate), ``n`` (part number),
    ``p`` (parity?), ``r`` (line term), ``s`` (save status), ``t``
    (transmission), ``u`` (output unit), ``x`` (exit).
    """
    if sub is None:
        return "i"
    if not (len(sub) == 1 and sub.isalpha()):
        raise ValueError("Service-Mode 'i' sub-command must be a single letter, got {0!r}".format(sub))
    return "i {0}".format(sub)


def cmd_single_shot() -> str:
    """Build the ``a`` command - single-shot measurement (§9 ``a``, p.~63)."""
    return "a"


def cmd_save() -> str:
    """Build the ``s`` command - save current settings to flash (§9 ``s``)."""
    return "s"


def cmd_restore_factory() -> str:
    """Build the ``z`` command - restore factory settings (§9 ``z``)."""
    return "z"


def cmd_exit(*, to_normal: bool = True) -> str:
    """Build the ``x`` command - exit Service Mode.

    Per §9.14 / Table 9-54 (p.87) the ``<exit_to>`` parameter is a
    letter, not a digit. The upper-case forms are used here because they
    exit immediately - no ``CONFIRM EXIT(Y/N)`` prompt even when there
    are unsaved changes, and no 3 s hold-time - which is the
    deterministic behaviour a programmatic caller wants:

      * ``x N`` - terminate and return immediately to Normal Mode
      * ``x I`` - terminate and return immediately to Init Mode

    The lower-case ``n`` / ``i`` variants (interactive confirmation plus
    hold-time) are intentionally not exposed.
    """
    return "x N" if to_normal else "x I"


def cmd_set_sample_rate(value: int) -> str:
    """Build the ``m`` command - set sample rate.

    Allowed values per §9 (sample rate sub-section):
      0 = 125 Hz, 1 = 250 Hz, 2 = 500 Hz, 3 = 1000 Hz, 4 = 2000 Hz,
      5 = External Trigger.
    """
    if not 0 <= value <= 5:
        raise ValueError("sample rate code must be 0..5, got {0}".format(value))
    return "m {0}".format(value)


def cmd_set_datagram_format(datagram_id: int, *, crlf: bool = False) -> str:
    """Build the ``d`` command - set Normal-Mode datagram format.

    Per §9 ``d``, the parameter selects one of the 16 datagram IDs in
    Table 5-21. The second parameter toggles CR+LF termination
    (``"yes"``/``"no"`` in the device's text; this helper formats it).
    """
    return "d {0:02x},{1}".format(datagram_id, "yes" if crlf else "no")


def cmd_set_line_termination(enabled: bool) -> str:
    """Build the ``r`` command - line termination on/off (§9 ``r``)."""
    return "r 1" if enabled else "r 0"


def cmd_set_transmission(*, bit_rate: int = 0, stop_bits: int = 1,
                          parity: str = "none") -> str:
    """Build the ``t`` command - transmission parameters.

    Per §9 ``t``: ``t <bit_rate>,<stop_bits>,<parity>``. ``bit_rate`` of
    ``0`` selects the device's currently-active bit rate (no change);
    pass an explicit value to set one of 374400 / 460800 / 921600 /
    1843200 or user-defined.
    """
    parity_map = {"none": "n", "even": "e", "odd": "o"}
    if parity not in parity_map:
        raise ValueError("parity must be 'none'/'even'/'odd', got {0!r}".format(parity))
    return "t {0},{1},{2}".format(bit_rate, stop_bits, parity_map[parity])
