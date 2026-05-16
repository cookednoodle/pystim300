"""Exception hierarchy for pystim300.

All errors raised by the library derive from ``Stim300Error`` so a caller
can catch the whole family with a single ``except`` clause.
"""

from typing import Optional


class Stim300Error(Exception):
    """Base class for every error raised by pystim300."""


class CrcError(Stim300Error):
    """A datagram or Utility-Mode response failed CRC validation.

    See §5.5.7 (CRC-32, p.37) and §10.2.3 (CRC-8) of the datasheet.
    """

    def __init__(self, message: str, *, expected: Optional[int] = None, actual: Optional[int] = None) -> None:
        super().__init__(message)
        self.expected = expected
        self.actual = actual


class ProtocolError(Stim300Error):
    """The byte stream did not conform to the expected protocol shape.

    Raised for malformed framing, unknown datagram IDs, truncated responses,
    or Service-Mode responses missing the ``\\r>`` terminator.
    """


class ModeError(Stim300Error):
    """An operation was attempted in the wrong device mode.

    For example, calling a Service-Mode command while the client believes
    the device is still in Normal Mode. See §7.5 of the datasheet for the
    mode state machine.
    """


class TimeoutError(Stim300Error):  # noqa: A001 - shadows builtin intentionally
    """A read on the underlying transport returned no bytes within the deadline."""


class CommandError(Stim300Error):
    """A Service- or Utility-Mode command returned an error response.

    Service Mode emits ``E<nnn> <description>\\r`` (§9, error codes
    E001-E007). Utility Mode emits ``#<cmd>,<status>,...`` with non-zero
    status (Table 10-2, §10.2.4).
    """

    def __init__(self, message: str, *, code: int, command: str, raw: bytes = b"") -> None:
        super().__init__(message)
        self.code = code
        self.command = command
        self.raw = raw
