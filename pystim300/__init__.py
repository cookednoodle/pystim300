"""pystim300 - Python driver for the Safran STIM300 IMU over RS422.

Public API is re-exported here. Implementation lives in sub-modules; nothing
in the codec layer imports pyserial, so the codec can be exercised without
the optional ``[serial]`` extra installed.

See the datasheet ``ts1524-r31-datasheet-stim300.pdf`` for protocol details.
"""

from pystim300.client import AuditEvent, MemoryAuditor, Mode, STIM300
from pystim300.configuration import Configuration
from pystim300.datagrams import (
    BiasTrimDatagram,
    ExtendedErrorDatagram,
    PartNumberDatagram,
    SerialNumberDatagram,
)
from pystim300.exceptions import (
    CommandError,
    CrcError,
    ModeError,
    ProtocolError,
    Stim300Error,
    TimeoutError,
)
from pystim300.normal import Measurement, NormalStreamParser
from pystim300.service import ServiceResponse
from pystim300.status import StatusByte
from pystim300.transport import FakeTransport, SerialTransport, Transport
from pystim300.utility import UtilityResponse

__version__ = "0.1.0"

__all__ = [
    "AuditEvent",
    "BiasTrimDatagram",
    "CommandError",
    "Configuration",
    "CrcError",
    "ExtendedErrorDatagram",
    "FakeTransport",
    "Measurement",
    "MemoryAuditor",
    "Mode",
    "ModeError",
    "NormalStreamParser",
    "PartNumberDatagram",
    "ProtocolError",
    "STIM300",
    "SerialNumberDatagram",
    "SerialTransport",
    "ServiceResponse",
    "Stim300Error",
    "StatusByte",
    "TimeoutError",
    "Transport",
    "UtilityResponse",
    "__version__",
]
