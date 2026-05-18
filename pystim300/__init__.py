"""pystim300 - Python driver for the Safran STIM300 IMU over RS422.

Public API is re-exported here. Implementation lives in sub-modules; nothing
in the codec layer imports pyserial, so the codec can be exercised without
the optional ``[serial]`` extra installed.

See the datasheet ``ts1524-r31-datasheet-stim300.pdf`` for protocol details.
"""

from pystim300.checkout import (
    CheckResult,
    CheckoutReport,
    ExpectedConfiguration,
    check_bias_trim_present,
    check_configuration,
    check_extended_error_clean,
    check_frame_rate,
    check_gravity_direction_consistent,
    check_gravity_magnitude,
    check_gyro_quiescent,
    check_inclinometer_gravity,
    check_latency_within,
    check_no_dropped_frames,
    check_parser_clean,
    check_part_number,
    check_serial_number,
    check_service_round_trip,
    check_status_bytes_clean,
    check_temperature_range,
    check_utility_round_trip,
)
from pystim300.client import (
    AuditEvent,
    InitSequence,
    MemoryAuditor,
    Mode,
    STIM300,
)
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
    "CheckResult",
    "CheckoutReport",
    "CommandError",
    "Configuration",
    "CrcError",
    "ExpectedConfiguration",
    "ExtendedErrorDatagram",
    "FakeTransport",
    "InitSequence",
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
    "check_bias_trim_present",
    "check_configuration",
    "check_extended_error_clean",
    "check_frame_rate",
    "check_gravity_direction_consistent",
    "check_gravity_magnitude",
    "check_gyro_quiescent",
    "check_inclinometer_gravity",
    "check_latency_within",
    "check_no_dropped_frames",
    "check_parser_clean",
    "check_part_number",
    "check_serial_number",
    "check_service_round_trip",
    "check_status_bytes_clean",
    "check_temperature_range",
    "check_utility_round_trip",
]
