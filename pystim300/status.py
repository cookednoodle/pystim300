"""STIM300 status byte decoder.

A status byte accompanies every measurement cluster (gyro, accelerometer,
inclinometer, and each temperature triple if present) in a Normal-Mode
datagram. The byte is **not latched**: each transmission reflects the
condition of *that* data only (§7.6).

Bit layout per Table 5-23 (p.38):

    Bit 7  System integrity error
    Bit 6  Start-up (set during ~0.5s after Init -> Normal transition)
    Bit 5  Outside operating conditions
    Bit 4  Overload          (bits 0-2 qualify which axis overflowed)
    Bit 3  Channel error     (bits 0-2 qualify which axis errored)
    Bit 2  Z-channel flag
    Bit 1  Y-channel flag
    Bit 0  X-channel flag
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class StatusByte:
    """Decoded STATUS byte from a Normal-Mode measurement cluster.

    See Table 5-23 (p.38) for the bit layout, and §7.6 for the
    self-diagnostics that drive these flags.
    """

    raw: int
    system_integrity_error: bool   # bit 7
    startup: bool                  # bit 6
    outside_operating_conditions: bool  # bit 5
    overload: bool                 # bit 4
    channel_error: bool            # bit 3
    axis_x: bool                   # bit 0
    axis_y: bool                   # bit 1
    axis_z: bool                   # bit 2

    @classmethod
    def decode(cls, byte: int) -> "StatusByte":
        if not 0 <= byte <= 0xFF:
            raise ValueError("status byte must be 0..255, got {0}".format(byte))
        return cls(
            raw=byte,
            system_integrity_error=bool(byte & 0x80),       # Table 5-23, bit 7
            startup=bool(byte & 0x40),                       # Table 5-23, bit 6
            outside_operating_conditions=bool(byte & 0x20),  # Table 5-23, bit 5
            overload=bool(byte & 0x10),                      # Table 5-23, bit 4
            channel_error=bool(byte & 0x08),                 # Table 5-23, bit 3
            axis_z=bool(byte & 0x04),                        # Table 5-23, bit 2
            axis_y=bool(byte & 0x02),                        # Table 5-23, bit 1
            axis_x=bool(byte & 0x01),                        # Table 5-23, bit 0
        )

    def ok(self) -> bool:
        """True iff no error or warning bits are set."""
        return self.raw == 0

    def axes(self) -> Tuple[bool, bool, bool]:
        """(x, y, z) channel flags. Meaningful when ``overload`` or ``channel_error`` is set."""
        return (self.axis_x, self.axis_y, self.axis_z)
