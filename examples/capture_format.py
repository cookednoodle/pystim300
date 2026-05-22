"""Binary capture-file format for raw STIM300 serial captures.

Stage 1 (``capture_raw.py``) writes a ``.cap`` file; stage 2
(``decode_capture.py``) reads it back. The file is a small fixed header
followed by a sequence of length-prefixed, timestamped records - one
record per transport read.

Layout (little-endian)::

    header:  magic[10] version:u16 epoch0:f64 perf0:f64 bit_rate:u32
             port_len:u16 port[port_len]
    record:  perf:f64 length:u32 payload[length]            (repeated)

``epoch0`` / ``perf0`` are ``time.time()`` and ``time.perf_counter()``
sampled back-to-back once at capture start; every record's ``perf`` is a
``time.perf_counter()`` reading. Absolute time is therefore
``epoch0 + (perf - perf0)`` and elapsed (drift-free) time is
``perf - perf0``.
"""

import struct
from dataclasses import dataclass
from typing import BinaryIO, Iterator, Tuple

MAGIC = b"STIM300CAP"
VERSION = 1

# magic, version, epoch0, perf0, bit_rate, port_len
_HEADER_FIXED = struct.Struct("<10sHddIH")
# perf, payload length
_RECORD_HEAD = struct.Struct("<dI")


@dataclass(frozen=True)
class CaptureHeader:
    """Parsed capture-file header (see module docstring for the layout)."""

    version: int
    epoch0: float
    perf0: float
    bit_rate: int
    port: str


def write_header(f: BinaryIO, *, epoch0: float, perf0: float,
                 bit_rate: int, port: str) -> None:
    """Write the capture-file header to ``f`` (must be the first write)."""
    port_bytes = port.encode("utf-8")
    f.write(_HEADER_FIXED.pack(MAGIC, VERSION, epoch0, perf0,
                               bit_rate, len(port_bytes)))
    f.write(port_bytes)


def write_record(f: BinaryIO, perf: float, payload: bytes) -> None:
    """Append one timestamped record to ``f``."""
    f.write(_RECORD_HEAD.pack(perf, len(payload)))
    f.write(payload)


def read_header(f: BinaryIO) -> CaptureHeader:
    """Read and validate the capture-file header from ``f``."""
    fixed = f.read(_HEADER_FIXED.size)
    if len(fixed) != _HEADER_FIXED.size:
        raise ValueError("capture file too short for a header")
    magic, version, epoch0, perf0, bit_rate, port_len = _HEADER_FIXED.unpack(fixed)
    if magic != MAGIC:
        raise ValueError("not a STIM300 capture file (bad magic {0!r})".format(magic))
    port = f.read(port_len).decode("utf-8", errors="replace")
    return CaptureHeader(version=version, epoch0=epoch0, perf0=perf0,
                         bit_rate=bit_rate, port=port)


def iter_records(f: BinaryIO) -> Iterator[Tuple[float, bytes]]:
    """Yield ``(perf, payload)`` for each record until end of file.

    A truncated trailing record - the writer was killed mid-write - is
    not an error: the iterator stops cleanly at the first short read.
    """
    while True:
        head = f.read(_RECORD_HEAD.size)
        if len(head) < _RECORD_HEAD.size:
            return
        perf, length = _RECORD_HEAD.unpack(head)
        payload = f.read(length)
        if len(payload) < length:
            return
        yield perf, payload
