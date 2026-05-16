"""CRC implementations for STIM300.

Two CRCs are used by the device, each documented in its own datasheet section:

* **CRC-32** for Normal-Mode and Init-Mode binary datagrams.
  Polynomial 0x04C11DB7, seed 0xFFFFFFFF, MSB-first, no input/output
  reflection, no final XOR. See §5.5.7 (p.37). Note: ``binascii.crc32``
  is the reflected Ethernet variant and is **not** compatible.

* **CRC-8** for Utility-Mode ASCII commands and responses.
  Polynomial 0x07, seed 0xFF, MSB-first, no reflection, no final XOR.
  Computed over the ASCII payload up to (and excluding) the final comma
  before the CRC field; rendered/parsed as ASCII decimal (e.g.
  ``$isn,28\\r``). See §10.2.3.

Both functions build a 256-entry lookup table once at import time.
"""

from typing import Tuple

CRC32_POLY = 0x04C11DB7  # §5.5.7, p.37
CRC32_SEED = 0xFFFFFFFF  # §5.5.7, p.37
CRC8_POLY = 0x07          # §10.2.3
CRC8_SEED = 0xFF          # §10.2.3


def _build_crc32_table(poly: int) -> Tuple[int, ...]:
    table = []
    for i in range(256):
        crc = i << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ poly) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
        table.append(crc)
    return tuple(table)


def _build_crc8_table(poly: int) -> Tuple[int, ...]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
        table.append(crc)
    return tuple(table)


_CRC32_TABLE = _build_crc32_table(CRC32_POLY)
_CRC8_TABLE = _build_crc8_table(CRC8_POLY)


def crc32_stim300(data: bytes) -> int:
    """Compute the STIM300 CRC-32 over ``data``.

    The caller is responsible for appending any required dummy 0x00 padding
    bytes to align ``data`` to a 4-byte boundary before calling (per
    Table 5-22, p.37); the padding is *not* present on the wire.
    """
    crc = CRC32_SEED
    table = _CRC32_TABLE
    for b in data:
        crc = ((crc << 8) ^ table[((crc >> 24) ^ b) & 0xFF]) & 0xFFFFFFFF
    return crc


def crc8_stim300(data: bytes) -> int:
    """Compute the STIM300 Utility-Mode CRC-8 over ``data``.

    ``data`` is the raw ASCII payload (e.g. ``b"$isn"``) up to and including
    the last byte before the CRC field. See §10.2.3.
    """
    crc = CRC8_SEED
    table = _CRC8_TABLE
    for b in data:
        crc = table[(crc ^ b) & 0xFF]
    return crc
