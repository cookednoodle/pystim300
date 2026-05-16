"""Tests for the CRC-32 and CRC-8 implementations.

Three flavours of test:

1. Invariants that must hold regardless of polynomial choice
   (determinism, empty input == seed, output range).
2. Cross-check against bit-banged reference implementations defined
   inline. The reference walks bit-by-bit through the same polynomial,
   seed, reflection and XOR settings as the production code, providing
   an independent computation path. If the table-driven implementation
   diverges from the reference, one of them has a bug.
3. Optional cross-check against ``crcmod`` (dev-only dependency).
   Skips gracefully when ``crcmod`` is not installed.
"""

import pytest

from pystim300.crc import (
    CRC8_POLY,
    CRC8_SEED,
    CRC32_POLY,
    CRC32_SEED,
    crc8_stim300,
    crc32_stim300,
)


def _bitbang_crc32(data: bytes) -> int:
    """Reference CRC-32: bit-by-bit, MSB-first, no reflection, no final XOR."""
    crc = CRC32_SEED
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ CRC32_POLY) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc


def _bitbang_crc8(data: bytes) -> int:
    """Reference CRC-8: bit-by-bit, MSB-first, no reflection, no final XOR."""
    crc = CRC8_SEED
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


class TestCrc32Invariants:
    def test_empty_input_returns_seed(self):
        assert crc32_stim300(b"") == CRC32_SEED

    def test_deterministic(self):
        msg = b"\x90\x01\x02\x03\x04\x05\x06\x07"
        assert crc32_stim300(msg) == crc32_stim300(msg)

    def test_output_is_32_bit(self):
        for msg in [b"", b"\x00", b"\xff" * 32, b"some bytes" * 10]:
            value = crc32_stim300(msg)
            assert 0 <= value <= 0xFFFFFFFF

    def test_single_byte_changes_output(self):
        # Sanity: changing any byte changes the CRC for random-ish input.
        base = b"\x90\xaa\xbb\xcc\xdd"
        baseline = crc32_stim300(base)
        for i in range(len(base)):
            mutated = bytearray(base)
            mutated[i] ^= 0x01
            assert crc32_stim300(bytes(mutated)) != baseline


class TestCrc32BitbangReference:
    """Cross-check the table-driven CRC-32 against a bit-banged reference."""

    @pytest.mark.parametrize(
        "msg",
        [
            b"",
            b"\x00",
            b"\xff",
            b"\x90",
            b"\x90\x00\x00\x00",
            bytes(range(16)),
            bytes(range(64)),
            b"\xde\xad\xbe\xef",
            b"STIM300" * 5,
            bytes([i ^ 0x5a for i in range(256)]),
        ],
    )
    def test_matches_bitbang(self, msg):
        assert crc32_stim300(msg) == _bitbang_crc32(msg)


class TestCrc8BitbangReference:
    @pytest.mark.parametrize(
        "msg",
        [
            b"",
            b"\x00",
            b"\xff",
            b"$isn",
            b"$iconf",
            b"#UTILITYMODE",
            b"$xn",
            bytes(range(32)),
            bytes(range(256)),
        ],
    )
    def test_matches_bitbang(self, msg):
        assert crc8_stim300(msg) == _bitbang_crc8(msg)


class TestCrc32Crcmod:
    """Cross-check against an independent CRC-32 implementation.

    ``crcmod`` is a dev-only dependency listed in ``[dev]`` extras. When
    not installed the tests are skipped rather than failing, so the
    library can be tested on a minimal install.
    """

    @pytest.fixture
    def reference(self):
        crcmod = pytest.importorskip("crcmod")
        # poly is supplied to crcmod with the explicit leading bit:
        # x^32 + 0x04C11DB7 -> 0x104C11DB7
        return crcmod.mkCrcFun(0x104C11DB7, initCrc=CRC32_SEED, rev=False, xorOut=0x00000000)

    @pytest.mark.parametrize(
        "msg",
        [
            b"",
            b"\x00",
            b"\xff",
            b"\x90",
            bytes(range(16)),
            bytes(range(64)),
            b"\xde\xad\xbe\xef",
            b"STIM300" * 5,
        ],
    )
    def test_matches_crcmod(self, reference, msg):
        assert crc32_stim300(msg) == reference(msg)


class TestCrc8Invariants:
    def test_empty_input_returns_seed(self):
        assert crc8_stim300(b"") == CRC8_SEED

    def test_deterministic(self):
        msg = b"$isn"
        assert crc8_stim300(msg) == crc8_stim300(msg)

    def test_output_is_8_bit(self):
        for msg in [b"", b"\x00", b"\xff" * 32, b"$utility"]:
            value = crc8_stim300(msg)
            assert 0 <= value <= 0xFF

    def test_single_byte_changes_output(self):
        base = b"$abcd"
        baseline = crc8_stim300(base)
        for i in range(len(base)):
            mutated = bytearray(base)
            mutated[i] ^= 0x01
            assert crc8_stim300(bytes(mutated)) != baseline


class TestCrc8Crcmod:
    @pytest.fixture
    def reference(self):
        crcmod = pytest.importorskip("crcmod")
        return crcmod.mkCrcFun(0x107, initCrc=CRC8_SEED, rev=False, xorOut=0x00)

    @pytest.mark.parametrize(
        "msg",
        [
            b"",
            b"\x00",
            b"\xff",
            b"$isn",
            b"$iconf",
            b"#UTILITYMODE",
            b"$xn",
            bytes(range(32)),
        ],
    )
    def test_matches_crcmod(self, reference, msg):
        assert crc8_stim300(msg) == reference(msg)


class TestCrcConstants:
    def test_crc32_constants(self):
        assert CRC32_POLY == 0x04C11DB7
        assert CRC32_SEED == 0xFFFFFFFF

    def test_crc8_constants(self):
        assert CRC8_POLY == 0x07
        assert CRC8_SEED == 0xFF
