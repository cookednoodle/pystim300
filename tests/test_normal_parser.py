"""Tests for the Normal-Mode streaming parser (framer, resync, interleaving)."""

import pytest

from pystim300.configuration import CONFIGURATION_PAYLOAD_LENGTH
from pystim300.crc import crc32_stim300
from pystim300.datagrams import (
    BIAS_TRIM_IDS,
    BIAS_TRIM_PAYLOAD_LENGTH,
    BiasTrimDatagram,
    EXTENDED_ERROR_IDS,
    EXTENDED_ERROR_PAYLOAD_LENGTH,
    ExtendedErrorDatagram,
    PART_NUMBER_IDS,
    PART_NUMBER_PAYLOAD_LENGTH,
    PartNumberDatagram,
    SERIAL_NUMBER_IDS,
    SERIAL_NUMBER_PAYLOAD_LENGTH,
    SerialNumberDatagram,
)
from pystim300.normal import (
    NORMAL_SPECS,
    Measurement,
    NormalStreamParser,
    SPECIAL_SPECS,
    build_normal_frame,
)
from pystim300.configuration import Configuration

from conftest import make_configuration, make_measurement


def _make_special_frame(datagram_id: int, payload: bytes) -> bytes:
    """Build the wire bytes for an Init / on-demand special datagram."""
    spec = SPECIAL_SPECS[datagram_id]
    assert len(payload) == spec.payload_length
    id_and_payload = bytes([datagram_id]) + payload
    crc = crc32_stim300(id_and_payload + b"\x00" * spec.dummy_bytes)
    frame = id_and_payload + crc.to_bytes(4, "big")
    # Odd-numbered IDs get CR+LF termination (Tables 5-13, 5-15, 5-16, 5-17, 5-18).
    if datagram_id in {PART_NUMBER_IDS[1], SERIAL_NUMBER_IDS[1], 0xBD,
                       BIAS_TRIM_IDS[1], EXTENDED_ERROR_IDS[1]}:
        frame += b"\r\n"
    return frame


class TestSingleFrame:
    def test_simple_round_trip(self):
        cfg = make_configuration(has_acceleration=True)
        meas = make_measurement(cfg)
        wire = build_normal_frame(meas, cfg)

        parser = NormalStreamParser(cfg)
        records = list(parser.feed(wire))
        assert len(records) == 1
        assert isinstance(records[0], Measurement)
        assert records[0].counter == meas.counter
        assert parser.resync_events == 0

    def test_multiple_frames(self):
        cfg = make_configuration(has_acceleration=True)
        frames = b"".join(build_normal_frame(
            make_measurement(cfg, counter=i, latency_us=i * 100), cfg
        ) for i in range(10))

        parser = NormalStreamParser(cfg)
        records = list(parser.feed(frames))
        assert len(records) == 10
        assert [r.counter for r in records] == list(range(10))


class TestPartialReads:
    def test_single_byte_at_a_time(self):
        cfg = make_configuration(has_acceleration=True, crlf=True)
        meas = make_measurement(cfg, counter=7)
        wire = build_normal_frame(meas, cfg)

        parser = NormalStreamParser(cfg)
        records = []
        for byte in wire:
            records.extend(parser.feed(bytes([byte])))
        assert len(records) == 1
        assert records[0].counter == 7

    @pytest.mark.parametrize("split_at", [0, 1, 5, 10, 18, 27])
    def test_split_across_two_chunks(self, split_at):
        cfg = make_configuration(has_acceleration=True)
        meas = make_measurement(cfg, counter=99)
        wire = build_normal_frame(meas, cfg)
        if split_at >= len(wire):
            pytest.skip("split point past end of frame")

        parser = NormalStreamParser(cfg)
        first = list(parser.feed(wire[:split_at]))
        second = list(parser.feed(wire[split_at:]))
        records = first + second
        assert len(records) == 1
        assert records[0].counter == 99

    def test_three_frames_split_at_every_boundary(self):
        cfg = make_configuration(has_acceleration=True)
        frames = b"".join(build_normal_frame(
            make_measurement(cfg, counter=i), cfg
        ) for i in range(3))

        for split in range(0, len(frames)):
            parser = NormalStreamParser(cfg)
            recs = list(parser.feed(frames[:split])) + list(parser.feed(frames[split:]))
            assert len(recs) == 3, "split at {0}".format(split)
            assert [r.counter for r in recs] == [0, 1, 2]


class TestCrcResync:
    def test_corrupted_first_frame_drops_then_resyncs(self):
        cfg = make_configuration(has_acceleration=True)
        # First frame: corrupt the middle byte.
        good = build_normal_frame(make_measurement(cfg, counter=10), cfg)
        bad = bytearray(good)
        bad[len(bad) // 2] ^= 0xFF
        # Append several follow-on frames so the parser has enough look-ahead
        # to rule out any special-ID candidates that randomly match a byte in
        # the corrupted frame. Resync requires up to a max-frame-length of
        # buffer beyond a misleading head byte to determine it is garbage.
        good_run = b"".join(build_normal_frame(
            make_measurement(cfg, counter=11 + i), cfg) for i in range(3))

        parser = NormalStreamParser(cfg)
        records = list(parser.feed(bytes(bad) + good_run))
        # Corrupted frame is dropped; subsequent frames decode normally.
        assert [r.counter for r in records] == [11, 12, 13]
        assert parser.resync_events > 0
        assert parser.bytes_discarded > 0

    def test_garbage_prefix_skipped(self):
        cfg = make_configuration(has_acceleration=True)
        meas = make_measurement(cfg, counter=5)
        good = build_normal_frame(meas, cfg)
        garbage = b"\xaa\xbb\xcc\xdd\x12\x34\x56\x78"  # bytes != configured ID

        parser = NormalStreamParser(cfg)
        records = list(parser.feed(garbage + good))
        assert len(records) == 1
        assert records[0].counter == 5
        assert parser.bytes_discarded == len(garbage)


class TestSpecialDatagramInterleaving:
    def test_part_number_in_stream(self):
        cfg = make_configuration(has_acceleration=True)
        normal_frame = build_normal_frame(make_measurement(cfg, counter=1), cfg)

        # Part Number payload: hand-crafted bytes.
        pn_payload = bytes(PART_NUMBER_PAYLOAD_LENGTH)
        pn_frame = _make_special_frame(PART_NUMBER_IDS[0], pn_payload)

        stream = normal_frame + pn_frame + build_normal_frame(
            make_measurement(cfg, counter=2), cfg)

        parser = NormalStreamParser(cfg)
        records = list(parser.feed(stream))
        assert len(records) == 3
        assert isinstance(records[0], Measurement) and records[0].counter == 1
        assert isinstance(records[1], PartNumberDatagram)
        assert isinstance(records[2], Measurement) and records[2].counter == 2

    def test_serial_number_in_stream(self):
        cfg = make_configuration()
        sn_payload = bytearray(SERIAL_NUMBER_PAYLOAD_LENGTH)
        sn_payload[0] = ord("N")
        sn_frame = _make_special_frame(SERIAL_NUMBER_IDS[1], bytes(sn_payload))
        normal = build_normal_frame(make_measurement(cfg), cfg)

        parser = NormalStreamParser(cfg)
        records = list(parser.feed(sn_frame + normal))
        assert isinstance(records[0], SerialNumberDatagram)
        assert isinstance(records[1], Measurement)

    def test_bias_trim_in_stream(self):
        cfg = make_configuration()
        bt_frame = _make_special_frame(BIAS_TRIM_IDS[0], bytes(BIAS_TRIM_PAYLOAD_LENGTH))
        normal = build_normal_frame(make_measurement(cfg), cfg)
        parser = NormalStreamParser(cfg)
        records = list(parser.feed(bt_frame + normal))
        assert isinstance(records[0], BiasTrimDatagram)
        assert isinstance(records[1], Measurement)

    def test_extended_error_in_stream(self):
        cfg = make_configuration()
        ee_frame = _make_special_frame(EXTENDED_ERROR_IDS[1],
                                        bytes(EXTENDED_ERROR_PAYLOAD_LENGTH))
        normal = build_normal_frame(make_measurement(cfg), cfg)
        parser = NormalStreamParser(cfg)
        records = list(parser.feed(ee_frame + normal))
        assert isinstance(records[0], ExtendedErrorDatagram)
        assert isinstance(records[1], Measurement)

    def test_configuration_in_stream(self):
        cfg = make_configuration()
        cfg_payload = bytes(CONFIGURATION_PAYLOAD_LENGTH)  # all-zero defaults
        cfg_frame = _make_special_frame(0xBC, cfg_payload)
        normal = build_normal_frame(make_measurement(cfg), cfg)
        parser = NormalStreamParser(cfg)
        records = list(parser.feed(cfg_frame + normal))
        assert isinstance(records[0], Configuration)
        assert isinstance(records[1], Measurement)


class TestConfigurationChange:
    def test_swap_configuration_mid_stream(self):
        # Stream frames at ID 0x90 (rate only), then swap to 0x91 (rate + accel)
        # and continue.
        cfg_a = make_configuration()                          # ID 0x90
        cfg_b = make_configuration(has_acceleration=True)     # ID 0x91

        frame_a1 = build_normal_frame(make_measurement(cfg_a, counter=10), cfg_a)
        frame_a2 = build_normal_frame(make_measurement(cfg_a, counter=11), cfg_a)
        frame_b1 = build_normal_frame(make_measurement(cfg_b, counter=20), cfg_b)

        parser = NormalStreamParser(cfg_a)
        records = list(parser.feed(frame_a1 + frame_a2))
        assert [r.counter for r in records] == [10, 11]

        parser.update_configuration(cfg_b)
        records2 = list(parser.feed(frame_b1))
        assert [r.counter for r in records2] == [20]
        assert records2[0].datagram_id == 0x91


class TestCrlfHandling:
    def test_crlf_required_when_configured(self):
        cfg = make_configuration(has_acceleration=True, crlf=True)
        meas = make_measurement(cfg)
        wire = build_normal_frame(meas, cfg)
        assert wire[-2:] == b"\r\n"

        parser = NormalStreamParser(cfg)
        records = list(parser.feed(wire))
        assert len(records) == 1

    def test_missing_crlf_treated_as_corruption(self):
        cfg = make_configuration(has_acceleration=True, crlf=True)
        meas = make_measurement(cfg)
        wire = build_normal_frame(meas, cfg)
        # Strip the CR+LF then append several valid follow-on frames so the
        # parser has enough look-ahead to resync past the corrupted frame.
        good_run = b"".join(build_normal_frame(
            make_measurement(cfg, counter=meas.counter + 1 + i), cfg)
            for i in range(3))
        broken = wire[:-2] + good_run

        parser = NormalStreamParser(cfg)
        records = list(parser.feed(broken))
        # Without the CR+LF marker the first frame can't be accepted; the
        # parser resyncs and finds the next frames.
        recovered = [r.counter for r in records]
        assert meas.counter + 1 in recovered
        assert meas.counter + 2 in recovered
        assert meas.counter + 3 in recovered
