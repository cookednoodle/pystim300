"""End-to-end tests for the STIM300 client against a FakeTransport."""

import pytest

from pystim300.client import (
    AuditEvent,
    MemoryAuditor,
    Mode,
    STIM300,
)
from pystim300.crc import crc8_stim300
from pystim300.exceptions import CommandError, ModeError
from pystim300.normal import Measurement, build_normal_frame
from pystim300.service import SERVICE_PROMPT
from pystim300.transport import FakeTransport

from conftest import make_configuration, make_measurement


def _crc8(s: str) -> int:
    return crc8_stim300(s.encode("ascii"))


def _utility_response(body_no_crc: str) -> bytes:
    """Build wire bytes for a Utility-Mode response with trailing CR."""
    crc = _crc8(body_no_crc)
    return (body_no_crc + str(crc) + "\r").encode("ascii")


class TestMemoryAuditor:
    def test_collects_events(self):
        auditor = MemoryAuditor()
        auditor(AuditEvent(Mode.SERVICE, "tx", b"a\r", 0.0))
        auditor(AuditEvent(Mode.SERVICE, "rx", b"\r>", 0.1))
        assert len(auditor) == 2
        assert auditor.events[0].direction == "tx"
        assert auditor.events[1].payload == b"\r>"

    def test_bounded(self):
        auditor = MemoryAuditor(max_events=2)
        for i in range(5):
            auditor(AuditEvent(Mode.SERVICE, "tx", str(i).encode(), float(i)))
        assert len(auditor) == 2
        assert [e.payload for e in auditor.events] == [b"3", b"4"]

    def test_clear(self):
        auditor = MemoryAuditor()
        auditor(AuditEvent(Mode.SERVICE, "tx", b"x", 0.0))
        auditor.clear()
        assert len(auditor) == 0


class TestServiceFlow:
    def test_enter_service_emits_banner(self):
        banner = b"PRODUCT = STIM300\rREV = -\r>"
        transport = FakeTransport(initial=banner, scripted=[
            (b"SERVICEMODE\r", b""),  # transport ignores reply since initial already has banner
        ])
        # The scripted entry just acks the write; banner is pre-loaded.
        client = STIM300(transport)
        # We're using a fake clock helper to avoid sleeping.
        response = client.enter_service()
        assert client.mode == Mode.SERVICE
        assert "PRODUCT = STIM300" in response.lines
        assert "REV = -" in response.lines
        assert response.raw == banner

    def test_service_command_round_trip_with_audit(self):
        # Pre-load entry banner + response to "i m".
        banner = b"OK\r>"
        info_response = b"sample rate = 2000\r>"
        transport = FakeTransport(initial=banner + info_response)
        auditor = MemoryAuditor()
        client = STIM300(transport, audit=auditor)
        client.enter_service()
        resp = client.service_command("i m")
        assert resp.lines == ("sample rate = 2000",)
        # Audit covers SERVICEMODE tx + banner rx + i m tx + response rx.
        tx_events = [e for e in auditor.events if e.direction == "tx"]
        rx_events = [e for e in auditor.events if e.direction == "rx"]
        assert b"SERVICEMODE\r" in [e.payload for e in tx_events]
        assert b"i m\r" in [e.payload for e in tx_events]
        assert banner in [e.payload for e in rx_events]
        assert info_response in [e.payload for e in rx_events]

    def test_service_command_error_raises(self):
        banner = b"\r>"
        err_response = b"E001 syntax error\r>"
        transport = FakeTransport(initial=banner + err_response)
        client = STIM300(transport)
        client.enter_service()
        with pytest.raises(CommandError) as info:
            client.service_command("bogus")
        assert info.value.code == 1
        assert info.value.command == "bogus"

    def test_exit_service(self):
        banner = b"\r>"
        exit_response = b"\r>"
        transport = FakeTransport(initial=banner + exit_response)
        client = STIM300(transport)
        client.enter_service()
        client.exit_service()
        assert client.mode == Mode.NORMAL
        assert b"x 1\r" in bytes(transport.written)

    def test_service_command_wrong_mode_raises(self):
        client = STIM300(FakeTransport())
        with pytest.raises(ModeError):
            client.service_command("a")


class TestUtilityFlow:
    def test_enter_utility(self):
        ack = _utility_response("#UTILITYMODE,")
        transport = FakeTransport(initial=ack)
        client = STIM300(transport)
        resp = client.enter_utility()
        assert client.mode == Mode.UTILITY
        assert resp.command == "UTILITYMODE"
        assert resp.status == 0
        assert b"UTILITYMODE\r" in bytes(transport.written)

    def test_utility_command_round_trip(self):
        ack = _utility_response("#UTILITYMODE,")
        isn_response = _utility_response("#isn,0,N123456789ABCDE,")
        transport = FakeTransport(initial=ack + isn_response)
        auditor = MemoryAuditor()
        client = STIM300(transport, audit=auditor)
        client.enter_utility()
        resp = client.utility_command("isn")
        assert resp.command == "isn"
        assert resp.fields == ("N123456789ABCDE",)
        assert resp.status == 0
        # Audit captures both directions for both round-trips.
        directions = [e.direction for e in auditor.events]
        assert directions == ["tx", "rx", "tx", "rx"]

    def test_utility_command_error(self):
        ack = _utility_response("#UTILITYMODE,")
        error = _utility_response("#sm,5,")  # status 5 = invalid parameter
        transport = FakeTransport(initial=ack + error)
        client = STIM300(transport)
        client.enter_utility()
        with pytest.raises(CommandError) as info:
            client.utility_command("sm", 99)
        assert info.value.code == 5
        assert info.value.command == "sm"

    def test_exit_utility(self):
        ack = _utility_response("#UTILITYMODE,")
        xn_response = _utility_response("#xn,0,")
        transport = FakeTransport(initial=ack + xn_response)
        client = STIM300(transport)
        client.enter_utility()
        client.exit_utility()
        assert client.mode == Mode.NORMAL


class TestNormalFlow:
    def test_read_measurements(self):
        cfg = make_configuration(has_acceleration=True)
        frames = b"".join(build_normal_frame(
            make_measurement(cfg, counter=i), cfg
        ) for i in range(3))
        transport = FakeTransport(initial=frames)
        client = STIM300(transport, configuration=cfg)
        assert client.mode == Mode.NORMAL

        records = list(client.read_measurements(limit=3))
        assert len(records) == 3
        assert all(isinstance(r, Measurement) for r in records)
        assert [r.counter for r in records] == [0, 1, 2]

    def test_read_records_yields_specials(self):
        cfg = make_configuration()
        normal_frame = build_normal_frame(make_measurement(cfg, counter=5), cfg)
        # Inject a Configuration datagram before the normal frame.
        from pystim300.crc import crc32_stim300
        cfg_payload = bytes(21)
        cfg_frame_id = bytes([0xBC])
        crc = crc32_stim300(cfg_frame_id + cfg_payload + b"\x00\x00")
        cfg_frame = cfg_frame_id + cfg_payload + crc.to_bytes(4, "big")

        transport = FakeTransport(initial=cfg_frame + normal_frame)
        client = STIM300(transport, configuration=cfg)
        from pystim300.configuration import Configuration as Cfg
        records = list(client.read_records(limit=2))
        assert isinstance(records[0], Cfg)
        assert isinstance(records[1], Measurement)
        assert records[1].counter == 5

    def test_read_measurements_filter_startup(self):
        cfg = make_configuration()
        startup_frame = build_normal_frame(
            make_measurement(cfg, counter=0, status_raw=0x40), cfg)  # startup bit
        valid_frame = build_normal_frame(
            make_measurement(cfg, counter=1, status_raw=0x00), cfg)
        transport = FakeTransport(initial=startup_frame + valid_frame)
        client = STIM300(transport, configuration=cfg)
        records = list(client.read_measurements(limit=1, include_startup=False))
        assert len(records) == 1
        assert records[0].counter == 1

    def test_read_records_without_configuration_raises(self):
        client = STIM300(FakeTransport())
        with pytest.raises(ModeError):
            list(client.read_records(limit=1))


class TestReset:
    def test_reset_moves_to_init(self):
        cfg = make_configuration()
        transport = FakeTransport()
        client = STIM300(transport, configuration=cfg)
        client.reset()
        assert client.mode == Mode.INIT
        assert bytes(transport.written) == b"R\r"

    def test_request_special_datagram_writes_byte(self):
        cfg = make_configuration()
        transport = FakeTransport()
        client = STIM300(transport, configuration=cfg)
        client.request_part_number()
        client.request_serial_number()
        client.request_configuration()
        client.request_bias_trim()
        client.request_extended_error()
        assert bytes(transport.written) == b"N\rI\rC\rT\rE\r"
