"""End-to-end tests for the STIM300 client against a FakeTransport."""

import pytest

from pystim300.client import (
    AuditEvent,
    InitSequence,
    MemoryAuditor,
    Mode,
    STIM300,
)
from pystim300.configuration import CONFIGURATION_PAYLOAD_LENGTH
from pystim300.crc import crc8_stim300, crc32_stim300
from pystim300.datagrams import (
    BIAS_TRIM_IDS,
    BIAS_TRIM_PAYLOAD_LENGTH,
    BiasTrimDatagram,
    PART_NUMBER_IDS,
    PART_NUMBER_PAYLOAD_LENGTH,
    PartNumberDatagram,
    SERIAL_NUMBER_IDS,
    SERIAL_NUMBER_PAYLOAD_LENGTH,
    SerialNumberDatagram,
)
from pystim300.exceptions import CommandError, ModeError, TimeoutError
from pystim300.normal import (
    Measurement,
    NormalStreamParser,
    SPECIAL_SPECS,
    build_normal_frame,
)
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
        # The EXIT response has no "\r>" prompt - the device leaves
        # Service Mode (§9.14, Figures 9-49/9-50).
        exit_response = b"SYSTEM RETURNING TO NORMAL MODE.\r"
        transport = FakeTransport(initial=banner + exit_response)
        client = STIM300(transport)
        client.enter_service()
        resp = client.exit_service()
        assert client.mode == Mode.NORMAL
        assert b"x N\r" in bytes(transport.written)
        assert "SYSTEM RETURNING TO NORMAL MODE." in resp.lines

    def test_exit_service_real_hardware_no_period(self):
        # Real TS1524 r.31 capture: the device echoes the command and emits
        # the success line WITHOUT the trailing period the datasheet figures
        # show, then resumes binary Normal-Mode traffic. Regression: this
        # used to time out because the sentinel required the "MODE." period.
        banner = b"\r>"
        exit_response = b"x N\rSYSTEM RETURNING TO NORMAL MODE\r\x93\x00\x01\x80"
        transport = FakeTransport(initial=banner + exit_response)
        client = STIM300(transport)
        client.enter_service()
        resp = client.exit_service()
        assert client.mode == Mode.NORMAL
        assert "SYSTEM RETURNING TO NORMAL MODE" in resp.lines
        # Bytes past the confirmation line are Normal-Mode traffic; they go
        # to carry-over for the next read, not into the EXIT response.
        assert client._carryover == b"\x93\x00\x01\x80"

    def test_exit_service_to_init(self):
        banner = b"\r>"
        exit_response = b"SYSTEM RETURNING TO INIT MODE.\r"
        transport = FakeTransport(initial=banner + exit_response)
        client = STIM300(transport)
        client.enter_service()
        client.exit_service(to_init=True)
        assert client.mode == Mode.INIT
        assert b"x I\r" in bytes(transport.written)

    def test_exit_service_rejected_parameter_raises(self):
        # A rejected parameter keeps the device in Service Mode: it emits
        # an E<nnn> line followed by the usual prompt.
        banner = b"\r>"
        exit_response = b"E003 INVALID PARAMETER\r>"
        transport = FakeTransport(initial=banner + exit_response)
        client = STIM300(transport)
        client.enter_service()
        with pytest.raises(CommandError) as info:
            client.exit_service()
        assert info.value.code == 3
        # Mode must NOT have changed - the device stayed in Service Mode.
        assert client.mode == Mode.SERVICE

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

    def test_enter_utility_skips_binary_datagram_preamble(self):
        # Per the §8.8 note the device finishes the in-progress Normal-Mode
        # datagram before acknowledging; that binary tail carries stray 0x0D
        # bytes. enter_utility must skip past them to the ack rather than
        # latching on the first CR (the real-hardware hang).
        preamble = b"\x92\x0d\x47\x00\x0d\x8e\xff\x0d\x01"
        ack = _utility_response("#UTILITYMODE,")
        transport = FakeTransport(initial=preamble + ack)
        client = STIM300(transport)
        resp = client.enter_utility()
        assert client.mode == Mode.UTILITY
        assert resp.command == "UTILITYMODE"
        assert resp.status == 0
        # raw is the clean acknowledgement only - binary preamble excluded.
        assert resp.raw == ack

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


def _build_special_frame(datagram_id: int, payload: bytes) -> bytes:
    """Wire bytes for an Init-Mode special datagram."""
    spec = SPECIAL_SPECS[datagram_id]
    assert len(payload) == spec.payload_length
    id_and_payload = bytes([datagram_id]) + payload
    crc = crc32_stim300(id_and_payload + b"\x00" * spec.dummy_bytes)
    frame = id_and_payload + crc.to_bytes(4, "big")
    if datagram_id in {PART_NUMBER_IDS[1], SERIAL_NUMBER_IDS[1], 0xBD,
                       BIAS_TRIM_IDS[1], 0xBF}:
        frame += b"\r\n"
    return frame


def _build_configuration_payload(*, bias_trim_at_startup: bool = False) -> bytes:
    """A minimal Configuration payload with no clusters, optional bias-trim flag."""
    payload = bytearray(CONFIGURATION_PAYLOAD_LENGTH)
    payload[0] = ord("-")   # revision_char
    payload[1] = 31         # firmware_revision
    payload[2] = 0b100_00000  # sample rate code 4 (2000Hz), no clusters, no CRLF
    payload[3] = 0b0011_0_00_0  # bit-rate code 3 (1843200), 1 stop, parity none
    if bias_trim_at_startup:
        payload[20] |= 0x02
    return bytes(payload)


def _make_init_sequence_bytes(*, bias_trim: bool = False) -> bytes:
    """Concatenated Init-Mode wire bytes (PN + SN + CFG [+ BT])."""
    pn = _build_special_frame(PART_NUMBER_IDS[0],
                                bytes(PART_NUMBER_PAYLOAD_LENGTH))
    sn_payload = bytearray(SERIAL_NUMBER_PAYLOAD_LENGTH)
    sn_payload[0] = ord("N")
    # Set the seven serial bytes to BCD digits ("01234567890ABC" sort of)
    sn_payload[1:8] = bytes([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD])
    sn = _build_special_frame(SERIAL_NUMBER_IDS[0], bytes(sn_payload))
    cfg_payload = _build_configuration_payload(bias_trim_at_startup=bias_trim)
    cfg = _build_special_frame(0xBC, cfg_payload)
    parts = [pn, sn, cfg]
    if bias_trim:
        bt = _build_special_frame(BIAS_TRIM_IDS[0],
                                    bytes(BIAS_TRIM_PAYLOAD_LENGTH))
        parts.append(bt)
    return b"".join(parts)


class TestInitSequence:
    def test_full_init_sequence_no_bias_trim(self):
        transport = FakeTransport(initial=_make_init_sequence_bytes(bias_trim=False))
        client = STIM300(transport)
        client.set_mode(Mode.INIT)
        seq = client.read_init_sequence(timeout=2.0)

        assert isinstance(seq, InitSequence)
        assert isinstance(seq.part_number, PartNumberDatagram)
        assert isinstance(seq.serial_number, SerialNumberDatagram)
        assert seq.bias_trim is None
        assert seq.configuration.bias_trim_at_startup is False
        # Client now knows its configuration and is back in Normal mode.
        assert client.configuration is seq.configuration
        assert client.mode == Mode.NORMAL
        assert client.stream_parser is not None
        # The raw stream is the full byte count we fed in.
        assert len(seq.raw) == len(_make_init_sequence_bytes(bias_trim=False))

    def test_full_init_sequence_with_bias_trim(self):
        transport = FakeTransport(initial=_make_init_sequence_bytes(bias_trim=True))
        client = STIM300(transport)
        seq = client.read_init_sequence(timeout=2.0)

        assert isinstance(seq.bias_trim, BiasTrimDatagram)
        assert seq.configuration.bias_trim_at_startup is True

    def test_timeout_when_no_data(self):
        transport = FakeTransport(initial=b"")
        client = STIM300(transport)
        with pytest.raises(TimeoutError):
            client.read_init_sequence(timeout=0.01)

    def test_partial_sequence_times_out_with_diagnostic(self):
        # PN only - SN and CFG missing.
        partial = _build_special_frame(PART_NUMBER_IDS[0],
                                         bytes(PART_NUMBER_PAYLOAD_LENGTH))
        transport = FakeTransport(initial=partial)
        client = STIM300(transport)
        with pytest.raises(TimeoutError) as info:
            client.read_init_sequence(timeout=0.05)
        msg = str(info.value)
        assert "SerialNumber" in msg
        assert "Configuration" in msg

    def test_bias_trim_required_when_configuration_says_so(self):
        # CFG declares bias_trim_at_startup=True but we don't send it.
        pn = _build_special_frame(PART_NUMBER_IDS[0],
                                    bytes(PART_NUMBER_PAYLOAD_LENGTH))
        sn_payload = bytearray(SERIAL_NUMBER_PAYLOAD_LENGTH)
        sn_payload[0] = ord("N")
        sn = _build_special_frame(SERIAL_NUMBER_IDS[0], bytes(sn_payload))
        cfg = _build_special_frame(0xBC,
                                     _build_configuration_payload(
                                         bias_trim_at_startup=True))
        transport = FakeTransport(initial=pn + sn + cfg)
        client = STIM300(transport)
        with pytest.raises(TimeoutError) as info:
            client.read_init_sequence(timeout=0.05)
        assert "BiasTrim" in str(info.value)

    def test_exit_service_to_init_then_read_init_sequence(self):
        # exit_service(to_init=True) leaves the Init datagrams that follow
        # "SYSTEM RETURNING TO INIT MODE." in the client's carry-over;
        # read_init_sequence must drain it rather than dropping the first
        # datagrams.
        banner = b"\r>"
        exit_response = b"SYSTEM RETURNING TO INIT MODE.\r"
        init_bytes = _make_init_sequence_bytes(bias_trim=False)
        transport = FakeTransport(initial=banner + exit_response + init_bytes)
        client = STIM300(transport)
        client.enter_service()
        client.exit_service(to_init=True)
        assert client.mode == Mode.INIT

        seq = client.read_init_sequence(timeout=2.0)
        assert isinstance(seq.part_number, PartNumberDatagram)
        assert isinstance(seq.serial_number, SerialNumberDatagram)
        assert client.mode == Mode.NORMAL
        assert len(seq.raw) == len(init_bytes)


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
