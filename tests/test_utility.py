"""Tests for the Utility-Mode encode/decode helpers."""

import pytest

from pystim300.crc import crc8_stim300
from pystim300.exceptions import CommandError, CrcError, ProtocolError
from pystim300.utility import (
    STATUS_DESCRIPTIONS,
    cmd_exit_to_normal,
    cmd_information,
    cmd_save,
    cmd_set_datagram,
    cmd_set_sample_rate,
    encode_command,
    find_entry_response_end,
    find_response_end,
    parse_entry_response,
    parse_response,
    raise_for_error,
)


def _crc_of(s: str) -> int:
    return crc8_stim300(s.encode("ascii"))


# Entry acknowledgement bytes, per Figure 10-1 (CRC 234 over "#UTILITYMODE,").
_ENTRY_ACK = b"#UTILITYMODE,234\r"
# A binary Normal-Mode datagram tail; per the §8.8 note the device finishes the
# in-progress datagram before sending the ack. It deliberately contains stray
# CR (0x0D) bytes - the case the plain first-CR sentinel cannot survive.
_BINARY_PREAMBLE = b"\x92\x0d\x47\x00\x0d\x8e\xff\x0d\x01"


class TestEncodeCommand:
    def test_no_params(self):
        wire = encode_command("isn")
        # $isn,<crc>\r per Figure 10-2 example (CRC = 28 over "$isn,")
        assert wire == b"$isn,28\r"
        assert _crc_of("$isn,") == 28

    def test_one_param(self):
        wire = encode_command("sm", 4)
        crc = _crc_of("$sm,4,")
        assert wire == "$sm,4,{0}\r".format(crc).encode("ascii")

    def test_two_params(self):
        wire = encode_command("sd", "a7", "1")
        crc = _crc_of("$sd,a7,1,")
        assert wire == "$sd,a7,1,{0}\r".format(crc).encode("ascii")

    def test_float_param(self):
        wire = encode_command("sbto", 0.00123)
        crc = _crc_of("$sbto,0.00123,")
        assert wire == "$sbto,0.00123,{0}\r".format(crc).encode("ascii")

    @pytest.mark.parametrize("bad", ["", "foo,bar", "foo\r", "foo\n"])
    def test_invalid_name(self, bad):
        with pytest.raises(ValueError):
            encode_command(bad)


class TestFindResponseEnd:
    def test_no_cr(self):
        assert find_response_end(b"#isn,0,N123") == -1

    def test_cr_at_end(self):
        data = b"#isn,0,N123,32\r"
        assert find_response_end(data) == len(data)


class TestFindEntryResponseEnd:
    def test_clean_ack(self):
        assert find_entry_response_end(_ENTRY_ACK) == len(_ENTRY_ACK)

    def test_skips_binary_preamble_with_embedded_cr(self):
        # The §8.8 datagram tail before the ack carries stray 0x0D bytes; the
        # sentinel must anchor on the "#UTILITYMODE," marker, not the first CR.
        data = _BINARY_PREAMBLE + _ENTRY_ACK
        assert find_entry_response_end(data) == len(data)

    def test_marker_present_but_cr_not_yet_arrived(self):
        assert find_entry_response_end(_BINARY_PREAMBLE + b"#UTILITYMODE,234") == -1

    def test_no_marker_returns_minus_one(self):
        assert find_entry_response_end(_BINARY_PREAMBLE) == -1
        assert find_entry_response_end(b"") == -1


class TestParseResponse:
    def test_isn_response(self):
        # Per Figure 10-2: $isn,28 -> #isn,0,N2558184602002,32
        body = "#isn,0,N2558184602002,"
        crc = _crc_of(body)
        raw = (body + str(crc) + "\r").encode("ascii")
        # The example uses 32 as the CRC; double-check that's consistent.
        assert crc == 32, "CRC of {0!r} should be 32 per Figure 10-2; got {1}".format(body, crc)

        resp = parse_response(raw)
        assert resp.command == "isn"
        assert resp.status == 0
        assert resp.fields == ("N2558184602002",)
        assert resp.raw == raw

    def test_entry_acknowledgement(self):
        # Figure 10-1: #UTILITYMODE,234\r
        body = "#UTILITYMODE,"
        crc = _crc_of(body)
        assert crc == 234, "CRC of '#UTILITYMODE,' should be 234 per Figure 10-1; got {0}".format(crc)
        raw = (body + str(crc) + "\r").encode("ascii")
        resp = parse_response(raw)
        assert resp.command == "UTILITYMODE"
        assert resp.status == 0
        assert resp.fields == ()

    def test_invalid_command_response(self):
        # Figure 10-3: #,1,180\r
        body = "#,1,"
        crc = _crc_of(body)
        raw = (body + str(crc) + "\r").encode("ascii")
        resp = parse_response(raw)
        assert resp.command == ""
        assert resp.status == 1
        assert resp.fields == ()

    def test_missing_cr_raises(self):
        with pytest.raises(ProtocolError):
            parse_response(b"#isn,0,N123,32")

    def test_missing_hash_raises(self):
        with pytest.raises(ProtocolError):
            parse_response(b"isn,0,N123,32\r")

    def test_bad_crc_raises(self):
        # Construct a response with a wrong CRC.
        body = "#isn,0,N123,"
        wrong = (body + "99" + "\r").encode("ascii")
        with pytest.raises(CrcError):
            parse_response(wrong)

    def test_non_integer_status_raises(self):
        body = "#isn,abc,"
        crc = _crc_of(body)
        raw = (body + str(crc) + "\r").encode("ascii")
        with pytest.raises(ProtocolError):
            parse_response(raw)


class TestParseEntryResponse:
    def test_clean_ack(self):
        resp = parse_entry_response(_ENTRY_ACK)
        assert resp.command == "UTILITYMODE"
        assert resp.status == 0
        assert resp.fields == ()
        assert resp.raw == _ENTRY_ACK

    def test_strips_binary_preamble(self):
        resp = parse_entry_response(_BINARY_PREAMBLE + _ENTRY_ACK)
        assert resp.command == "UTILITYMODE"
        assert resp.status == 0
        # raw is the clean acknowledgement only - preamble excluded.
        assert resp.raw == _ENTRY_ACK

    def test_no_marker_raises(self):
        with pytest.raises(ProtocolError):
            parse_entry_response(_BINARY_PREAMBLE)


class TestStatusCodes:
    @pytest.mark.parametrize("code", list(range(1, 9)))
    def test_each_error_code_raises_command_error(self, code):
        body = "#sm,{0},".format(code)
        crc = _crc_of(body)
        raw = (body + str(crc) + "\r").encode("ascii")
        resp = parse_response(raw)
        with pytest.raises(CommandError) as info:
            raise_for_error(resp)
        assert info.value.code == code
        assert info.value.command == "sm"

    def test_ok_status_passes(self):
        body = "#sm,0,"
        crc = _crc_of(body)
        raw = (body + str(crc) + "\r").encode("ascii")
        resp = parse_response(raw)
        assert raise_for_error(resp) is resp

    def test_all_status_codes_have_descriptions(self):
        for code in range(9):
            assert code in STATUS_DESCRIPTIONS


class TestCommandBuilders:
    def test_information_query(self):
        wire = cmd_information("isn")
        assert wire.startswith(b"$isn,")
        assert wire.endswith(b"\r")

    def test_information_must_start_with_i(self):
        with pytest.raises(ValueError):
            cmd_information("xn")

    def test_exit_to_normal(self):
        wire = cmd_exit_to_normal()
        crc = _crc_of("$xn,")
        assert wire == "$xn,{0}\r".format(crc).encode("ascii")

    def test_save(self):
        wire = cmd_save()
        crc = _crc_of("$save,")
        assert wire == "$save,{0}\r".format(crc).encode("ascii")

    @pytest.mark.parametrize("code", [0, 1, 2, 3, 4, 5])
    def test_set_sample_rate_valid(self, code):
        wire = cmd_set_sample_rate(code)
        crc = _crc_of("$sm,{0},".format(code))
        assert wire == "$sm,{0},{1}\r".format(code, crc).encode("ascii")

    @pytest.mark.parametrize("bad", [-1, 6, 100])
    def test_set_sample_rate_invalid(self, bad):
        with pytest.raises(ValueError):
            cmd_set_sample_rate(bad)

    def test_set_datagram(self):
        wire = cmd_set_datagram(0xA7, crlf=True)
        crc = _crc_of("$sd,a7,1,")
        assert wire == "$sd,a7,1,{0}\r".format(crc).encode("ascii")


class TestRoundTrip:
    def test_encode_then_parse_response(self):
        # A device-style response we can build with the same CRC machinery.
        body = "#im,0,2000,"
        crc = _crc_of(body)
        raw = (body + str(crc) + "\r").encode("ascii")
        resp = parse_response(raw)
        assert resp.command == "im"
        assert resp.status == 0
        assert resp.fields == ("2000",)
