"""Tests for the Service-Mode encode/decode helpers."""

import pytest

from pystim300.exceptions import CommandError, ProtocolError
from pystim300.service import (
    SERVICE_PROMPT,
    SERVICE_TERMINATOR,
    cmd_exit,
    cmd_information,
    cmd_save,
    cmd_set_datagram_format,
    cmd_set_line_termination,
    cmd_set_sample_rate,
    cmd_set_transmission,
    cmd_single_shot,
    detect_error,
    encode_command,
    find_exit_response_end,
    find_response_end,
    parse_exit_response,
    parse_response,
    raise_for_error,
)


class TestEncodeCommand:
    def test_simple_command(self):
        assert encode_command("a") == b"a\r"

    def test_command_with_args(self):
        assert encode_command("m 4") == b"m 4\r"

    def test_empty_command_is_just_cr(self):
        # Empty command re-displays the prompt (§7.5.3).
        assert encode_command("") == b"\r"

    @pytest.mark.parametrize("bad", ["a\r", "b\n", "c\rd"])
    def test_rejects_cr_lf_in_command(self, bad):
        with pytest.raises(ValueError):
            encode_command(bad)


class TestFindResponseEnd:
    def test_no_prompt_returns_minus_one(self):
        assert find_response_end(b"foo bar") == -1
        assert find_response_end(b"some response\r") == -1

    def test_finds_prompt_at_end(self):
        data = b"response\r>"
        assert find_response_end(data) == len(data)

    def test_finds_prompt_in_middle(self):
        # Useful for buffered streams where the next response also lurks.
        data = b"first\r>second"
        assert find_response_end(data) == len(b"first\r>")


class TestFindExitResponseEnd:
    def test_incomplete_buffer_returns_minus_one(self):
        assert find_exit_response_end(b"SYSTEM RETURNING TO NORMAL ") == -1
        assert find_exit_response_end(b"") == -1

    def test_finds_success_confirmation_line(self):
        data = b"SYSTEM RETURNING TO NORMAL MODE.\r"
        assert find_exit_response_end(data) == len(data)

    def test_success_marker_before_trailing_binary(self):
        # The device resumes Normal-Mode traffic right after the line; the
        # sentinel must stop at the CR, not scan into the binary tail.
        data = b"SYSTEM RETURNING TO INIT MODE.\r\x90\x00\x01\r>"
        assert find_exit_response_end(data) == len(b"SYSTEM RETURNING TO INIT MODE.\r")

    def test_success_marker_present_but_cr_not_yet_arrived(self):
        assert find_exit_response_end(b"SYSTEM RETURNING TO NORMAL MODE.") == -1

    def test_falls_back_to_prompt_on_rejected_parameter(self):
        # A bad parameter keeps the device in Service Mode: E<nnn> + prompt.
        data = b"E003 INVALID PARAMETER\r>"
        assert find_exit_response_end(data) == len(data)


class TestParseExitResponse:
    def test_success_line(self):
        raw = b"SYSTEM RETURNING TO NORMAL MODE.\r"
        resp = parse_exit_response(raw, "x N")
        assert resp.command == "x N"
        assert resp.lines == ("SYSTEM RETURNING TO NORMAL MODE.",)
        assert resp.raw == raw

    def test_keeps_echoed_command_line(self):
        # Real hardware echoes the command; harmless extra line.
        raw = b"x N\rSYSTEM RETURNING TO NORMAL MODE.\r"
        resp = parse_exit_response(raw, "x N")
        assert resp.lines == ("x N", "SYSTEM RETURNING TO NORMAL MODE.")

    def test_rejected_parameter_surfaces_through_raise_for_error(self):
        raw = b"E003 INVALID PARAMETER\r>"
        resp = parse_exit_response(raw, "x N")
        assert detect_error(resp) == (3, "INVALID PARAMETER")
        with pytest.raises(CommandError) as info:
            raise_for_error(resp)
        assert info.value.code == 3


class TestParseResponse:
    def test_single_line_response(self):
        raw = b"sample rate = 2000\r>"
        resp = parse_response(raw, "i m")
        assert resp.command == "i m"
        assert resp.lines == ("sample rate = 2000",)
        assert resp.raw == raw

    def test_multi_line_response(self):
        raw = b"line A\rline B\rline C\r>"
        resp = parse_response(raw, "i d")
        assert resp.lines == ("line A", "line B", "line C")

    def test_empty_body_yields_empty_lines(self):
        resp = parse_response(b"\r>", "")
        assert resp.lines == ()
        assert resp.raw == b"\r>"

    def test_missing_prompt_raises(self):
        with pytest.raises(ProtocolError):
            parse_response(b"no prompt here\r", "a")

    def test_raw_includes_prompt(self):
        raw = b"hello\r>"
        resp = parse_response(raw, "x")
        assert resp.raw.endswith(SERVICE_PROMPT)


class TestErrorDetection:
    def test_no_error_returns_none(self):
        resp = parse_response(b"sample rate = 2000\r>", "i m")
        assert detect_error(resp) is None

    def test_error_response(self):
        raw = b"E001 syntax error\r>"
        resp = parse_response(raw, "i z")
        err = detect_error(resp)
        assert err == (1, "syntax error")

    def test_raise_for_error_passes_through_good(self):
        resp = parse_response(b"ok\r>", "s")
        assert raise_for_error(resp) is resp

    def test_raise_for_error_raises_on_bad(self):
        resp = parse_response(b"E007 something broke\r>", "m 99")
        with pytest.raises(CommandError) as info:
            raise_for_error(resp)
        assert info.value.code == 7
        assert info.value.command == "m 99"
        assert b"E007" in info.value.raw


class TestCommandBuilders:
    def test_information_default(self):
        assert cmd_information() == "i"
        assert encode_command(cmd_information()) == b"i\r"

    def test_information_subcommand(self):
        assert cmd_information("d") == "i d"
        assert cmd_information("m") == "i m"

    def test_information_subcommand_validation(self):
        with pytest.raises(ValueError):
            cmd_information("dd")
        with pytest.raises(ValueError):
            cmd_information("1")
        with pytest.raises(ValueError):
            cmd_information("")

    def test_single_shot(self):
        assert cmd_single_shot() == "a"

    def test_save(self):
        assert cmd_save() == "s"

    def test_exit(self):
        # §9.14 / Table 9-54: upper-case letter parameter, not a digit.
        assert cmd_exit(to_normal=True) == "x N"
        assert cmd_exit(to_normal=False) == "x I"

    @pytest.mark.parametrize("code", [0, 1, 2, 3, 4, 5])
    def test_set_sample_rate_valid(self, code):
        assert cmd_set_sample_rate(code) == "m {0}".format(code)

    @pytest.mark.parametrize("bad", [-1, 6, 100])
    def test_set_sample_rate_invalid(self, bad):
        with pytest.raises(ValueError):
            cmd_set_sample_rate(bad)

    def test_set_datagram_format(self):
        assert cmd_set_datagram_format(0xA7, crlf=True) == "d a7,yes"
        assert cmd_set_datagram_format(0x90, crlf=False) == "d 90,no"

    def test_set_line_termination(self):
        assert cmd_set_line_termination(True) == "r 1"
        assert cmd_set_line_termination(False) == "r 0"

    def test_set_transmission(self):
        assert cmd_set_transmission(bit_rate=1843200, stop_bits=1, parity="none") == \
            "t 1843200,1,n"
        assert cmd_set_transmission(bit_rate=460800, stop_bits=2, parity="even") == \
            "t 460800,2,e"
        assert cmd_set_transmission(bit_rate=921600, stop_bits=1, parity="odd") == \
            "t 921600,1,o"

    def test_set_transmission_bad_parity(self):
        with pytest.raises(ValueError):
            cmd_set_transmission(parity="mark")
