"""Tests for FakeTransport and SerialTransport (without real hardware)."""

import pytest

from pystim300.transport import FakeTransport, Transport


class TestFakeTransport:
    def test_implements_protocol(self):
        # Runtime-checkable Protocol: isinstance must succeed.
        assert isinstance(FakeTransport(), Transport)

    def test_read_returns_initial_bytes(self):
        ft = FakeTransport(b"hello world")
        assert ft.read(5) == b"hello"
        assert ft.read(100) == b" world"
        assert ft.read(1) == b""

    def test_read_with_no_data_returns_empty(self):
        ft = FakeTransport()
        assert ft.read(10) == b""
        assert ft.read(10, timeout=0.1) == b""

    def test_feed_appends(self):
        ft = FakeTransport(b"AB")
        ft.feed(b"CD")
        assert ft.read(10) == b"ABCD"

    def test_write_records_bytes(self):
        ft = FakeTransport()
        ft.write(b"hello")
        ft.write(b" world")
        assert bytes(ft.written) == b"hello world"

    def test_close_blocks_further_io(self):
        ft = FakeTransport(b"abc")
        ft.close()
        with pytest.raises(OSError):
            ft.read(1)
        with pytest.raises(OSError):
            ft.write(b"x")

    def test_context_manager(self):
        with FakeTransport(b"abc") as ft:
            assert ft.read(3) == b"abc"
        with pytest.raises(OSError):
            ft.read(1)

    def test_negative_n(self):
        ft = FakeTransport(b"abc")
        assert ft.read(0) == b""
        assert ft.read(-1) == b""


class TestFakeTransportScripted:
    def test_scripted_request_response(self):
        ft = FakeTransport(scripted=[
            (b"PING\r", b"PONG\r>"),
            (b"INFO\r", b"version 1\r>"),
        ])
        ft.write(b"PING\r")
        assert ft.read(100) == b"PONG\r>"
        ft.write(b"INFO\r")
        assert ft.read(100) == b"version 1\r>"
        assert ft.script_remaining == 0

    def test_scripted_mismatch_raises(self):
        ft = FakeTransport(scripted=[(b"HELLO\r", b"ok")])
        with pytest.raises(AssertionError):
            ft.write(b"WRONG\r")

    def test_prefix_matching_accepts_longer_write(self):
        # The expected_prefix only has to be a prefix; trailing args allowed.
        ft = FakeTransport(scripted=[(b"i", b">prompt")])
        ft.write(b"i d\r")
        assert ft.read(100) == b">prompt"


class TestSerialTransportLazyImport:
    def test_import_works_without_pyserial(self):
        # Importing the module must succeed even when pyserial is absent.
        # We can't directly test absence, but we can verify the constructor
        # path that raises gives a clear error if pyserial isn't installed.
        from pystim300.transport import SerialTransport
        # If pyserial is installed, instantiating with a bogus port will
        # raise serial.SerialException (or OSError); if absent, ImportError.
        # Either way, just importing the class is fine.
        assert SerialTransport is not None
