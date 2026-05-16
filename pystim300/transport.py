"""Transport abstraction for talking to the STIM300.

The codec never touches I/O directly; everything goes through a small
``Transport`` Protocol with three methods (``read``, ``write``, ``close``).
This keeps the codec dependency-free and makes it trivial to drop in an
async sibling later (or any other I/O mechanism).

Two concrete implementations ship in this module:

* ``SerialTransport`` - pyserial-backed, for real hardware. pyserial is
  imported lazily inside the constructor so the codec works without the
  optional ``[serial]`` extra installed.
* ``FakeTransport`` - in-memory, for tests and offline development.
  Supports a simple ``feed(bytes)`` API plus an optional scripted
  request/response mode that asserts the exact bytes sent before
  delivering a canned reply.
"""

from collections import deque
from typing import Callable, Deque, Optional, Sequence, Tuple

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover - Python <3.8 fallback (unreachable here)
    from typing_extensions import Protocol, runtime_checkable  # type: ignore


@runtime_checkable
class Transport(Protocol):
    """Minimal byte-stream interface the client owns.

    ``read`` returns *up to* ``n`` bytes; it may return fewer (including 0
    bytes) when the optional timeout elapses with nothing read. The
    streaming parser tolerates short reads, so transports should not block
    forever waiting for exactly ``n`` bytes.
    """

    def read(self, n: int, timeout: Optional[float] = None) -> bytes: ...
    def write(self, data: bytes) -> None: ...
    def close(self) -> None: ...


class FakeTransport:
    """In-memory transport for tests and offline work.

    Pre-loaded read data is appended via the constructor or ``feed`` and
    consumed by successive ``read`` calls. All bytes ever written are
    accumulated on ``written`` for assertions.

    Optional scripted mode: pass ``scripted`` as a sequence of
    ``(expected_prefix, reply)`` pairs. The N-th ``write`` call must start
    with ``expected_prefix`` (otherwise ``AssertionError``); the matching
    ``reply`` is appended to the read buffer immediately so the next
    ``read`` returns it. This is what Service-Mode tests use to drive a
    multi-turn exchange.
    """

    def __init__(
        self,
        initial: bytes = b"",
        *,
        scripted: Optional[Sequence[Tuple[bytes, bytes]]] = None,
    ) -> None:
        self._read_buf = bytearray(initial)
        self.written = bytearray()
        self._script: Deque[Tuple[bytes, bytes]] = deque(scripted or ())
        self._closed = False

    def feed(self, data: bytes) -> None:
        """Append ``data`` to the bytes the next ``read`` will draw from."""
        self._read_buf.extend(data)

    def read(self, n: int, timeout: Optional[float] = None) -> bytes:
        if self._closed:
            raise OSError("FakeTransport is closed")
        if n <= 0:
            return b""
        take = min(n, len(self._read_buf))
        if take == 0:
            return b""
        out = bytes(self._read_buf[:take])
        del self._read_buf[:take]
        return out

    def write(self, data: bytes) -> None:
        if self._closed:
            raise OSError("FakeTransport is closed")
        self.written.extend(data)
        if self._script:
            expected_prefix, reply = self._script.popleft()
            if not data.startswith(expected_prefix):
                raise AssertionError(
                    "scripted write mismatch: expected prefix {0!r}, got {1!r}".format(
                        expected_prefix, data))
            self._read_buf.extend(reply)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "FakeTransport":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def script_remaining(self) -> int:
        return len(self._script)


class SerialTransport:
    """pyserial-backed Transport. Imports pyserial lazily.

    Defaults to the STIM300 factory settings (1843200 8N1) per
    Table 5-12 (p.28). Set ``timeout=None`` for blocking reads or pass a
    finite value to bound each ``read`` call. The constructor opens the
    port; ``close`` releases it. The object is a context manager.
    """

    def __init__(
        self,
        port: str,
        *,
        bit_rate: int = 1843200,
        parity: str = "N",        # "N", "E", or "O"
        stop_bits: int = 1,
        timeout: Optional[float] = 1.0,
    ) -> None:
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for SerialTransport; install pystim300[serial]"
            ) from exc

        parity_map = {
            "N": serial.PARITY_NONE,
            "E": serial.PARITY_EVEN,
            "O": serial.PARITY_ODD,
        }
        stop_map = {
            1: serial.STOPBITS_ONE,
            2: serial.STOPBITS_TWO,
        }
        if parity not in parity_map:
            raise ValueError("parity must be one of 'N','E','O', got {0!r}".format(parity))
        if stop_bits not in stop_map:
            raise ValueError("stop_bits must be 1 or 2, got {0}".format(stop_bits))

        self._serial = serial.Serial(
            port=port,
            baudrate=bit_rate,
            bytesize=serial.EIGHTBITS,
            parity=parity_map[parity],
            stopbits=stop_map[stop_bits],
            timeout=timeout,
        )

    def read(self, n: int, timeout: Optional[float] = None) -> bytes:
        if timeout is not None and self._serial.timeout != timeout:
            self._serial.timeout = timeout
        return self._serial.read(n)

    def write(self, data: bytes) -> None:
        self._serial.write(data)
        self._serial.flush()

    def close(self) -> None:
        if self._serial.is_open:
            self._serial.close()

    def __enter__(self) -> "SerialTransport":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
