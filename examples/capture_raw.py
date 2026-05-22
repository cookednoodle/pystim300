"""Stage 1: capture the raw STIM300 serial stream with host timestamps.

Reads the device's Init sequence to learn the fixed Normal-Mode datagram
length, then captures the stream **one datagram per read** - each datagram
stamped with ``time.perf_counter()`` the instant it is in hand - into a
binary ``.cap`` file. No datagram decoding happens on the capture path,
so the host timestamps reflect true arrival timing as closely as the
serial link allows. Turn the ``.cap`` into a CSV afterwards with
``decode_capture.py``.

For best timestamp accuracy lower the USB-serial adapter's latency timer
to 1 ms (on Linux: write ``1`` to
``/sys/bus/usb-serial/devices/<dev>/latency_timer``); otherwise the
adapter batches several milliseconds of bytes before the host sees them.

Usage::

    python examples/capture_raw.py /dev/ttyUSB0 --duration 1.0
    python examples/capture_raw.py /dev/ttyUSB0 --no-reset --out run.cap
"""

import argparse
import datetime
import pathlib
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pystim300 import Configuration, NormalStreamParser
from pystim300.transport import SerialTransport, Transport

from examples.capture_format import write_header, write_record

# Init-phase reads are small fixed chunks; 16 < the smallest Normal-Mode
# datagram (18 bytes) so the over-read past the Init sequence is always
# less than one frame.
_PHASE_A_CHUNK = 16


def run_capture(transport: Transport, out_path, *, bit_rate: int = 0,
                port: str = "", reset_first: bool = True,
                duration=None, max_records=None,
                init_timeout: float = 10.0, read_timeout: float = 1.0,
                log=None):
    """Capture the raw stream from ``transport`` into ``out_path``.

    Phase A reads the Init sequence to derive the fixed Normal-Mode
    datagram length; Phase B then captures one datagram per read until
    ``duration`` seconds elapse, ``max_records`` Phase-B records are
    written, or the caller interrupts with Ctrl-C. ``log`` is an optional
    file object for human-readable progress.

    Returns ``(phase_a_records, phase_b_records, total_bytes)``.
    """
    def _log(msg):
        if log is not None:
            print(msg, file=log)

    epoch0 = time.time()
    perf0 = time.perf_counter()
    out_path = pathlib.Path(out_path)
    phase_a = 0
    phase_b = 0
    total_bytes = 0
    f = open(out_path, "wb")
    try:
        write_header(f, epoch0=epoch0, perf0=perf0, bit_rate=bit_rate, port=port)
        if reset_first:
            transport.write(b"R\r")
            _log("Sent reset (R); waiting for Init-Mode datagrams...")
        else:
            _log("Waiting for Init-Mode datagrams (power-cycle the STIM300 now)...")

        # --- Phase A: capture the Init sequence, derive datagram length ---
        parser = NormalStreamParser(configuration=None)
        configuration = None
        deadline = time.perf_counter() + init_timeout
        while configuration is None:
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    "no Configuration datagram within {0}s of capture start; "
                    "is the device connected and emitting its Init sequence?"
                    .format(init_timeout))
            chunk = transport.read(_PHASE_A_CHUNK, timeout=read_timeout)
            if not chunk:
                continue
            perf = time.perf_counter()
            write_record(f, perf, chunk)
            phase_a += 1
            total_bytes += len(chunk)
            for record in parser.feed(chunk):
                if isinstance(record, Configuration):
                    configuration = record
                    break

        frame_length = configuration.frame_length()
        _log("Init captured: datagram 0x{0:02X}, {1} bytes/frame, {2} Hz.".format(
            configuration.datagram_id, frame_length, configuration.sample_rate_hz))

        # --- Phase B: per-datagram capture (timing-critical, no parsing) --
        start = time.monotonic()
        last_flush = start
        last_progress = start
        while True:
            if max_records is not None and phase_b >= max_records:
                break
            if duration is not None and time.monotonic() - start >= duration:
                break
            data = transport.read(frame_length, timeout=read_timeout)
            if data:
                perf = time.perf_counter()
                write_record(f, perf, data)
                phase_b += 1
                total_bytes += len(data)
            now = time.monotonic()
            if now - last_flush >= 1.0:
                f.flush()
                last_flush = now
            if log is not None and now - last_progress >= 10.0:
                _log("  captured {0} datagrams, {1} bytes, {2:.0f}s elapsed".format(
                    phase_b, total_bytes, now - start))
                last_progress = now
    except KeyboardInterrupt:
        _log("Interrupted; finalizing capture file.")
    finally:
        f.flush()
        f.close()

    _log("Capture complete: {0} init + {1} datagram records, {2} bytes -> {3}".format(
        phase_a, phase_b, total_bytes, out_path))
    return phase_a, phase_b, total_bytes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("port", help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument("--bit-rate", type=int, default=1843200,
                        help="Serial bit rate (STIM300 factory default 1843200).")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="Capture file path (default: timestamped *.cap).")
    parser.add_argument("--duration", type=float, default=None,
                        help="Capture duration in HOURS (default: until Ctrl-C).")
    parser.add_argument("--no-reset", action="store_true",
                        help="Wait for an operator power-cycle instead of "
                             "issuing a commanded reset.")
    args = parser.parse_args(argv)

    out = args.out
    if out is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = pathlib.Path("stim300_{0}.cap".format(stamp))
    duration_s = args.duration * 3600.0 if args.duration is not None else None

    print("Tip: for best timestamp accuracy lower the USB-serial latency timer "
          "to 1 ms\n"
          "     (Linux: echo 1 | sudo tee "
          "/sys/bus/usb-serial/devices/<dev>/latency_timer).", file=sys.stderr)
    with SerialTransport(args.port, bit_rate=args.bit_rate, timeout=1.0) as transport:
        run_capture(transport, out, bit_rate=args.bit_rate, port=args.port,
                    reset_first=not args.no_reset, duration=duration_s,
                    log=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
