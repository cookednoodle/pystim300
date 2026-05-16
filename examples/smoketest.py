"""Hardware bring-up smoke test for a STIM300.

Drains the Normal-Mode stream for N seconds and reports:

* achieved frame rate vs configured
* counter-delta jumps (dropped frames)
* CRC error count (visible as parser resync events)
* any STATUS bits that were ever set

This is the first end-to-end check on real hardware. Captured frames
can later be saved to ``tests/vectors/`` to lock in device-grounded
test vectors for the CRC and parser.

Usage::

    python examples/smoketest.py /dev/ttyUSB0 --seconds 10
"""

import argparse
import pathlib
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pystim300 import STIM300, SerialTransport
from pystim300.configuration import Configuration, compute_datagram_id


def _default_configuration(bit_rate: int) -> Configuration:
    return Configuration(
        raw_payload=bytes(21),
        revision_char="-",
        firmware_revision=0,
        sample_rate_hz=2000,
        has_pps=False,
        has_temperature=True,
        has_inclination=True,
        has_acceleration=True,
        crlf_termination=False,
        bit_rate=bit_rate,
        stop_bits=1,
        parity="none",
        line_termination=False,
        gyro_axes_active=(True, True, True),
        gyro_output_unit=0,
        gyro_lp_filter_hz=(262, 262, 262),
        gyro_g_compensation=0,
        accel_axes_active=(True, True, True),
        accel_output_unit=0,
        accel_lp_filter_hz=(262, 262, 262),
        incl_axes_active=(True, True, True),
        incl_output_unit=0,
        incl_lp_filter_hz=(262, 262, 262),
        pps_output_unit=0,
        pps_lp_filter_hz=262,
        has_aux_input=False,
        has_pps_input=False,
        accel_range_g=(10, 10, 10),
        datagram_id=compute_datagram_id(
            has_pps=False, has_temperature=True,
            has_inclination=True, has_acceleration=True),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port")
    parser.add_argument("--bit-rate", type=int, default=1843200)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--expected-rate", type=int, default=2000,
                        help="Configured sample rate in Hz")
    args = parser.parse_args(argv)

    cfg = _default_configuration(args.bit_rate)
    flags_seen = {}
    last_counter = None
    drops = 0
    frames = 0

    print("Smoke test on {0} @ {1} bps for {2:.1f}s ...".format(
        args.port, args.bit_rate, args.seconds))

    with SerialTransport(args.port, bit_rate=args.bit_rate, timeout=1.0) as transport:
        client = STIM300(transport, configuration=cfg, timeout=2.0)
        deadline = time.monotonic() + args.seconds
        for m in client.read_measurements():
            frames += 1
            # Counter wraps at 256.
            if last_counter is not None:
                delta = (m.counter - last_counter) & 0xFF
                if delta != 1:
                    drops += (delta - 1)
            last_counter = m.counter
            for attr in ("system_integrity_error", "startup",
                          "outside_operating_conditions", "overload",
                          "channel_error"):
                if getattr(m.gyro_status, attr):
                    flags_seen[attr] = flags_seen.get(attr, 0) + 1
            if time.monotonic() >= deadline:
                break

    elapsed = args.seconds
    achieved = frames / elapsed if elapsed > 0 else 0
    print()
    print("Frames received:    {0}".format(frames))
    print("Elapsed:            {0:.2f}s".format(elapsed))
    print("Achieved rate:      {0:.1f} Hz".format(achieved))
    print("Configured rate:    {0} Hz".format(args.expected_rate))
    print("Rate error:         {0:+.1f}%".format(
        100.0 * (achieved - args.expected_rate) / args.expected_rate))
    print("Counter-delta drops: {0}".format(drops))
    print("Parser resyncs:     {0}".format(client._normal_parser.resync_events))
    print("Bytes discarded:    {0}".format(client._normal_parser.bytes_discarded))
    if flags_seen:
        print("STATUS flags seen:")
        for name, n in sorted(flags_seen.items()):
            print("  {0:32s} {1} frames".format(name, n))
    else:
        print("STATUS flags seen:  (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
