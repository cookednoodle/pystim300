"""Stream Normal-Mode measurements from a STIM300 over RS422.

Usage::

    python examples/read_normal.py /dev/ttyUSB0

The example assumes the device is already in Normal Mode (default after
power-on) and that the active datagram format is the canonical full-
content datagram 0xA7 (rate + accel + incl + temperature, CRLF off).
Adjust the make_configuration call below if your device is configured
differently, or enter Service Mode first to query it.
"""

import argparse
import sys

# Add the project root to sys.path so this script runs from the repo
# without an editable install. Remove if pystim300 is installed normally.
import pathlib
_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pystim300 import STIM300, SerialTransport
from pystim300.configuration import compute_datagram_id, Configuration


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument("--bit-rate", type=int, default=1843200)
    parser.add_argument("--count", type=int, default=10,
                        help="Number of measurements to print before exiting")
    args = parser.parse_args(argv)

    # Canonical full-content datagram with no CR+LF.
    cfg = Configuration(
        raw_payload=bytes(21),
        revision_char="-",
        firmware_revision=0,
        sample_rate_hz=2000,
        has_pps=False,
        has_temperature=True,
        has_inclination=True,
        has_acceleration=True,
        crlf_termination=False,
        bit_rate=args.bit_rate,
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

    with SerialTransport(args.port, bit_rate=args.bit_rate) as transport:
        client = STIM300(transport, configuration=cfg, timeout=2.0)
        print("Reading {0} measurements from {1} ...".format(args.count, args.port))
        for i, m in enumerate(client.read_measurements(limit=args.count)):
            print("[{0:4d}] counter={1:3d} latency={2:5d}us  gyro={3}  accel={4}".format(
                i, m.counter, m.latency_us, m.gyro, m.accel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
