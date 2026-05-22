"""Stage 2: decode a raw ``.cap`` capture into an engineering-units CSV.

Replays the byte stream captured by ``capture_raw.py`` through the
library's Normal-Mode parser, writing one CSV row per measurement - each
row carrying the host receive time recorded during capture, plus the
device counter and latency for cross-checking. Fully offline; re-run it
as often as needed without touching hardware.

A ``<csv>.meta.json`` sidecar records the device identity, the full
decoded configuration, the per-cluster engineering units, the CSV column
list, and a decode summary (resyncs, dropped frames, and inter-sample
interval statistics for evaluating timing consistency).

Usage::

    python examples/decode_capture.py stim300_20260522_120000.cap
    python examples/decode_capture.py run.cap --out run.csv --skip-startup
"""

import argparse
import csv
import datetime
import json
import pathlib
import statistics
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pystim300 import Configuration, Measurement, NormalStreamParser
from pystim300.datagrams import (
    BiasTrimDatagram,
    PartNumberDatagram,
    SerialNumberDatagram,
)

from examples.capture_format import iter_records, read_header


def _columns(cfg: Configuration):
    """Build the ordered CSV column list for a given device configuration."""
    cols = ["host_monotonic", "host_epoch", "host_time_utc",
            "counter", "counter_gap", "latency_us",
            "gyro_x", "gyro_y", "gyro_z", "gyro_status"]
    if cfg.has_acceleration:
        cols += ["accel_x", "accel_y", "accel_z", "accel_status"]
    if cfg.has_inclination:
        cols += ["incl_x", "incl_y", "incl_z", "incl_status"]
    if cfg.has_temperature:
        cols += ["gyro_temp_x", "gyro_temp_y", "gyro_temp_z", "gyro_temp_status"]
        if cfg.has_acceleration:
            cols += ["accel_temp_x", "accel_temp_y", "accel_temp_z",
                     "accel_temp_status"]
        if cfg.has_inclination:
            cols += ["incl_temp_x", "incl_temp_y", "incl_temp_z",
                     "incl_temp_status"]
    if cfg.has_pps:
        cols += ["pps", "pps_status"]
    return cols


def _row(m: Measurement, cfg: Configuration, host_monotonic: float,
         host_epoch: float, counter_gap: int):
    """Build one CSV row; column order matches ``_columns``."""
    iso = datetime.datetime.fromtimestamp(
        host_epoch, datetime.timezone.utc).isoformat()
    row = [host_monotonic, host_epoch, iso,
           m.counter, counter_gap, m.latency_us,
           m.gyro[0], m.gyro[1], m.gyro[2], m.gyro_status.raw]
    if cfg.has_acceleration:
        row += [m.accel[0], m.accel[1], m.accel[2], m.accel_status.raw]
    if cfg.has_inclination:
        row += [m.incl[0], m.incl[1], m.incl[2], m.incl_status.raw]
    if cfg.has_temperature:
        row += [m.gyro_temp[0], m.gyro_temp[1], m.gyro_temp[2],
                m.gyro_temp_status.raw]
        if cfg.has_acceleration:
            row += [m.accel_temp[0], m.accel_temp[1], m.accel_temp[2],
                    m.accel_temp_status.raw]
        if cfg.has_inclination:
            row += [m.incl_temp[0], m.incl_temp[1], m.incl_temp[2],
                    m.incl_temp_status.raw]
    if cfg.has_pps:
        row += [m.pps, m.pps_status.raw]
    return row


def _units(cfg: Configuration):
    """Best-effort per-cluster engineering units (Table 5-16 unit codes)."""
    # Gyro: bit 0 of the unit code selects rate ([deg/s]) vs angle ([deg]).
    gyro = "deg" if (cfg.gyro_output_unit & 0b0001) else "deg/s"
    # Accel/incl: codes 0 / 2 are [g]; the rest are incremental/integrated.
    accel = "g" if cfg.accel_output_unit in (0b0000, 0b0010) else "m/s/sample"
    incl = "g" if cfg.incl_output_unit in (0b0000, 0b0010) else "m/s/sample"
    units = {"gyro": gyro, "latency_us": "microseconds"}
    if cfg.has_acceleration:
        units["accel"] = accel
    if cfg.has_inclination:
        units["incl"] = incl
    if cfg.has_temperature:
        units["temperature"] = "degC"
    if cfg.has_pps:
        units["pps"] = "microseconds (TIME_SINCE) or filtered [0,1]"
    return units


def _config_dict(cfg: Configuration):
    return {
        "revision_char": cfg.revision_char,
        "firmware_revision": cfg.firmware_revision,
        "sample_rate_hz": cfg.sample_rate_hz,
        "bit_rate": cfg.bit_rate,
        "datagram_id": "0x{0:02X}".format(cfg.datagram_id),
        "frame_length_bytes": cfg.frame_length(),
        "crlf_termination": cfg.crlf_termination,
        "has_acceleration": cfg.has_acceleration,
        "has_inclination": cfg.has_inclination,
        "has_temperature": cfg.has_temperature,
        "has_pps": cfg.has_pps,
        "gyro_output_unit": cfg.gyro_output_unit,
        "accel_output_unit": cfg.accel_output_unit,
        "incl_output_unit": cfg.incl_output_unit,
        "pps_output_unit": cfg.pps_output_unit,
        "accel_range_g": list(cfg.accel_range_g),
        "bias_trim_at_startup": cfg.bias_trim_at_startup,
    }


def decode_capture(capture_path, csv_path, *, skip_startup: bool = False):
    """Decode ``capture_path`` into ``csv_path`` plus a ``.meta.json`` sidecar.

    Returns a summary dict (row count, parser resync counters, dropped
    frames, and inter-sample interval statistics).
    """
    capture_path = pathlib.Path(capture_path)
    csv_path = pathlib.Path(csv_path)
    meta_path = csv_path.with_name(csv_path.name + ".meta.json")

    parser = NormalStreamParser(configuration=None)
    cfg = None
    part_number = None
    serial_number = None
    bias_trim = None
    columns = None
    rows = 0
    prev_counter = None
    dropped = 0
    monotonics = []
    csv_file = None
    writer = None

    try:
        with open(capture_path, "rb") as f:
            header = read_header(f)
            for perf, payload in iter_records(f):
                for record in parser.feed(payload):
                    if isinstance(record, Configuration):
                        if cfg is None:
                            cfg = record
                            # Re-key the same parser so the bytes already
                            # buffered past the Configuration datagram are
                            # framed as Normal-Mode measurements.
                            parser.update_configuration(cfg)
                            columns = _columns(cfg)
                            csv_file = open(csv_path, "w", newline="")
                            writer = csv.writer(csv_file)
                            writer.writerow(columns)
                    elif isinstance(record, PartNumberDatagram):
                        part_number = record
                    elif isinstance(record, SerialNumberDatagram):
                        serial_number = record
                    elif isinstance(record, BiasTrimDatagram):
                        bias_trim = record
                    elif isinstance(record, Measurement):
                        gap = (0 if prev_counter is None
                               else (record.counter - prev_counter - 1) % 256)
                        prev_counter = record.counter
                        if skip_startup and record.gyro_status.startup:
                            continue
                        dropped += gap
                        host_monotonic = perf - header.perf0
                        host_epoch = header.epoch0 + host_monotonic
                        writer.writerow(
                            _row(record, cfg, host_monotonic, host_epoch, gap))
                        monotonics.append(host_monotonic)
                        rows += 1
    finally:
        if csv_file is not None:
            csv_file.close()

    if cfg is None:
        raise ValueError(
            "no Configuration datagram in {0}; the capture must start at or "
            "before device initialization (use capture_raw.py with a reset "
            "or an operator power-cycle)".format(capture_path))

    intervals = [b - a for a, b in zip(monotonics, monotonics[1:])]
    interval_stats = {}
    if intervals:
        interval_stats = {
            "min_s": min(intervals),
            "max_s": max(intervals),
            "mean_s": statistics.mean(intervals),
            "stdev_s": statistics.pstdev(intervals) if len(intervals) > 1 else 0.0,
        }
    summary = {
        "rows": rows,
        "resync_events": parser.resync_events,
        "bytes_discarded": parser.bytes_discarded,
        "dropped_frames": dropped,
        "interval_stats": interval_stats,
    }

    meta = {
        "source_capture": capture_path.name,
        "capture_header": {
            "version": header.version,
            "epoch0": header.epoch0,
            "perf0": header.perf0,
            "bit_rate": header.bit_rate,
            "port": header.port,
        },
        "part_number": part_number.part_number if part_number else None,
        "serial_number": serial_number.serial_number if serial_number else None,
        "bias_trim_present": bias_trim is not None,
        "configuration": _config_dict(cfg),
        "units": _units(cfg),
        "columns": columns,
        "decode_summary": summary,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=pathlib.Path,
                        help="Raw .cap file produced by capture_raw.py")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="Output CSV path (default: capture name with .csv).")
    parser.add_argument("--skip-startup", action="store_true",
                        help="Drop measurements with the gyro startup bit set.")
    args = parser.parse_args(argv)

    csv_path = args.out if args.out is not None else args.capture.with_suffix(".csv")
    summary = decode_capture(args.capture, csv_path, skip_startup=args.skip_startup)

    print("Decoded {0} measurements -> {1}".format(summary["rows"], csv_path))
    print("  resync events: {0}   bytes discarded: {1}   dropped frames: {2}".format(
        summary["resync_events"], summary["bytes_discarded"],
        summary["dropped_frames"]))
    stats = summary["interval_stats"]
    if stats:
        print("  inter-sample interval: mean {0:.3f} ms  min {1:.3f}  "
              "max {2:.3f}  stdev {3:.3f} ms".format(
                  stats["mean_s"] * 1e3, stats["min_s"] * 1e3,
                  stats["max_s"] * 1e3, stats["stdev_s"] * 1e3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
