"""Functional-checkout primitives for the STIM300.

These are the STIM300-specific building blocks a downstream hardware test
harness composes to confirm a unit and its UART interface are healthy
before deployment. This module does NOT drive power, drive the test
sequence, or know about the harness's transport - it only provides:

* ``CheckResult`` / ``CheckoutReport`` - the result container types.
* ``ExpectedConfiguration`` - what the device should report (each field
  optional; ``None`` skips the corresponding check).
* ``check_*(...)`` - pure check functions that take pre-collected data
  (Measurements, Configuration, etc.) and return ``CheckResult``.

The downstream harness is responsible for power-on, transport setup, and
the orchestrating sequence; see ``examples/functional_checkout.py`` for
the recommended composition pattern.

Stdlib-only (math + statistics). No numpy.
"""

import json
import math
import statistics
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from pystim300.client import AuditEvent, InitSequence
from pystim300.configuration import Configuration
from pystim300.datagrams import (
    BiasTrimDatagram,
    ExtendedErrorDatagram,
    PartNumberDatagram,
    SerialNumberDatagram,
)
from pystim300.normal import Measurement, NormalStreamParser
from pystim300.service import ServiceResponse
from pystim300.utility import UtilityResponse


# ---------------------------------------------------------------------------
# Result container types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckResult:
    """The outcome of one checkout check.

    ``name`` is a short identifier (e.g. ``"gravity_magnitude"``).
    ``detail`` is one human-readable line. ``measured`` and ``expected``
    carry the raw values when relevant (omitted from the printed summary
    on pass; always included in JSON dumps).
    """

    name: str
    passed: bool
    detail: str
    measured: Any = None
    expected: Any = None


@dataclass(frozen=True)
class CheckoutReport:
    """The complete report from one functional-checkout run."""

    checks: Tuple[CheckResult, ...]
    init_sequence: Optional[InitSequence] = None
    measurement_count: int = 0
    duration_seconds: float = 0.0
    parser_resync_events: int = 0
    parser_bytes_discarded: int = 0
    audit: Tuple[AuditEvent, ...] = ()

    def passed(self) -> bool:
        """True iff every check in the report passed."""
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        """A multi-line, console-ready summary of the run."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("STIM300 functional checkout - {0}".format(
            "PASS" if self.passed() else "FAIL"))
        lines.append("=" * 60)
        if self.init_sequence is not None:
            lines.append("Part Number:   {0} (rev {1})".format(
                self.init_sequence.part_number.part_number,
                self.init_sequence.part_number.revision))
            lines.append("Serial Number: {0}".format(
                self.init_sequence.serial_number.serial_number))
            cfg = self.init_sequence.configuration
            lines.append("Configuration: id=0x{0:02X} rate={1}Hz baud={2} crlf={3}".format(
                cfg.datagram_id, cfg.sample_rate_hz, cfg.bit_rate,
                cfg.crlf_termination))
            lines.append("Clusters:      gyro{0}{1}{2}{3}{4}".format(
                "+accel" if cfg.has_acceleration else "",
                "+incl" if cfg.has_inclination else "",
                "+temp" if cfg.has_temperature else "",
                "+pps" if cfg.has_pps else "",
                " (bias_trim_at_startup)" if cfg.bias_trim_at_startup else ""))
        lines.append("Measurements:  {0} over {1:.3f}s".format(
            self.measurement_count, self.duration_seconds))
        lines.append("Parser stats:  {0} resyncs, {1} bytes discarded".format(
            self.parser_resync_events, self.parser_bytes_discarded))
        lines.append("-" * 60)
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append("[{0}] {1}: {2}".format(mark, c.name, c.detail))
        lines.append("-" * 60)
        passed = sum(1 for c in self.checks if c.passed)
        lines.append("Result: {0}/{1} checks passed".format(passed, len(self.checks)))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict rendering of the report."""
        return {
            "passed": self.passed(),
            "measurement_count": self.measurement_count,
            "duration_seconds": self.duration_seconds,
            "parser_resync_events": self.parser_resync_events,
            "parser_bytes_discarded": self.parser_bytes_discarded,
            "init_sequence": _to_jsonable(self.init_sequence),
            "checks": [_to_jsonable(c) for c in self.checks],
            "audit": [_to_jsonable(e) for e in self.audit],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Render the report as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    if isinstance(obj, (frozenset, set)):
        return sorted(obj)
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError("not JSON serializable: {0}".format(type(obj).__name__))


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert a value to a plain JSON-compatible structure."""
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    if isinstance(obj, (frozenset, set)):
        return sorted(_to_jsonable(x) for x in obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    return str(obj)


# ---------------------------------------------------------------------------
# Expected-configuration container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExpectedConfiguration:
    """Per-check thresholds and expected values.

    Each ``Optional`` field defaults to ``None``, which means the
    corresponding check is skipped. Physics-threshold fields have
    lab-bench defaults that assume a still, level unit; tighten or loosen
    them for the deployment.
    """

    # Identity
    part_number: Optional[str] = None
    serial_numbers: Optional[FrozenSet[str]] = None    # any-of match

    # Configuration cross-check
    sample_rate_hz: Optional[int] = None
    bit_rate: Optional[int] = None
    has_acceleration: Optional[bool] = None
    has_inclination: Optional[bool] = None
    has_temperature: Optional[bool] = None
    has_pps: Optional[bool] = None
    crlf_termination: Optional[bool] = None
    accel_range_g: Optional[Tuple[int, int, int]] = None
    gyro_output_unit: Optional[int] = None
    accel_output_unit: Optional[int] = None
    incl_output_unit: Optional[int] = None
    bias_trim_at_startup: Optional[bool] = None

    # Stream / framing
    max_latency_us: Optional[int] = None
    sample_rate_tolerance_pct: float = 1.0
    startup_frame_allowance: int = 0     # frames whose status.startup bit is allowed

    # Physics (defaults assume a level, still unit at room temperature)
    temp_min_c: float = 0.0
    temp_max_c: float = 50.0
    gyro_max_mean_dps: float = 0.5       # |mean rate| per axis
    gyro_max_std_dps: float = 2.0        # std-dev noise per axis
    gravity_magnitude_g: float = 1.0
    gravity_tolerance_g: float = 0.05
    gravity_direction_std_g: float = 0.01

    # Extended Error
    extended_error_ignore_bits: FrozenSet[int] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Identity & configuration checks
# ---------------------------------------------------------------------------

def _normalize_serial(serial: str) -> str:
    """Canonicalize a STIM300 serial number for comparison.

    The serial number is a fixed 'N' prefix followed by digits
    (Figure 10-2 / Table 5-15). The Init-Mode ``SerialNumberDatagram``
    strips the prefix while the Utility-Mode ``$isn`` response keeps it,
    so the two paths must be normalized before they can be compared.
    Upper-cases, strips surrounding whitespace, and drops a single
    leading 'N'.
    """
    s = serial.strip().upper()
    if s.startswith("N"):
        s = s[1:]
    return s


def check_part_number(pn: PartNumberDatagram,
                       expected: Optional[str]) -> CheckResult:
    if expected is None:
        return CheckResult(
            name="part_number",
            passed=True,
            detail="not checked (no expected value); got {0}".format(pn.part_number),
            measured=pn.part_number,
        )
    ok = pn.part_number == expected
    return CheckResult(
        name="part_number",
        passed=ok,
        detail=("matches {0}".format(expected) if ok
                else "got {0!r}, expected {1!r}".format(pn.part_number, expected)),
        measured=pn.part_number,
        expected=expected,
    )


def check_serial_number(sn: SerialNumberDatagram,
                         allowed: Optional[FrozenSet[str]]) -> CheckResult:
    if not allowed:
        return CheckResult(
            name="serial_number",
            passed=True,
            detail="not checked (no allowed set); got {0}".format(sn.serial_number),
            measured=sn.serial_number,
        )
    # Normalize both sides so an allowed entry may carry the 'N' prefix
    # (as Utility Mode reports it) or omit it (as Init Mode reports it).
    norm_allowed = {_normalize_serial(a) for a in allowed}
    ok = _normalize_serial(sn.serial_number) in norm_allowed
    return CheckResult(
        name="serial_number",
        passed=ok,
        detail=("{0} is in allowed set".format(sn.serial_number) if ok
                else "{0} not in allowed set {1}".format(sn.serial_number, sorted(allowed))),
        measured=sn.serial_number,
        expected=sorted(allowed),
    )


_CONFIG_FIELDS = (
    "sample_rate_hz",
    "bit_rate",
    "has_acceleration",
    "has_inclination",
    "has_temperature",
    "has_pps",
    "crlf_termination",
    "accel_range_g",
    "gyro_output_unit",
    "accel_output_unit",
    "incl_output_unit",
    "bias_trim_at_startup",
)


def check_configuration(cfg: Configuration,
                          expected: ExpectedConfiguration) -> List[CheckResult]:
    """Compare the reported Configuration against every non-None expected field.

    Returns one CheckResult per field that was actually compared.
    """
    results: List[CheckResult] = []
    for field_name in _CONFIG_FIELDS:
        want = getattr(expected, field_name)
        if want is None:
            continue
        got = getattr(cfg, field_name)
        ok = got == want
        results.append(CheckResult(
            name="configuration.{0}".format(field_name),
            passed=ok,
            detail=("{0!r}".format(got) if ok
                    else "got {0!r}, expected {1!r}".format(got, want)),
            measured=got,
            expected=want,
        ))
    return results


def check_bias_trim_present(bt: Optional[BiasTrimDatagram],
                              cfg: Configuration) -> CheckResult:
    """Verify the bias-trim datagram showed up iff the configuration says it should."""
    expected_present = cfg.bias_trim_at_startup
    got_present = bt is not None
    ok = expected_present == got_present
    return CheckResult(
        name="bias_trim_present",
        passed=ok,
        detail=("present={0} matches bias_trim_at_startup={1}".format(
                    got_present, expected_present) if ok
                else "got present={0}, expected {1} (bias_trim_at_startup={1})".format(
                    got_present, expected_present)),
        measured=got_present,
        expected=expected_present,
    )


# ---------------------------------------------------------------------------
# Stream-health checks
# ---------------------------------------------------------------------------

def check_parser_clean(parser: NormalStreamParser) -> CheckResult:
    """``resync_events`` and ``bytes_discarded`` should both be zero."""
    resyncs = parser.resync_events
    discarded = parser.bytes_discarded
    ok = resyncs == 0 and discarded == 0
    return CheckResult(
        name="parser_clean",
        passed=ok,
        detail=("no resyncs, no bytes discarded" if ok
                else "{0} resyncs, {1} bytes discarded".format(resyncs, discarded)),
        measured={"resyncs": resyncs, "discarded": discarded},
        expected={"resyncs": 0, "discarded": 0},
    )


def check_no_dropped_frames(measurements: Sequence[Measurement]) -> CheckResult:
    """Counter must increment by 1 mod 256 between every adjacent pair."""
    if len(measurements) < 2:
        return CheckResult(
            name="no_dropped_frames",
            passed=True,
            detail="not enough measurements to check ({0})".format(len(measurements)),
            measured=len(measurements),
        )
    gaps: List[Tuple[int, int, int]] = []
    for i in range(1, len(measurements)):
        prev = measurements[i - 1].counter
        cur = measurements[i].counter
        delta = (cur - prev) % 256
        if delta != 1:
            gaps.append((i, prev, cur))
    if not gaps:
        return CheckResult(
            name="no_dropped_frames",
            passed=True,
            detail="all {0} frames sequential".format(len(measurements)),
            measured=len(measurements),
        )
    first = gaps[0]
    return CheckResult(
        name="no_dropped_frames",
        passed=False,
        detail="{0} gap(s); first at index {1}: counter {2} -> {3}".format(
            len(gaps), first[0], first[1], first[2]),
        measured=gaps,
    )


def check_frame_rate(measurements: Sequence[Measurement],
                       duration_seconds: float, expected_hz: int,
                       *, tolerance_pct: float = 1.0) -> CheckResult:
    """Achieved rate (len/duration) within ``tolerance_pct`` of ``expected_hz``."""
    if duration_seconds <= 0:
        return CheckResult(
            name="frame_rate",
            passed=False,
            detail="duration_seconds={0} not positive".format(duration_seconds),
            measured=duration_seconds,
            expected=expected_hz,
        )
    actual_hz = len(measurements) / duration_seconds
    tolerance = expected_hz * tolerance_pct / 100.0
    ok = abs(actual_hz - expected_hz) <= tolerance
    return CheckResult(
        name="frame_rate",
        passed=ok,
        detail="{0:.2f} Hz vs {1} Hz (tolerance +/-{2:.2f}%)".format(
            actual_hz, expected_hz, tolerance_pct),
        measured=actual_hz,
        expected=expected_hz,
    )


def check_latency_within(measurements: Sequence[Measurement],
                           max_us: int) -> CheckResult:
    """Every Measurement's ``latency_us`` must be <= ``max_us``."""
    if not measurements:
        return CheckResult(
            name="latency_within",
            passed=True,
            detail="no measurements to check",
            expected=max_us,
        )
    worst = max(m.latency_us for m in measurements)
    ok = worst <= max_us
    return CheckResult(
        name="latency_within",
        passed=ok,
        detail="worst {0} us vs limit {1} us".format(worst, max_us),
        measured=worst,
        expected=max_us,
    )


_FATAL_STATUS_MASK = 0x80 | 0x20 | 0x10 | 0x08
# system_integrity (0x80) | outside_operating_conditions (0x20)
# | overload (0x10) | channel_error (0x08)


def check_status_bytes_clean(measurements: Sequence[Measurement],
                               *, startup_frame_allowance: int = 0) -> CheckResult:
    """No fatal STATUS bits on any cluster.

    The ``startup`` bit (0x40) is permitted on the first
    ``startup_frame_allowance`` frames; after that, any startup bit is a
    failure. All other listed bits are fatal at any time.
    """
    offenders: List[Tuple[int, str, int]] = []
    for idx, m in enumerate(measurements):
        per_cluster = [
            ("gyro", m.gyro_status),
            ("accel", m.accel_status),
            ("incl", m.incl_status),
            ("gyro_temp", m.gyro_temp_status),
            ("accel_temp", m.accel_temp_status),
            ("incl_temp", m.incl_temp_status),
            ("pps", m.pps_status),
        ]
        for cluster_name, status in per_cluster:
            if status is None:
                continue
            if status.raw & _FATAL_STATUS_MASK:
                offenders.append((idx, cluster_name, status.raw))
                continue
            if status.startup and idx >= startup_frame_allowance:
                offenders.append((idx, cluster_name + ".startup", status.raw))
    if not offenders:
        return CheckResult(
            name="status_bytes_clean",
            passed=True,
            detail="{0} frames, no fatal status bits".format(len(measurements)),
            measured=len(measurements),
        )
    first = offenders[0]
    return CheckResult(
        name="status_bytes_clean",
        passed=False,
        detail="{0} offender(s); first at frame {1} {2} status=0x{3:02X}".format(
            len(offenders), first[0], first[1], first[2]),
        measured=offenders[:10],   # cap to keep the report compact
    )


def check_temperature_range(measurements: Sequence[Measurement],
                              min_c: float, max_c: float) -> CheckResult:
    """Every cluster's reported temperature must be inside ``[min_c, max_c]``."""
    triples: List[float] = []
    for m in measurements:
        for trip in (m.gyro_temp, m.accel_temp, m.incl_temp):
            if trip is not None:
                triples.extend(trip)
    if not triples:
        return CheckResult(
            name="temperature_range",
            passed=True,
            detail="no temperature data in stream",
            measured=None,
        )
    lo, hi = min(triples), max(triples)
    ok = min_c <= lo and hi <= max_c
    return CheckResult(
        name="temperature_range",
        passed=ok,
        detail="range [{0:.2f}, {1:.2f}] degC vs allowed [{2:.2f}, {3:.2f}]".format(
            lo, hi, min_c, max_c),
        measured=(lo, hi),
        expected=(min_c, max_c),
    )


def check_gyro_quiescent(measurements: Sequence[Measurement],
                           max_mean_dps: float,
                           max_std_dps: float) -> CheckResult:
    """Per-axis ``|mean|`` and ``std`` of the gyro rate must be within bounds.

    Assumes the gyro output unit is ANGULAR_RATE (or AVERAGE_RATE) - the
    same units as the thresholds. Callers using INCREMENTAL_ANGLE outputs
    should set the thresholds in degrees and treat the check accordingly.
    """
    if len(measurements) < 2:
        return CheckResult(
            name="gyro_quiescent",
            passed=True,
            detail="not enough measurements ({0})".format(len(measurements)),
        )
    xs = [m.gyro[0] for m in measurements]
    ys = [m.gyro[1] for m in measurements]
    zs = [m.gyro[2] for m in measurements]
    means = tuple(statistics.fmean(axis) for axis in (xs, ys, zs))
    stds = tuple(statistics.pstdev(axis) for axis in (xs, ys, zs))
    mean_ok = all(abs(m) <= max_mean_dps for m in means)
    std_ok = all(s <= max_std_dps for s in stds)
    ok = mean_ok and std_ok
    return CheckResult(
        name="gyro_quiescent",
        passed=ok,
        detail=("means {0} <={1}; stds {2} <={3}".format(
                    _fmt3(means), max_mean_dps, _fmt3(stds), max_std_dps)),
        measured={"means": means, "stds": stds},
        expected={"max_mean": max_mean_dps, "max_std": max_std_dps},
    )


def check_gravity_magnitude(measurements: Sequence[Measurement],
                              expected_g: float,
                              tolerance_g: float) -> CheckResult:
    """Mean ``||accel||`` within ``tolerance_g`` of ``expected_g``.

    Skipped (passes with a 'no data' note) if no measurement carries an
    accelerometer cluster.
    """
    mags = [_vec_mag(m.accel) for m in measurements if m.accel is not None]
    if not mags:
        return CheckResult(
            name="gravity_magnitude",
            passed=True,
            detail="no accelerometer data in stream",
        )
    mean_mag = statistics.fmean(mags)
    ok = abs(mean_mag - expected_g) <= tolerance_g
    return CheckResult(
        name="gravity_magnitude",
        passed=ok,
        detail="mean ||a|| = {0:.4f} g (expected {1:.3f} +/-{2:.3f})".format(
            mean_mag, expected_g, tolerance_g),
        measured=mean_mag,
        expected=expected_g,
    )


def check_gravity_direction_consistent(measurements: Sequence[Measurement],
                                          max_std_g: float) -> CheckResult:
    """Per-axis std-dev of the accel vector must be <= ``max_std_g``.

    Detects vibration or a flaky channel. Skipped if no accel data.
    """
    accels = [m.accel for m in measurements if m.accel is not None]
    if len(accels) < 2:
        return CheckResult(
            name="gravity_direction_consistent",
            passed=True,
            detail="not enough accelerometer samples ({0})".format(len(accels)),
        )
    stds = tuple(statistics.pstdev([a[i] for a in accels]) for i in range(3))
    ok = all(s <= max_std_g for s in stds)
    return CheckResult(
        name="gravity_direction_consistent",
        passed=ok,
        detail="per-axis stds {0} <= {1}".format(_fmt3(stds), max_std_g),
        measured=stds,
        expected=max_std_g,
    )


def check_inclinometer_gravity(measurements: Sequence[Measurement],
                                  expected_g: float,
                                  tolerance_g: float) -> CheckResult:
    """Inclinometer-derived ``||a||`` within tolerance of ``expected_g``.

    Skipped if no inclinometer cluster in the stream.
    """
    mags = [_vec_mag(m.incl) for m in measurements if m.incl is not None]
    if not mags:
        return CheckResult(
            name="inclinometer_gravity",
            passed=True,
            detail="no inclinometer data in stream",
        )
    mean_mag = statistics.fmean(mags)
    ok = abs(mean_mag - expected_g) <= tolerance_g
    return CheckResult(
        name="inclinometer_gravity",
        passed=ok,
        detail="mean ||incl|| = {0:.4f} g (expected {1:.3f} +/-{2:.3f})".format(
            mean_mag, expected_g, tolerance_g),
        measured=mean_mag,
        expected=expected_g,
    )


# ---------------------------------------------------------------------------
# Service / Utility round-trip checks (interpret pre-collected audit events)
# ---------------------------------------------------------------------------

def check_service_round_trip(audit_events: Iterable[AuditEvent]) -> CheckResult:
    """Confirm a Service-Mode round-trip completed cleanly.

    The check inspects the slice of audit events captured during the
    round-trip and asserts the basic shape: at least one TX into the
    SERVICEMODE entry and at least one RX containing the
    ``>`` prompt. Detailed protocol validation is already done by
    ``service.parse_response``; this check is the cross-check that the
    bytes actually flowed.
    """
    events = list(audit_events)
    txs = [e for e in events if e.direction == "tx"]
    rxs = [e for e in events if e.direction == "rx"]
    if not txs or not rxs:
        return CheckResult(
            name="service_round_trip",
            passed=False,
            detail="no audit events captured ({0} tx, {1} rx)".format(
                len(txs), len(rxs)),
            measured={"tx": len(txs), "rx": len(rxs)},
        )
    saw_prompt = any(b">" in e.payload for e in rxs)
    if not saw_prompt:
        return CheckResult(
            name="service_round_trip",
            passed=False,
            detail="no Service-Mode prompt '>' seen in {0} rx events".format(len(rxs)),
            measured={"tx": len(txs), "rx": len(rxs)},
        )
    return CheckResult(
        name="service_round_trip",
        passed=True,
        detail="{0} tx / {1} rx, prompt observed".format(len(txs), len(rxs)),
        measured={"tx": len(txs), "rx": len(rxs)},
    )


def check_utility_round_trip(response: UtilityResponse,
                                init_serial: SerialNumberDatagram) -> CheckResult:
    """Validate a Utility-Mode ``$isn`` response cross-checks the Init-Mode serial.

    Pass the parsed ``UtilityResponse`` from an ``isn`` command; the
    function asserts status==0 and that the returned serial-number field
    matches the one captured during Init-Mode. ``parse_response`` will
    already have raised on CRC failure - this check simply confirms the
    two paths agree.
    """
    if response.status != 0:
        return CheckResult(
            name="utility_round_trip",
            passed=False,
            detail="$isn status={0} (expected 0)".format(response.status),
            measured=response.status,
            expected=0,
        )
    if not response.fields:
        return CheckResult(
            name="utility_round_trip",
            passed=False,
            detail="$isn returned no serial field",
            measured=response.fields,
        )
    reported = response.fields[0].strip()
    expected = init_serial.serial_number
    # Utility Mode reports the serial with its fixed 'N' prefix; the
    # Init-Mode datagram strips it. Normalize before comparing.
    ok = _normalize_serial(reported) == _normalize_serial(expected)
    return CheckResult(
        name="utility_round_trip",
        passed=ok,
        detail=("$isn={0} matches Init-Mode serial".format(reported) if ok
                else "$isn={0} != Init-Mode serial {1}".format(reported, expected)),
        measured=reported,
        expected=expected,
    )


# ---------------------------------------------------------------------------
# Extended Error
# ---------------------------------------------------------------------------

def check_extended_error_clean(eed: Optional[ExtendedErrorDatagram],
                                 *, ignore_bits: FrozenSet[int] = frozenset()
                                 ) -> CheckResult:
    """No Extended Error bits set, modulo an optional ignore-list."""
    if eed is None:
        return CheckResult(
            name="extended_error_clean",
            passed=False,
            detail="no Extended Error datagram captured",
        )
    mask = 0
    for bit in ignore_bits:
        mask |= (1 << bit)
    residual = eed.error_bits & ~mask
    ok = residual == 0
    return CheckResult(
        name="extended_error_clean",
        passed=ok,
        detail=("all error bits clear (flags={0})".format(sorted(eed.flags)) if ok
                else "residual error bits = 0x{0:032x} (flags={1})".format(
                    residual, sorted(eed.flags))),
        measured=eed.error_bits,
        expected=0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec_mag(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _fmt3(v: Tuple[float, float, float]) -> str:
    return "({0:+.4f}, {1:+.4f}, {2:+.4f})".format(*v)
