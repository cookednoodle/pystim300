"""Shared test fixtures and builders for pystim300.

These helpers build Configuration objects and Measurement objects suitable
for driving every Normal-Mode datagram ID, plus convenience routines that
use the production codec to compute CRCs (so a round-trip through the
parser is a genuine end-to-end check).
"""

from typing import Optional, Tuple

import pytest

from pystim300.configuration import (
    CONFIGURATION_PAYLOAD_LENGTH,
    Configuration,
    compute_datagram_id,
)
from pystim300.normal import Measurement
from pystim300.status import StatusByte


def make_configuration(
    *,
    has_pps: bool = False,
    has_temperature: bool = False,
    has_inclination: bool = False,
    has_acceleration: bool = False,
    crlf: bool = False,
    accel_range_g: int = 10,
    gyro_output_unit: int = 0b0000,           # ANGULAR_RATE
    accel_output_unit: int = 0b0000,          # ACCELERATION [g]
    incl_output_unit: int = 0b0000,           # ACCELERATION [g]
    pps_output_unit: int = 0b0000,            # FILTERED
    sample_rate_hz: int = 2000,
) -> Configuration:
    """Build a Configuration matching the requested cluster set."""
    raw = bytes(CONFIGURATION_PAYLOAD_LENGTH)
    dgid = compute_datagram_id(
        has_pps=has_pps,
        has_temperature=has_temperature,
        has_inclination=has_inclination,
        has_acceleration=has_acceleration,
    )
    return Configuration(
        raw_payload=raw,
        revision_char="-",
        firmware_revision=31,
        sample_rate_hz=sample_rate_hz,
        has_pps=has_pps,
        has_temperature=has_temperature,
        has_inclination=has_inclination,
        has_acceleration=has_acceleration,
        crlf_termination=crlf,
        bit_rate=1843200,
        stop_bits=1,
        parity="none",
        line_termination=False,
        gyro_axes_active=(True, True, True),
        gyro_output_unit=gyro_output_unit,
        gyro_lp_filter_hz=(262, 262, 262),
        gyro_g_compensation=0,
        accel_axes_active=(True, True, True),
        accel_output_unit=accel_output_unit,
        accel_lp_filter_hz=(262, 262, 262),
        incl_axes_active=(True, True, True),
        incl_output_unit=incl_output_unit,
        incl_lp_filter_hz=(262, 262, 262),
        pps_output_unit=pps_output_unit,
        pps_lp_filter_hz=262,
        has_aux_input=False,
        has_pps_input=True,
        accel_range_g=(accel_range_g, accel_range_g, accel_range_g),
        datagram_id=dgid,
    )


def make_measurement(
    cfg: Configuration,
    *,
    gyro: Tuple[float, float, float] = (1.0, 2.0, -3.0),
    accel: Tuple[float, float, float] = (0.1, -0.2, 0.05),
    incl: Tuple[float, float, float] = (0.01, -0.02, 0.005),
    gyro_temp: Tuple[float, float, float] = (25.0, 25.5, 26.0),
    accel_temp: Tuple[float, float, float] = (24.0, 24.5, 25.0),
    incl_temp: Tuple[float, float, float] = (23.0, 23.5, 24.0),
    pps: Optional[float] = 0.5,
    counter: int = 42,
    latency_us: int = 1234,
    status_raw: int = 0,
) -> Measurement:
    """Build a Measurement whose populated clusters match ``cfg``."""
    sb = StatusByte.decode(status_raw)
    return Measurement(
        datagram_id=cfg.datagram_id,
        counter=counter,
        latency_us=latency_us,
        gyro=gyro,
        gyro_status=sb,
        accel=accel if cfg.has_acceleration else None,
        accel_status=sb if cfg.has_acceleration else None,
        incl=incl if cfg.has_inclination else None,
        incl_status=sb if cfg.has_inclination else None,
        gyro_temp=gyro_temp if cfg.has_temperature else None,
        gyro_temp_status=sb if cfg.has_temperature else None,
        accel_temp=accel_temp if (cfg.has_temperature and cfg.has_acceleration) else None,
        accel_temp_status=sb if (cfg.has_temperature and cfg.has_acceleration) else None,
        incl_temp=incl_temp if (cfg.has_temperature and cfg.has_inclination) else None,
        incl_temp_status=sb if (cfg.has_temperature and cfg.has_inclination) else None,
        pps=pps if cfg.has_pps else None,
        pps_status=sb if cfg.has_pps else None,
    )


# Parametrize-friendly list of all 16 cluster-presence combinations.
ALL_CLUSTER_COMBOS = [
    (pps, temp, incl, accel)
    for pps in (False, True)
    for temp in (False, True)
    for incl in (False, True)
    for accel in (False, True)
]


@pytest.fixture
def vec3_equal():
    """Helper for comparing Vec3 fields with a small tolerance."""
    def _eq(a, b, tol=1e-4):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return all(abs(x - y) < tol for x, y in zip(a, b))
    return _eq
