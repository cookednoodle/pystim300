"""Tests for the STATUS byte decoder (Table 5-23, p.38)."""

import pytest

from pystim300.status import StatusByte


class TestStatusByte:
    def test_all_zero_is_ok(self):
        s = StatusByte.decode(0x00)
        assert s.ok()
        assert not s.system_integrity_error
        assert not s.startup
        assert not s.outside_operating_conditions
        assert not s.overload
        assert not s.channel_error
        assert s.axes() == (False, False, False)

    def test_each_bit(self):
        cases = [
            (0x80, "system_integrity_error"),
            (0x40, "startup"),
            (0x20, "outside_operating_conditions"),
            (0x10, "overload"),
            (0x08, "channel_error"),
            (0x04, "axis_z"),
            (0x02, "axis_y"),
            (0x01, "axis_x"),
        ]
        for value, attr in cases:
            s = StatusByte.decode(value)
            assert getattr(s, attr), "expected {0} to be set for byte 0x{1:02x}".format(attr, value)
            assert not s.ok()
            assert s.raw == value

    def test_overload_z_axis(self):
        # Datasheet §7.6 example: overload on Z = bit 4 + bit 2 set.
        s = StatusByte.decode(0x14)
        assert s.overload
        assert s.axes() == (False, False, True)

    def test_overload_x_and_y(self):
        # Compound axis flags.
        s = StatusByte.decode(0x13)  # 0x10 overload + 0x02 + 0x01
        assert s.overload
        assert s.axes() == (True, True, False)

    def test_axes_helper_returns_xyz_order(self):
        s = StatusByte.decode(0x01)
        assert s.axes() == (True, False, False)
        s = StatusByte.decode(0x02)
        assert s.axes() == (False, True, False)
        s = StatusByte.decode(0x04)
        assert s.axes() == (False, False, True)

    @pytest.mark.parametrize("bad", [-1, 256, 1000])
    def test_invalid_byte_value_raises(self, bad):
        with pytest.raises(ValueError):
            StatusByte.decode(bad)

    def test_round_trip_all_bytes(self):
        # Decoding every possible byte value should preserve `raw`.
        for v in range(256):
            assert StatusByte.decode(v).raw == v
