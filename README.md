# pystim300

Python driver for the Safran STIM300 IMU over RS422.

Implements all three runtime protocols documented in
`ts1524-r31-datasheet-stim300.pdf`:

- **Normal Mode** — binary measurement datagrams (§5.5.6, §7.5.2).
- **Service Mode** — interactive ASCII commands (§7.5.3, §9).
- **Utility Mode** — programmatic ASCII commands with CRC-8 (§7.5.4, §10).

## Install

```bash
pip install pystim300            # codec only (pure stdlib)
pip install pystim300[serial]    # + pyserial-backed SerialTransport
```

Python 3.8+.

## Quick start

```python
from pystim300 import STIM300, SerialTransport

with SerialTransport("/dev/ttyUSB0", bit_rate=1843200) as transport:
    stim = STIM300(transport)
    for measurement in stim.read_measurements():
        print(measurement.gyro, measurement.accel, measurement.status)
```

See `examples/read_normal.py` for a fuller streaming example and
`examples/smoketest.py` for a hardware bring-up script.

## Architecture

Three concentric layers:

1. **Codec** (pure, no I/O) — CRC, scaling, datagram parsers/builders.
2. **Transport** — `Transport` Protocol; `SerialTransport` for real
   hardware, `FakeTransport` for tests.
3. **Client** — `STIM300` owns the transport, mode state, and stream
   parser.

Service- and Utility-Mode exchanges are auditable: every response
dataclass carries the raw bytes that produced it, and the client
accepts an optional `audit` callback that fires on every byte of
ASCII I/O.

Every parser, dataclass, and command carries an inline citation back
to the datasheet (§X.Y / Table Z / p.N) so the implementation can be
checked against the spec line-by-line.

## License

MIT — see `LICENSE`.
