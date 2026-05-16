# pystim300

Python driver for the Safran STIM300 IMU over RS422.

Implements all three runtime protocols documented in
`ts1524-r31-datasheet-stim300.pdf`:

- **Normal Mode** — binary measurement datagrams (§5.5.6, §7.5.2).
- **Service Mode** — interactive ASCII commands (§7.5.3, §9).
- **Utility Mode** — programmatic ASCII commands with CRC-8 (§7.5.4, §10).

Pure Python, no required runtime dependencies. Python 3.8+. MIT licensed.

## Install

```bash
pip install pystim300            # codec only (pure stdlib)
pip install pystim300[serial]    # + pyserial-backed SerialTransport
```

## Quick start

```python
from pystim300 import STIM300, SerialTransport
from pystim300.configuration import Configuration, compute_datagram_id

cfg = Configuration(...)  # see examples/read_normal.py for a full example

with SerialTransport("/dev/ttyUSB0", bit_rate=1843200) as transport:
    stim = STIM300(transport, configuration=cfg)
    for m in stim.read_measurements(limit=10):
        print(m.counter, m.gyro, m.accel, m.gyro_status)
```

See `examples/read_normal.py` for a complete streaming example and
`examples/smoketest.py` for a hardware bring-up script that reports
achieved frame rate, dropped frames, and STATUS-byte flags.

## Service and Utility modes

```python
# Service Mode: human-readable ASCII commands
banner = stim.enter_service()
print(banner.lines)
resp = stim.service_command("i m")          # query sample rate
stim.service_command("m 4")                  # set sample rate to 2000 Hz
stim.service_command("s")                    # save to flash
stim.exit_service()                          # back to Normal Mode

# Utility Mode: machine-to-machine commands with CRC-8
ack = stim.enter_utility()
resp = stim.utility_command("isn")           # read serial number
print(resp.fields)                            # e.g. ("N1234567890ABCD",)
stim.utility_command("sm", 4)                # set sample rate to 2000 Hz
stim.utility_command("save")                  # persist to flash
stim.exit_utility()
```

Every Service/Utility response carries the raw wire bytes on
`response.raw` for auditing. For full session-level auditing pass a
callback:

```python
from pystim300 import MemoryAuditor

auditor = MemoryAuditor()
stim = STIM300(transport, audit=auditor)
# ... do work ...
for event in auditor.events:
    print(event.mode, event.direction, event.payload)
```

## Architecture

Three layers, each individually testable:

1. **Codec** (`crc`, `scaling`, `status`, `configuration`, `datagrams`,
   `normal`, `service`, `utility`) — pure functions and dataclasses;
   zero I/O, zero runtime deps.
2. **Transport** (`transport`) — `Transport` Protocol with `read`,
   `write`, `close`. `SerialTransport` for real hardware (pyserial in
   the `[serial]` extra), `FakeTransport` for hardware-free tests with
   optional scripted request/response pairs.
3. **Client** (`client`) — `STIM300` owns the transport, tracks the
   active `Mode`, runs the I/O loops, and fires `AuditEvent` records
   for Service/Utility byte exchanges.

Every parser, dataclass, command, and bit-field carries an inline
citation back to the datasheet (`§X.Y / Table Z / p.N`) so the
implementation can be checked against the spec line-by-line.

## Tests

```bash
pip install pytest
python -m pytest
```

All 300+ tests run against `FakeTransport` — no hardware needed for
CI. The CRC-8 implementation is verified against the worked examples
in the datasheet (Figure 10-1 and 10-2 give known CRC values for
known strings). The CRC-32 implementation is cross-checked against
a bit-banged reference and optionally against `crcmod` (dev-only).

## License

MIT — see `LICENSE`.
