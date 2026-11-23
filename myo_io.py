"""
EMG sources with one common interface, so the rest of the pipeline doesn't care
whether data comes from the real armband or from a recording.

Every source calls on_sample(t, x) once per EMG sample:
    t : timestamp in seconds (epoch time, high resolution, see clock())
    x : np.ndarray of shape (8,), raw EMG values (int8 range, as float32)

Sources:
    MyoBLE       - the real Myo over Bluetooth LE (bleak)
    ReplayStream - plays back a recorded (N, 8) array at 200 Hz
"""
import asyncio
import time

import numpy as np

FS = 200  # Myo EMG sampling rate (Hz)

_T0_EPOCH = time.time()
_T0_PC = time.perf_counter()


def clock():
    """Epoch seconds with perf_counter precision. Use the same clock for every device
    (EMG now, ultrasound later) so recordings can be aligned afterwards."""
    return _T0_EPOCH + (time.perf_counter() - _T0_PC)


class MyoBLE:
    CONTROL = "d5060401-a904-deb9-4748-2c7f4a124842"
    EMG_CHARS = [
        "d5060105-a904-deb9-4748-2c7f4a124842",
        "d5060205-a904-deb9-4748-2c7f4a124842",
        "d5060305-a904-deb9-4748-2c7f4a124842",
        "d5060405-a904-deb9-4748-2c7f4a124842",
    ]

    def __init__(self, name="Myo"):
        self.name = name
        self.device = None
        self.client = None
        self._on_sample = None

    async def connect(self):
        from bleak import BleakClient, BleakScanner
        while self.device is None:
            print("Scanning for Myo...")
            devices = await BleakScanner.discover(timeout=5.0)
            for d in devices:
                if d.name:
                    print("  found:", d.name, d.address)
            self.device = next((d for d in devices
                                if d.name and self.name.lower() in d.name.lower()), None)
            if self.device is None:
                print("Myo not found. Charged, awake, not connected elsewhere? Retrying...\n")
        self.client = BleakClient(self.device,
                                  disconnected_callback=lambda _: print("Myo disconnected!"))
        await self.client.connect()
        print("Connected to", self.device.name)
        await self.client.write_gatt_char(self.CONTROL, bytearray([0x09, 0x01, 0x01]))              # never sleep
        await self.client.write_gatt_char(self.CONTROL, bytearray([0x01, 0x03, 0x02, 0x00, 0x00]))  # EMG on

    def _handler(self, _, payload):
        t = clock()
        v = np.frombuffer(bytes(payload[:16]), dtype=np.int8).astype(np.float32)
        # One notification carries 2 consecutive samples; the first is one sample period older.
        # Arrival times jitter (Bluetooth batches packets), so for precise alignment
        # combine these timestamps with the sample index (index / FS).
        self._on_sample(t - 1.0 / FS, v[:8])
        self._on_sample(t, v[8:])

    async def start(self, on_sample):
        self._on_sample = on_sample
        for c in self.EMG_CHARS:
            await self.client.start_notify(c, self._handler)

    async def stop(self):
        for c in self.EMG_CHARS:
            await self.client.stop_notify(c)

    @property
    def connected(self):
        return self.client is not None and self.client.is_connected

    async def disconnect(self):
        if self.connected:
            await self.client.disconnect()


class ReplayStream:
    """Plays back a recorded EMG array (N, 8) at FS Hz through the same interface as MyoBLE."""

    def __init__(self, emg, speed=1.0, chunk=2):
        self.emg = np.asarray(emg, dtype=np.float32)
        self.speed = speed
        self.chunk = chunk      # samples per tick (2 = like a Myo notification)
        self.pos = 0            # index of the next sample to play
        self._task = None
        self._done = False

    async def connect(self):
        print(f"Replay: {len(self.emg)} samples ({len(self.emg) / FS:.1f} s) at {self.speed}x speed")

    async def _run(self, on_sample):
        dt = self.chunk / FS / self.speed
        next_tick = time.perf_counter()
        while self.pos < len(self.emg):
            for _ in range(self.chunk):
                if self.pos >= len(self.emg):
                    break
                on_sample(clock(), self.emg[self.pos])
                self.pos += 1
            next_tick += dt
            await asyncio.sleep(max(0.0, next_tick - time.perf_counter()))
        self._done = True

    async def start(self, on_sample):
        self._done = False
        self._task = asyncio.create_task(self._run(on_sample))

    async def stop(self):
        if self._task:
            self._task.cancel()

    @property
    def connected(self):
        return not self._done

    async def disconnect(self):
        await self.stop()