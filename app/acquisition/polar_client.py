"""
Polar Verity Sense BLE client.

Handles connection, data streaming (HR via standard service, PPI via
PMD SDK protocol), battery level, and signal quality monitoring via bleak.

PMD (Polar Measurement Data) protocol:
  - Control point (fb005c81): write commands, receive indications
  - Data channel  (fb005c82): receive PPI notification frames

PPI frame format (verified on Polar Sense EC6DBD29):
  Header: 10 bytes
    - byte 0:   measurement type (0x03 = PPI)
    - bytes 1-8: timestamp (uint64 LE, nanoseconds) — always 0 on Verity Sense
    - byte 9:   frame type (0x00 = raw)
  Samples: N × 6 bytes each
    - byte 0:   HR (uint8) — always 0, HR comes from HR service
    - bytes 1-2: PPI interval (uint16 LE, milliseconds)
    - bytes 3-4: error estimate (uint16 LE, milliseconds)
    - byte 5:   flags (bit0=skin_contact, bit1=contact_supported, bit2=rr_interval_valid)
"""

import asyncio
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice

from app.config.settings import BLEConfig

logger = logging.getLogger(__name__)

# PMD protocol constants
PMD_CONTROL_UUID = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA_UUID = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"
PMD_CMD_START = 0x02
PMD_CMD_STOP = 0x03
PMD_TYPE_PPI = 0x03
PMD_HEADER_SIZE = 10
PMD_SAMPLE_SIZE = 6


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    SCANNING = "scanning"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass
class PolarSample:
    timestamp: float
    hr: int
    ppi_ms: list[int] = field(default_factory=list)
    ppi_errors_ms: list[int] = field(default_factory=list)
    rr_quality: list[bool] = field(default_factory=list)


@dataclass
class DeviceInfo:
    name: str = ""
    address: str = ""
    battery_level: int = -1
    signal_quality: float = 0.0
    connection_state: ConnectionState = ConnectionState.DISCONNECTED


class PolarClient:
    def __init__(self, config: BLEConfig):
        self._config = config
        self._client: Optional[BleakClient] = None
        self._device: Optional[BLEDevice] = None
        self._info = DeviceInfo()
        self._on_sample: Optional[Callable[[PolarSample], None]] = None
        self._on_state_change: Optional[Callable[[DeviceInfo], None]] = None
        self._on_unexpected_disconnect: Optional[Callable[[], None]] = None
        self._running = False
        self._quality_window: list[bool] = []
        self._quality_window_size = 50
        self._pmd_streaming = False

    @property
    def info(self) -> DeviceInfo:
        return self._info

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    def on_sample(self, callback: Callable[[PolarSample], None]):
        self._on_sample = callback

    def on_state_change(self, callback: Callable[[DeviceInfo], None]):
        self._on_state_change = callback

    def on_unexpected_disconnect(self, callback: Callable[[], None]):
        self._on_unexpected_disconnect = callback

    def _set_state(self, state: ConnectionState):
        self._info.connection_state = state
        if self._on_state_change:
            self._on_state_change(self._info)

    async def scan(self) -> Optional[BLEDevice]:
        self._set_state(ConnectionState.SCANNING)
        logger.info("Scanning for %s...", self._config.device_name)
        try:
            devices = await BleakScanner.discover(
                timeout=self._config.scan_timeout
            )
            for d in devices:
                if d.name and self._config.device_name.lower() in d.name.lower():
                    logger.info("Found device: %s [%s]", d.name, d.address)
                    self._device = d
                    self._info.name = d.name or ""
                    self._info.address = d.address
                    return d
            logger.warning("Device not found")
            self._set_state(ConnectionState.DISCONNECTED)
            return None
        except Exception as e:
            logger.error("Scan error: %s", e)
            self._set_state(ConnectionState.ERROR)
            return None

    async def connect(self) -> bool:
        if self._device is None:
            device = await self.scan()
            if device is None:
                return False

        self._set_state(ConnectionState.CONNECTING)
        for attempt in range(1, self._config.reconnect_attempts + 1):
            try:
                self._client = BleakClient(
                    self._device,
                    disconnected_callback=self._on_disconnect,
                )
                await self._client.connect()
                if self._client.is_connected:
                    logger.info("Connected to %s", self._info.name)
                    self._set_state(ConnectionState.CONNECTED)
                    await self._read_battery()
                    return True
            except Exception as e:
                logger.warning("Connection attempt %d failed: %s", attempt, e)
                await asyncio.sleep(self._config.reconnect_delay)

        self._set_state(ConnectionState.ERROR)
        return False

    async def start_streaming(self):
        """Start HR notifications and PPI via PMD SDK protocol."""
        if not self.is_connected:
            raise RuntimeError("Not connected to device")

        self._running = True
        self._set_state(ConnectionState.STREAMING)

        # 1. Start HR standard service notifications
        await self._client.start_notify(
            self._config.hr_uuid, self._handle_hr_data
        )
        logger.info("HR notifications started")

        # 2. Subscribe to PMD control point (indications) and data channel (notifications)
        await self._client.start_notify(
            PMD_CONTROL_UUID, self._handle_pmd_control
        )
        await self._client.start_notify(
            PMD_DATA_UUID, self._handle_pmd_data
        )
        logger.info("PMD subscriptions active")

        # 3. Send START PPI command via PMD control point
        start_cmd = bytearray([PMD_CMD_START, PMD_TYPE_PPI])
        await self._client.write_gatt_char(PMD_CONTROL_UUID, start_cmd, response=True)
        self._pmd_streaming = True
        logger.info("PPI streaming started via PMD SDK")

    async def stop_streaming(self):
        self._running = False

        if self.is_connected:
            # Stop PPI via PMD
            if self._pmd_streaming:
                try:
                    stop_cmd = bytearray([PMD_CMD_STOP, PMD_TYPE_PPI])
                    await self._client.write_gatt_char(
                        PMD_CONTROL_UUID, stop_cmd, response=True
                    )
                    self._pmd_streaming = False
                    logger.info("PPI streaming stopped")
                except Exception as e:
                    logger.warning("Error stopping PPI stream: %s", e)

            # Unsubscribe notifications
            for uuid in [self._config.hr_uuid, PMD_CONTROL_UUID, PMD_DATA_UUID]:
                try:
                    await self._client.stop_notify(uuid)
                except Exception as e:
                    logger.warning("Error stopping notify on %s: %s", uuid, e)

        self._set_state(ConnectionState.CONNECTED)
        logger.info("Streaming stopped")

    async def disconnect(self):
        self._running = False
        if self._client and self._client.is_connected:
            try:
                if self._pmd_streaming:
                    stop_cmd = bytearray([PMD_CMD_STOP, PMD_TYPE_PPI])
                    await self._client.write_gatt_char(
                        PMD_CONTROL_UUID, stop_cmd, response=True
                    )
                    self._pmd_streaming = False
            except Exception:
                pass
            try:
                await self._client.disconnect()
            except Exception as e:
                logger.warning("Disconnect error: %s", e)
        self._client = None
        self._set_state(ConnectionState.DISCONNECTED)
        logger.info("Disconnected")

    async def _read_battery(self):
        try:
            data = await self._client.read_gatt_char(self._config.battery_uuid)
            self._info.battery_level = data[0]
            logger.info("Battery level: %d%%", self._info.battery_level)
        except Exception as e:
            logger.warning("Could not read battery: %s", e)

    # ─── HR Standard Service ───

    def _handle_hr_data(self, _sender, data: bytearray):
        flags = data[0]
        hr_format_16bit = flags & 0x01
        hr = struct.unpack_from("<H" if hr_format_16bit else "<B", data, 1)[0]

        sample = PolarSample(timestamp=time.time(), hr=hr)
        if self._on_sample:
            self._on_sample(sample)

    # ─── PMD SDK (PPI) ───

    def _handle_pmd_control(self, _sender, data: bytearray):
        """Handle PMD control point indications (command responses)."""
        if len(data) >= 4 and data[0] == 0xF0:
            cmd = data[1]
            mtype = data[2]
            status = data[3]
            cmd_name = {0x01: "GET_SETTINGS", 0x02: "START", 0x03: "STOP"}.get(cmd, f"CMD({cmd})")
            status_name = "OK" if status == 0 else f"ERROR({status})"
            logger.info("PMD response: %s PPI -> %s", cmd_name, status_name)

            if cmd == PMD_CMD_START and status != 0:
                logger.error("PMD START PPI failed with status %d", status)

    def _handle_pmd_data(self, _sender, data: bytearray):
        """Parse PMD PPI data frame.

        Frame layout:
          [0]    : measurement type (0x03 = PPI)
          [1-8]  : timestamp (uint64 LE) — 0 on Verity Sense
          [9]    : frame type (0x00 = raw)
          [10+]  : N × 6-byte samples:
                     [0]   HR (uint8, always 0)
                     [1-2] PPI (uint16 LE, ms)
                     [3-4] error estimate (uint16 LE, ms)
                     [5]   flags (bit0=skin_contact, bit1=contact_supported)
        """
        if len(data) < PMD_HEADER_SIZE + PMD_SAMPLE_SIZE:
            return

        meas_type = data[0]
        if meas_type != PMD_TYPE_PPI:
            return

        timestamp = time.time()
        raw = data[PMD_HEADER_SIZE:]
        ppis = []
        errors = []
        qualities = []

        index = 0
        while index + PMD_SAMPLE_SIZE <= len(raw):
            _hr = raw[index]
            ppi = struct.unpack_from("<H", raw, index + 1)[0]
            err = struct.unpack_from("<H", raw, index + 3)[0]
            flags = raw[index + 5]

            # bit 1 (contact_supported) AND bit 0 (skin_contact)
            # On Verity Sense: flags 0x07 = good contact, 0x06 = supported but no contact
            skin_contact = bool(flags & 0x01)

            ppis.append(ppi)
            errors.append(err)
            qualities.append(skin_contact)
            index += PMD_SAMPLE_SIZE

        if not ppis:
            return

        self._update_signal_quality(qualities)

        sample = PolarSample(
            timestamp=timestamp,
            hr=0,  # HR comes from HR service, not PPI
            ppi_ms=ppis,
            ppi_errors_ms=errors,
            rr_quality=qualities,
        )

        logger.debug(
            "PPI frame: %d samples, values=%s, errors=%s",
            len(ppis), ppis, errors
        )

        if self._on_sample:
            self._on_sample(sample)

    def _update_signal_quality(self, qualities: list[bool]):
        self._quality_window.extend(qualities)
        if len(self._quality_window) > self._quality_window_size:
            self._quality_window = self._quality_window[-self._quality_window_size:]
        if self._quality_window:
            self._info.signal_quality = sum(self._quality_window) / len(
                self._quality_window
            )

    def _on_disconnect(self, _client):
        logger.warning("Device disconnected unexpectedly")
        was_streaming = self._running
        self._pmd_streaming = False
        self._set_state(ConnectionState.DISCONNECTED)
        self._running = False
        if was_streaming and self._on_unexpected_disconnect:
            self._on_unexpected_disconnect()
