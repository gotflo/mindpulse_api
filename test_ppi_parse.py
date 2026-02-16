"""
Analyse precise du format PPI du Polar Verity Sense.
On va tester tous les formats de parsing possibles.
"""

import asyncio
import struct
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from bleak import BleakClient, BleakScanner

PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA    = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"

all_ppi = []
all_hr = []

HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


def handle_hr(sender, data: bytearray):
    flags = data[0]
    hr = struct.unpack_from("<H" if flags & 0x01 else "<B", data, 1)[0]
    all_hr.append(hr)


def handle_pmd_data(sender, data: bytearray):
    """
    Format PMD PPI Polar SDK (d'apres la doc Polar):
    Byte 0: type de mesure (0x03 = PPI)
    Bytes 1-8: timestamp (8 bytes, uint64 little-endian, en nanoseconds)
    Byte 9: frame type (0x00 = raw)
    Bytes 10+: samples PPI, chaque sample = 6 bytes:
        - byte 0: HR (uint8)
        - bytes 1-2: PP interval (uint16 LE, ms)
        - bytes 3-4: error estimate (uint16 LE, ms)
        - byte 5: flags (bit0=skin_contact, bit1=contact_supported,
                         bit2=rr_not_from_pp, bit3=..?)

    MAIS les valeurs semblent decalees. Testons aussi le format:
        - bytes 0-1: PP interval (uint16 LE, ms)
        - bytes 2-3: error estimate (uint16 LE, ms)
        - byte 4: HR (uint8)
        - byte 5: flags
    """
    if len(data) < 10:
        return

    meas_type = data[0]
    if meas_type != 0x03:
        return

    timestamp = struct.unpack_from("<Q", data, 1)[0]
    frame_type = data[9]

    raw = data[10:]
    n_bytes = len(raw)

    print(f"\n  --- Packet: {n_bytes} data bytes, timestamp={timestamp}, frame={frame_type} ---")
    print(f"  Raw hex: {raw.hex()}")
    print(f"  Raw bytes: {list(raw)}")

    # === FORMAT A: Polar SDK officiel ===
    # HR(1) + PPI(2) + errEst(2) + flags(1) = 6 bytes
    if n_bytes % 6 == 0:
        n = n_bytes // 6
        print(f"\n  FORMAT A (HR,PPI,err,flags) - {n} samples:")
        for i in range(n):
            o = i * 6
            hr = raw[o]
            ppi = struct.unpack_from("<H", raw, o + 1)[0]
            err = struct.unpack_from("<H", raw, o + 3)[0]
            flags = raw[o + 5]
            skin = "Y" if flags & 0x01 else "N"
            supp = "Y" if flags & 0x02 else "N"
            print(f"    [{i}] HR={hr:3d} PPI={ppi:4d}ms err={err:3d}ms skin={skin} supp={supp} flags=0b{flags:08b}")
            if 250 < ppi < 2000:
                all_ppi.append(ppi)

    # === FORMAT B: PPI(2) + errEst(2) + HR(1) + flags(1) = 6 bytes ===
    if n_bytes % 6 == 0:
        n = n_bytes // 6
        print(f"\n  FORMAT B (PPI,err,HR,flags) - {n} samples:")
        for i in range(n):
            o = i * 6
            ppi = struct.unpack_from("<H", raw, o)[0]
            err = struct.unpack_from("<H", raw, o + 2)[0]
            hr = raw[o + 4]
            flags = raw[o + 5]
            skin = "Y" if flags & 0x01 else "N"
            print(f"    [{i}] PPI={ppi:4d}ms err={err:3d}ms HR={hr:3d} skin={skin} flags=0b{flags:08b}")

    # === FORMAT C: flags(1) + PPI(2) + errEst(2) + HR(1) = 6 bytes ===
    if n_bytes % 6 == 0:
        n = n_bytes // 6
        print(f"\n  FORMAT C (flags,PPI,err,HR) - {n} samples:")
        for i in range(n):
            o = i * 6
            flags = raw[o]
            ppi = struct.unpack_from("<H", raw, o + 1)[0]
            err = struct.unpack_from("<H", raw, o + 3)[0]
            hr = raw[o + 5]
            skin = "Y" if flags & 0x01 else "N"
            print(f"    [{i}] PPI={ppi:4d}ms err={err:3d}ms HR={hr:3d} skin={skin} flags=0b{flags:08b}")

    # === Try 5-byte samples too ===
    if n_bytes % 5 == 0:
        n = n_bytes // 5
        print(f"\n  FORMAT D (5-byte: PPI(2),err(2),flags(1)) - {n} samples:")
        for i in range(n):
            o = i * 5
            ppi = struct.unpack_from("<H", raw, o)[0]
            err = struct.unpack_from("<H", raw, o + 2)[0]
            flags = raw[o + 4]
            print(f"    [{i}] PPI={ppi:4d}ms err={err:3d}ms flags=0b{flags:08b}")


def handle_pmd_control(sender, data: bytearray):
    if len(data) > 0 and data[0] == 0xF0:
        status = data[3] if len(data) > 3 else -1
        status_str = "OK" if status == 0 else f"ERR({status})"
        print(f"  [CTRL] Response status: {status_str}")
    else:
        print(f"  [CTRL] {data.hex()}")


async def main():
    print("[SCAN]...")
    devices = await BleakScanner.discover(timeout=10.0)
    polar = next((d for d in devices if d.name and "polar" in d.name.lower() and "sense" in d.name.lower()), None)

    if not polar:
        print("Polar Sense non trouve")
        return

    print(f"[OK] {polar.name}\n")

    async with BleakClient(polar) as client:
        # Subscribe
        await client.start_notify(PMD_CONTROL, handle_pmd_control)
        await client.start_notify(PMD_DATA, handle_pmd_data)
        await client.start_notify(HR_UUID, handle_hr)
        await asyncio.sleep(1)

        # Start PPI
        print("[START PPI]")
        await client.write_gatt_char(PMD_CONTROL, bytearray([0x02, 0x03]), response=True)
        await asyncio.sleep(1)

        # Collect 30 seconds
        print("[COLLECTING 30 seconds...]")
        await asyncio.sleep(30)

        # Stop
        await client.write_gatt_char(PMD_CONTROL, bytearray([0x03, 0x03]), response=True)
        await asyncio.sleep(1)

        await client.stop_notify(PMD_DATA)
        await client.stop_notify(PMD_CONTROL)
        await client.stop_notify(HR_UUID)

        print(f"\n{'='*60}")
        print(f"RESUME: {len(all_ppi)} PPI valides collectes")
        if all_ppi:
            import numpy as np
            arr = np.array(all_ppi)
            print(f"  PPI mean={arr.mean():.0f}ms, std={arr.std():.0f}ms")
            print(f"  PPI min={arr.min()}ms, max={arr.max()}ms")
            print(f"  HR estim = {60000/arr.mean():.0f} bpm")
        if all_hr:
            print(f"  HR direct mean = {sum(all_hr)/len(all_hr):.0f} bpm")

    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
