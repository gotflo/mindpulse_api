"""
Test : extraire HR + RR intervals depuis le service Heart Rate standard.
Le flag bit 4 du HR measurement indique la presence d'intervalles RR.
"""

import asyncio
import struct
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from bleak import BleakClient, BleakScanner

HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# PMD (Polar Measurement Data)
PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"


def parse_hr_measurement(data: bytearray):
    """Parse complet du Heart Rate Measurement selon le standard BLE."""
    flags = data[0]
    hr_16bit = flags & 0x01
    contact_detected = (flags >> 1) & 0x01
    contact_supported = (flags >> 2) & 0x01
    energy_present = (flags >> 3) & 0x01
    rr_present = (flags >> 4) & 0x01

    offset = 1

    # HR
    if hr_16bit:
        hr = struct.unpack_from("<H", data, offset)[0]
        offset += 2
    else:
        hr = data[offset]
        offset += 1

    # Energy Expended (skip if present)
    if energy_present:
        offset += 2

    # RR intervals (1/1024 sec resolution)
    rr_intervals = []
    if rr_present:
        while offset + 1 < len(data):
            rr_raw = struct.unpack_from("<H", data, offset)[0]
            rr_ms = rr_raw / 1024.0 * 1000.0  # Convert to ms
            rr_intervals.append(round(rr_ms, 1))
            offset += 2

    return hr, rr_intervals, contact_detected, contact_supported


sample_count = 0
rr_count = 0


def handle_hr(sender, data: bytearray):
    global sample_count, rr_count
    sample_count += 1

    hr, rr_list, contact, supported = parse_hr_measurement(data)

    flags_hex = data[0]
    rr_flag = "OUI" if (flags_hex >> 4) & 0x01 else "NON"

    line = f"  [#{sample_count:3d}] HR={hr} bpm | Contact={contact} | RR flag={rr_flag}"

    if rr_list:
        rr_count += len(rr_list)
        rr_str = ", ".join(f"{rr:.1f}ms" for rr in rr_list)
        line += f" | RR=[{rr_str}]"
    else:
        line += " | RR=(aucun)"

    print(line)


def handle_pmd_data(sender, data: bytearray):
    print(f"  [PMD DATA] {len(data)} bytes: {data[:20].hex()}...")


async def main():
    print("[SCAN] Recherche du Polar...")
    devices = await BleakScanner.discover(timeout=10.0)
    polar = next((d for d in devices if d.name and "polar" in d.name.lower()), None)

    if not polar:
        print("[ERREUR] Polar non trouve")
        return

    print(f"[OK] {polar.name}\n")

    async with BleakClient(polar) as client:
        # === TEST 1 : HR Standard avec RR ===
        print("=" * 60)
        print("TEST 1 : Heart Rate Standard (avec RR intervals)")
        print("=" * 60)

        await client.start_notify(HR_UUID, handle_hr)
        await asyncio.sleep(15)
        await client.stop_notify(HR_UUID)

        print(f"\n>> {sample_count} notifications HR, {rr_count} intervalles RR recus")

        # === TEST 2 : PMD SDK (PPI stream) ===
        print("\n" + "=" * 60)
        print("TEST 2 : PMD SDK (Polar Measurement Data)")
        print("=" * 60)

        # Lire les capabilities du PMD
        try:
            pmd_caps = await client.read_gatt_char(PMD_CONTROL)
            print(f"[PMD] Capabilities: {pmd_caps.hex()}")
            print(f"[PMD] Raw bytes: {list(pmd_caps)}")
        except Exception as e:
            print(f"[PMD] Erreur lecture capabilities: {e}")

        # Ecouter le data channel
        await client.start_notify(PMD_DATA, handle_pmd_data)

        # Envoyer commande pour demarrer le PPI stream
        # Format PMD: 0x02 = start, 0x03 = PPI measurement type
        start_ppi_cmd = bytearray([0x02, 0x03])
        print(f"[PMD] Envoi commande start PPI: {start_ppi_cmd.hex()}")

        try:
            await client.write_gatt_char(PMD_CONTROL, start_ppi_cmd, response=True)
            print("[PMD] Commande envoyee, attente 15 secondes...")
            await asyncio.sleep(15)
        except Exception as e:
            print(f"[PMD] Erreur: {e}")

            # Essayer aussi avec indicate au lieu de notify
            print("[PMD] Tentative avec indicate sur le control point...")
            try:
                await client.start_notify(PMD_CONTROL, lambda s, d: print(f"  [PMD CTRL] {d.hex()}"))
                await client.write_gatt_char(PMD_CONTROL, start_ppi_cmd, response=True)
                await asyncio.sleep(15)
            except Exception as e2:
                print(f"[PMD] Erreur indicate: {e2}")

        await client.stop_notify(PMD_DATA)

        print("\n[TERMINE]")


if __name__ == "__main__":
    asyncio.run(main())
