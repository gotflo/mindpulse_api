"""
Test rapide : scan BLE → trouver le Polar → connecter → lire HR + PPI pendant 30s.
"""

import asyncio
import struct
import time

from bleak import BleakClient, BleakScanner

# UUIDs Polar
HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
PPI_UUID = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
BATTERY_UUID = "00002a19-0000-1000-8000-00805f9b34fb"


def handle_hr(sender, data: bytearray):
    flags = data[0]
    hr = struct.unpack_from("<H" if flags & 0x01 else "<B", data, 1)[0]
    print(f"  ❤️  HR = {hr} bpm")


def handle_ppi(sender, data: bytearray):
    index = 0
    ppis = []
    while index + 6 <= len(data):
        ppi = struct.unpack_from("<H", data, index + 1)[0]
        flags = data[index + 5]
        contact = "✅" if flags & 0x01 else "❌"
        ppis.append(f"{ppi}ms({contact})")
        index += 6
    print(f"  📊 PPI = {', '.join(ppis)}")


async def main():
    # ──── 1. SCAN ────
    print("=" * 50)
    print("🔍 Scan BLE en cours (10 secondes)...")
    print("=" * 50)

    devices = await BleakScanner.discover(timeout=10.0)

    # Afficher tous les devices avec un nom
    named = [d for d in devices if d.name]
    print(f"\n📱 {len(named)} appareils détectés avec un nom :\n")
    for i, d in enumerate(named, 1):
        print(f"  {i}. {d.name} [{d.address}]")

    # ──── 2. TROUVER LE POLAR ────
    polar = None
    for d in devices:
        if d.name and "polar" in d.name.lower():
            polar = d
            break

    if polar is None:
        print("\n❌ Aucun capteur Polar trouvé. Vérifie qu'il est allumé.")
        return

    print(f"\n✅ Polar trouvé : {polar.name} [{polar.address}]")

    # ──── 3. CONNEXION ────
    print(f"\n🔗 Connexion à {polar.name}...")

    async with BleakClient(polar) as client:
        if not client.is_connected:
            print("❌ Échec de connexion")
            return

        print(f"✅ Connecté à {polar.name}")

        # Batterie
        try:
            battery = await client.read_gatt_char(BATTERY_UUID)
            print(f"🔋 Batterie : {battery[0]}%")
        except Exception:
            print("🔋 Batterie : lecture impossible")

        # ──── 4. STREAMING 30 SECONDES ────
        print("\n" + "=" * 50)
        print("📡 Streaming HR + PPI pendant 30 secondes...")
        print("=" * 50 + "\n")

        await client.start_notify(HR_UUID, handle_hr)
        await client.start_notify(PPI_UUID, handle_ppi)

        await asyncio.sleep(30)

        await client.stop_notify(HR_UUID)
        await client.stop_notify(PPI_UUID)

        print("\n✅ Streaming terminé. Déconnexion.")

    print("✅ Déconnecté proprement.")


if __name__ == "__main__":
    asyncio.run(main())
