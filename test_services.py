"""
Test : decouvrir tous les services et caracteristiques GATT du Polar.
On va trouver le bon UUID pour le PPI/RR.
"""

import asyncio
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from bleak import BleakClient, BleakScanner


async def main():
    print("[SCAN] Recherche du Polar...")
    devices = await BleakScanner.discover(timeout=10.0)
    polar = next((d for d in devices if d.name and "polar" in d.name.lower()), None)

    if not polar:
        print("[ERREUR] Polar non trouve")
        return

    print(f"[OK] {polar.name} [{polar.address}]\n")

    async with BleakClient(polar) as client:
        print("=" * 60)
        print("TOUS LES SERVICES & CARACTERISTIQUES GATT")
        print("=" * 60)

        for service in client.services:
            print(f"\n[SERVICE] {service.uuid}")
            print(f"  Description: {service.description}")

            for char in service.characteristics:
                props = ", ".join(char.properties)
                print(f"  |-- {char.uuid} [{props}]")
                print(f"  |   Description: {char.description}")

                # Si c'est readable, on essaie de lire la valeur
                if "read" in char.properties:
                    try:
                        val = await client.read_gatt_char(char.uuid)
                        print(f"  |   Value: {val.hex()} ({list(val)})")
                    except Exception as e:
                        print(f"  |   Value: (erreur: {e})")

                for desc in char.descriptors:
                    print(f"  |   |-- Descriptor: {desc.uuid}")


if __name__ == "__main__":
    asyncio.run(main())
