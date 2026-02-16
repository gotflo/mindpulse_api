"""
Test complet du protocole PMD (Polar Measurement Data) SDK.

Le protocole PMD necessite :
1. Souscrire aux INDICATIONS sur le control point (fb005c81)
2. Souscrire aux NOTIFICATIONS sur le data channel (fb005c82)
3. Envoyer une requete GET capabilities pour le type de mesure
4. Envoyer une commande START avec les bons parametres
"""

import asyncio
import struct
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from bleak import BleakClient, BleakScanner

PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA    = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"

# PMD measurement types
TYPE_ECG = 0x00
TYPE_ACC = 0x02
TYPE_PPG = 0x01
TYPE_PPI = 0x03
TYPE_GYRO = 0x05

# PMD commands
CMD_GET_SETTINGS = 0x01
CMD_START = 0x02
CMD_STOP = 0x03

ppi_samples = []


def parse_pmd_control_response(data: bytearray):
    """Parse la reponse du PMD control point."""
    if len(data) < 2:
        print(f"  [CTRL] Reponse trop courte: {data.hex()}")
        return

    response_code = data[0]

    if response_code == 0x0F:
        # Features response
        print(f"  [CTRL] Features bitmap: {data.hex()}")
        features = data[1]
        types = []
        if features & 0x01: types.append("ECG")
        if features & 0x02: types.append("PPG")
        if features & 0x04: types.append("ACC")
        if features & 0x08: types.append("PPI")
        if features & 0x10: types.append("GYRO")
        if features & 0x20: types.append("MAG")
        print(f"  [CTRL] Types supportes: {', '.join(types)}")
        return

    # Command response: [op_code, measurement_type, status, ...]
    op_code = data[0]

    if op_code == 0xF0:
        # This is a response to a command
        cmd_code = data[1]
        meas_type = data[2]
        status = data[3] if len(data) > 3 else -1

        status_str = {0: "OK", 1: "INVALID_OP", 2: "INVALID_TYPE",
                      3: "NOT_ALLOWED", 4: "INVALID_PARAM", 5: "ALREADY_IN_USE",
                      6: "INVALID_RESOLUTION", 7: "INVALID_SAMPLE_RATE",
                      8: "INVALID_RANGE", 9: "INVALID_MTU",
                      10: "INVALID_CHANNELS", 11: "ERROR"}.get(status, f"UNKNOWN({status})")

        cmd_str = {1: "GET_SETTINGS", 2: "START", 3: "STOP"}.get(cmd_code, f"CMD({cmd_code})")
        type_str = {0: "ECG", 1: "PPG", 2: "ACC", 3: "PPI", 5: "GYRO"}.get(meas_type, f"TYPE({meas_type})")

        print(f"  [CTRL] Response: {cmd_str} {type_str} -> {status_str}")

        if cmd_code == CMD_GET_SETTINGS and status == 0 and len(data) > 4:
            print(f"  [CTRL] Settings data: {data[4:].hex()}")
            parse_settings(data[4:], meas_type)
    else:
        print(f"  [CTRL] Raw response: {data.hex()}")
        print(f"  [CTRL] Bytes: {list(data)}")


def parse_settings(data: bytearray, meas_type: int):
    """Parse les parametres disponibles pour un type de mesure."""
    idx = 0
    while idx < len(data):
        if idx + 1 >= len(data):
            break
        param_type = data[idx]
        idx += 1

        param_names = {0: "SAMPLE_RATE", 1: "RESOLUTION", 2: "RANGE", 3: "RANGE_MILLIUNIT",
                       4: "CHANNELS", 5: "FACTOR"}
        param_name = param_names.get(param_type, f"PARAM({param_type})")

        # Count of values
        if idx >= len(data):
            break
        count = data[idx]
        idx += 1

        values = []
        for _ in range(count):
            if idx + 1 >= len(data):
                break
            val = struct.unpack_from("<H", data, idx)[0]
            values.append(val)
            idx += 2

        print(f"  [CTRL]   {param_name}: {values}")


def handle_pmd_data(sender, data: bytearray):
    """Parse les donnees PMD."""
    if len(data) < 2:
        return

    meas_type = data[0]
    type_str = {0: "ECG", 1: "PPG", 2: "ACC", 3: "PPI", 5: "GYRO"}.get(meas_type, f"TYPE({meas_type})")

    if meas_type == TYPE_PPI:
        # PPI data format:
        # byte 0: measurement type (0x03)
        # byte 1: timestamp info / frame type
        # Then PPI samples: each has HR(1), PP(2), errEst(2), flags(1) = 6 bytes
        # OR newer format with different structure

        print(f"  [PPI DATA] {len(data)} bytes: {data.hex()}")

        # Try to parse PPI samples starting from different offsets
        for start_offset in [1, 2, 3, 8, 9, 10]:
            remaining = data[start_offset:]
            if len(remaining) >= 6 and len(remaining) % 6 == 0:
                n_samples = len(remaining) // 6
                print(f"  [PPI] Trying offset {start_offset} ({n_samples} samples):")
                for i in range(n_samples):
                    chunk = remaining[i*6:(i+1)*6]
                    hr = chunk[0]
                    pp = struct.unpack_from("<H", chunk, 1)[0]
                    err = struct.unpack_from("<H", chunk, 3)[0]
                    flags = chunk[5]
                    print(f"  [PPI]   HR={hr}, PPI={pp}ms, err={err}ms, flags=0x{flags:02x}")
                    ppi_samples.append(pp)
                break
        else:
            # Just dump the raw data for analysis
            print(f"  [PPI] Cannot find 6-byte alignment. Raw: {list(data)}")
    else:
        print(f"  [{type_str} DATA] {len(data)} bytes: {data[:30].hex()}...")


def handle_pmd_control(sender, data: bytearray):
    """Handle PMD control point indications."""
    parse_pmd_control_response(data)


async def main():
    print("[SCAN] Recherche du Polar...")
    devices = await BleakScanner.discover(timeout=10.0)

    # Trouver les deux Polar
    polars = [d for d in devices if d.name and "polar" in d.name.lower()]
    for p in polars:
        print(f"  Found: {p.name} [{p.address}]")

    polar = next((d for d in polars if "sense" in d.name.lower()), polars[0] if polars else None)

    if not polar:
        print("[ERREUR] Polar non trouve")
        return

    print(f"\n[CONNECT] {polar.name}...")

    async with BleakClient(polar) as client:
        print("[OK] Connecte\n")

        # === Subscribe to control point (indications) ===
        print("=" * 60)
        print("ETAPE 1 : Subscribe au control point + data channel")
        print("=" * 60)

        await client.start_notify(PMD_CONTROL, handle_pmd_control)
        print("[OK] Subscribed to PMD Control (indications)")

        await client.start_notify(PMD_DATA, handle_pmd_data)
        print("[OK] Subscribed to PMD Data (notifications)")

        await asyncio.sleep(1)

        # === Get PPI measurement settings ===
        print("\n" + "=" * 60)
        print("ETAPE 2 : GET settings pour PPI (type 0x03)")
        print("=" * 60)

        get_ppi_settings = bytearray([CMD_GET_SETTINGS, TYPE_PPI])
        print(f"[CMD] Envoi: {get_ppi_settings.hex()}")
        await client.write_gatt_char(PMD_CONTROL, get_ppi_settings, response=True)
        await asyncio.sleep(2)

        # === Get PPG measurement settings ===
        print("\n" + "=" * 60)
        print("ETAPE 3 : GET settings pour PPG (type 0x01)")
        print("=" * 60)

        get_ppg_settings = bytearray([CMD_GET_SETTINGS, TYPE_PPG])
        print(f"[CMD] Envoi: {get_ppg_settings.hex()}")
        await client.write_gatt_char(PMD_CONTROL, get_ppg_settings, response=True)
        await asyncio.sleep(2)

        # === Start PPI stream ===
        print("\n" + "=" * 60)
        print("ETAPE 4 : START PPI stream")
        print("=" * 60)

        # Simple start command
        start_ppi = bytearray([CMD_START, TYPE_PPI])
        print(f"[CMD] Envoi simple: {start_ppi.hex()}")
        await client.write_gatt_char(PMD_CONTROL, start_ppi, response=True)
        await asyncio.sleep(2)

        # If that didn't work, try with empty parameters
        # Format: CMD_START, TYPE, then setting parameters
        # PPI typically needs no parameters (or minimal ones)

        # Try with sample rate parameter: type=0x00, value
        start_ppi_v2 = bytearray([CMD_START, TYPE_PPI, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00])
        print(f"[CMD] Envoi avec params v2: {start_ppi_v2.hex()}")
        try:
            await client.write_gatt_char(PMD_CONTROL, start_ppi_v2, response=True)
        except Exception as e:
            print(f"[CMD] Erreur v2: {e}")
        await asyncio.sleep(2)

        # === Wait and collect data ===
        print("\n" + "=" * 60)
        print("ETAPE 5 : Attente de donnees PPI (20 secondes)...")
        print("=" * 60)

        await asyncio.sleep(20)

        # === Stop ===
        stop_ppi = bytearray([CMD_STOP, TYPE_PPI])
        try:
            await client.write_gatt_char(PMD_CONTROL, stop_ppi, response=True)
        except:
            pass

        await asyncio.sleep(1)

        await client.stop_notify(PMD_DATA)
        await client.stop_notify(PMD_CONTROL)

        print(f"\n[RESULTAT] {len(ppi_samples)} samples PPI collectes")
        if ppi_samples:
            print(f"[RESULTAT] PPI values: {ppi_samples[:20]}")

    print("[DECONNECTE]")


if __name__ == "__main__":
    asyncio.run(main())
