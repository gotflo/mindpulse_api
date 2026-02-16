"""
Test d'integration complet : Polar BLE -> PPI -> HRV features -> scores cognitifs.
Utilise le vrai PolarClient mis a jour avec le protocole PMD SDK.
"""

import asyncio
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.config.settings import load_config
from app.acquisition.polar_client import PolarClient, PolarSample, ConnectionState
from app.signal.ppi_cleaning import PPICleaner
from app.signal.windowing import SlidingWindow, WindowData
from app.features.hrv_features import HRVFeatureExtractor
from app.ml.model import CognitiveModel
from app.ml.inference import CognitiveInference

# Stats
hr_count = 0
ppi_count = 0
window_count = 0
last_hr = 0


def main():
    config = load_config()

    # Build pipeline components
    cleaner = PPICleaner(config.signal)
    extractor = HRVFeatureExtractor()
    inference = CognitiveInference(config.ml, cleaner, extractor)
    window = SlidingWindow(config.signal)
    client = PolarClient(config.ble)

    def on_sample(sample: PolarSample):
        global hr_count, ppi_count, last_hr

        if sample.hr > 0:
            hr_count += 1
            last_hr = sample.hr
            if hr_count % 5 == 0:
                print(f"  [HR] #{hr_count} HR={sample.hr} bpm")

        if sample.ppi_ms:
            ppi_count += len(sample.ppi_ms)
            ppi_str = ", ".join(f"{p}ms" for p in sample.ppi_ms)
            print(f"  [PPI] +{len(sample.ppi_ms)} samples: [{ppi_str}]  (total={ppi_count})")

            # Feed to sliding window
            window.add_samples(sample.ppi_ms, sample.timestamp)

    def on_window(win: WindowData):
        global window_count
        window_count += 1

        print(f"\n  {'='*55}")
        print(f"  [WINDOW #{window_count}] {win.sample_count} samples, "
              f"span={win.window_end - win.window_start:.1f}s")

        # Run full inference
        result = inference.process_window(win)

        f = result.features
        s = result.scores
        t = result.fatigue_trend

        print(f"  [HRV] HR={f.mean_hr:.0f}bpm RMSSD={f.rmssd:.1f}ms "
              f"SDNN={f.sdnn:.1f}ms pNN50={f.pnn50:.1f}%")
        print(f"  [HRV] LF/HF={f.lf_hf_ratio:.2f} SD1={f.sd1:.1f} SD2={f.sd2:.1f}")
        print(f"  [SCORES] Stress={s.stress:.1f} Load={s.cognitive_load:.1f} "
              f"Fatigue={s.fatigue:.1f}")
        print(f"  [TREND] slope={t.slope:+.2f}/min "
              f"predicted_10min={t.predicted_fatigue_10min:.1f} "
              f"confidence={t.confidence:.2f}")
        print(f"  [QUALITY] {result.window_quality:.1%}")
        print(f"  {'='*55}\n")

    def on_state(info):
        print(f"  [STATE] {info.connection_state.value} | "
              f"battery={info.battery_level}% | "
              f"signal={info.signal_quality:.1%}")

    # Wire callbacks
    client.on_sample(on_sample)
    client.on_state_change(on_state)
    window.on_window(on_window)

    async def run():
        print("=" * 60)
        print("TEST COMPLET DU PIPELINE")
        print("Polar -> PPI (PMD SDK) -> HRV -> Scores cognitifs")
        print("=" * 60)

        # Scan + connect
        print("\n[1/4] Scan...")
        device = await client.scan()
        if not device:
            print("Polar non trouve!")
            return

        print(f"\n[2/4] Connexion a {device.name}...")
        ok = await client.connect()
        if not ok:
            print("Echec connexion!")
            return

        print(f"\n[3/4] Demarrage streaming (HR + PPI via PMD)...")
        await client.start_streaming()

        print(f"\n[4/4] Collecte pendant 60 secondes...\n")
        await asyncio.sleep(60)

        # Stop
        print("\n[STOP] Arret du streaming...")
        await client.stop_streaming()
        await client.disconnect()

        # Summary
        print("\n" + "=" * 60)
        print("RESUME")
        print("=" * 60)
        print(f"  HR samples:     {hr_count}")
        print(f"  PPI samples:    {ppi_count}")
        print(f"  Windows:        {window_count}")
        print(f"  Last HR:        {last_hr} bpm")
        print(f"  Signal quality: {client.info.signal_quality:.1%}")

        if window_count == 0:
            print("\n  PROBLEME: Aucune fenetre generee!")
            print(f"  Buffer: {window.sample_count} samples, "
                  f"duree={window.buffer_duration_sec:.1f}s")
            print(f"  (il faut ~{config.signal.window_size_sec}s de PPI)")
        else:
            print(f"\n  Pipeline fonctionnel!")

    asyncio.run(run())


if __name__ == "__main__":
    main()
