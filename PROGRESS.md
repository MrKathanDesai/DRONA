# Project Progress

Combined work from branches `001-vidhi` (Vidhi) and `002` (Kathan).

## Data
- **Audio datasets** (DroneAudioDataset, ESC-50, ITU ARIS, Svanström outdoor): Kathan.
- **RF datasets** (DroneRF): Vidhi; Tampere second-receiver recordings: Kathan.
- **Step 1, manifest and leakage-safe splits:** Kathan.

## Audio
- **Random-forest baseline:** Vidhi; tier-1 test F1 **0.904**.
- **2D audio CNN:** Vidhi; tier-1 test F1 **0.933**, recall **0.959**.
- **Frozen pretrained encoder + trained head, tiers 1–4:** Kathan.
- **Architecture comparison (CNN, CNN+GRU, CNN+LSTM, transformer):** both;
  `audio_architecture_comparison.md` (Vidhi) and `deep_audio.md` (Kathan).
  Indoors all are strong (PR-AUC ~0.98); outdoors all are weak.
- **Noise robustness:** synthetic white/pink noise (Vidhi), and the full
  voice / traffic / wind / music / mixed / real speech / real music grid
  at 20 / 10 / 0 dB (Kathan).
- **Outdoor confusers:** Svanström helicopter/background negatives in training (Vidhi).
- **Uncertainty and open set:** four states per window (Kathan).

## RF
- **v1** (forest / logistic regression / 1D-CNN on DroneRF): looks perfect,
  but calls pure noise "drone" on a second receiver (`rf_tampere.md`).
- **v2** (deployed): fixed training data; noise is "no drone", 9 of 10 unseen
  Tampere drones caught (`rf_v2.md`).
- **v3** (experiment): simulated Wi-Fi / Bluetooth / microwave negatives,
  forest vs 1D-CNN, 3-of-5 smoothing, time-frequency features (`rf_v3.md`).
- **Calibration and held-out Phantom:** Vidhi (`rf_trust.md`).
- **DroneDetect subset:** Vidhi; 2-model classification only, no background (`dronedetect_subset.md`).
- **Uncertainty / unknown-drone flags and which-drone confusion matrices:** Kathan.

## Music masking
- Benchmark of drones under real music (-10 to +10 dB), a stationary-component
  front-end (negative result) and a music-aware detector (works; experiment only): Kathan (`masking.md`).

## Live page
- `python -m console`: operations console with acoustic sensor, music-masking
  monitor, recorded-radio sensor, site map, timeline and incident queue: Kathan.
- Alarm rule calibrated on validation recordings (`live_calibration.md`).
- Opt-in detector trained on simulated speaker -> microphone playback (`mic_robustness.md`).

## Fusion and dashboard
- **Audio + RF fusion ablation** (sum rule and learned logistic regression): Kathan (`fusion.md`).
  An upper bound, since audio and RF pairs are not co-recorded.
- **Dashboard:** Vidhi's 6-tab layout, with Kathan's RF feed + fused live state
  and the fusion results merged in. `streamlit run app/dashboard.py`.
- **Dashboard, professor-facing pass:** project briefing first (readiness read
  from files in `data/`, how a prediction is made, the four states, headline
  results with denominators and sources, what the results do / do not show,
  blockers); replay page with the recording-level result first, separate audio
  / RF / decision-rule evidence, RF clearly marked as recorded replay; evidence
  sum is the default rule and the learned layer is labelled experimental;
  System / Light / Dark theme (`.streamlit/config.toml`). Logic in
  `src/dronedet/status.py`, tested in `tests/test_status.py`.

## Tests
- 182 passing (`python -m pytest -q`; `tests/test_console.py` needs
  `pip install -r console/requirements.txt`).

## Outdoor validation and learning loop
- Outdoor aircraft (AeroSonicDB, held-out sessions): no alarms on 99 aircraft or
  317 ambience recordings (`aircraft.md`): Kathan.
- Operator-in-the-loop retraining with a champion/challenger gate, versions and
  rollback (`src/dronedet/feedback.py`): Kathan.

## Still open
- (Done 14 Nov 2026) RF/fusion artifacts rebuilt locally from verified data:
  DroneRF cache (23/23 archives, segment counts match the paper), Tampere
  (12/12 checksums), `rf_forest`, `rf_trust`, `rf_detector` (v2) and both
  fusion files. Every regenerated report matches the earlier one except the
  small RF v1 1D-CNN (recall changes of 0.01-0.04; CPU training is not
  bit-for-bit repeatable) and last-digit changes in fusion/calibration.
- No RF receiver integration: RF in the dashboard and console is recorded replay.
- Independent RF validation: the KU Leuven Drone RF dataset (new receiver,
  43.5 GB) needs a download request and shows no background recordings; CardRF
  (with Wi-Fi / Bluetooth) needs an IEEE DataPort login.
- Audio + RF fusion on paired data: no public dataset records both for the same
  flight; the fusion ablation stays an upper bound.
- Outdoor audio is weak for every model (recall ~0.2–0.45 at 1% false alarms).
