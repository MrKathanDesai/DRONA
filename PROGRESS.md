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

## Fusion and dashboard
- **Audio + RF fusion ablation** (sum rule and learned logistic regression): Kathan (`fusion.md`).
  An upper bound, since audio and RF pairs are not co-recorded.
- **Dashboard:** Vidhi's 6-tab layout, with Kathan's RF feed + fused live state
  and the fusion results merged in. `streamlit run app/dashboard.py`.

## Tests
- 125 passing (`python -m pytest -q`).

## Still open
- Real Wi-Fi / Bluetooth RF recordings (CardRF / DroneDetect need a login) to
  measure RF false alarms on real traffic.
- Outdoor audio is weak for every model (recall ~0.2–0.45 at 1% false alarms).
