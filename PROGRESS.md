# Project Progress

- **Audio dataset:** Downloaded by Kathan.
- **RF dataset:** Downloaded and handled by Vidhi.
- **Step 1 — Manifest and data splits:** Kathan.
- **Step 2 — Random Forest audio baseline:** Vidhi; test F1 **0.904**.
- **Step 3 — Audio CNN:** Vidhi; test F1 **0.933**, recall **0.959**.
  Same leakage-safe Tier 1 split; CNN improved recall/F1 over the baseline,
  with slightly lower precision (**0.907** vs **0.923**). Trained on CPU;
  best checkpoint was epoch 7.
- **Step 4 — Initial noise test:** Vidhi; tested white and pink synthetic noise
  at 20, 10, 5, and 0 dB on the same 2,004 held-out clips. Thresholds stayed
  fixed from clean validation. CNN handled white noise better than the RF
  baseline at 10/5/0 dB, but at 0 dB F1 fell to **0.652**; with pink noise at
  0 dB CNN F1 fell to **0.345**. Real recorded noise testing is still needed.
- **Tests:** 19 passed.
- **Results:** `reports/audio_baseline.json`, `reports/audio_cnn.json`;
  `reports/audio_noise_robustness.json` and `.csv`; noise plot at
  `reports/figures/audio_noise_robustness.png`; saved CNN weights:
  `reports/models/audio_cnn_state_dict.pt`.
- **Next:** Test with independent real environmental recordings. RF model,
  fusion, and dashboard are not yet implemented.
