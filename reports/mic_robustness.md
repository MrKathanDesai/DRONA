# Microphone robustness: detector retrained with speaker -> mic copies

Played from a speaker and heard by a laptop microphone, a drone loses its low rotor hum and gains echo and hiss. The deployed detector was never trained on that. `scripts/train_mic_detector.py` adds simulated speaker -> room -> microphone copies x3 of each training clip (26358 copies; all drones, a sample of backgrounds) and rebuilds the same tier-4 detector (same split, source balancing, 1% false alarms on validation negatives, uncertainty flags). Test clips get their own channel copies. Indoor rows use validation recordings: the tier-1 test split is tier-4 training data, so it cannot be used.

| test set | input | drone recall before | after | false alarms before | after |
|---|---|---|---|---|---|
| indoor validation | clean | 0.72 | 0.59 | 0.000 | 0.012 |
| indoor validation | through speaker + mic | 0.45 | 0.61 | 0.002 | 0.003 |
| outdoor held-out (Svanström) | clean | 0.15 | 0.14 | 0.002 | 0.003 |
| outdoor held-out (Svanström) | through speaker + mic | 0.06 | 0.12 | 0.005 | 0.008 |

The channel is simulated, not measured on a real speaker and microphone, so real playback may differ. It is lower on clean audio, so the live page keeps the original detector by default; `python -m console --mic-model speaker` switches the microphone to this one.
