# Microphone robustness: detector retrained with speaker -> mic copies

Played from a speaker and heard by a laptop microphone, a drone loses its low rotor hum and gains echo and hiss. The deployed detector was never trained on that. `scripts/train_mic_detector.py` adds simulated speaker -> room -> microphone copies x3 of each training clip (28269 copies; all drones, a sample of backgrounds) and rebuilds the same tier-4 detector (same split, source balancing, 1% false alarms on validation negatives, uncertainty flags). Test clips get their own channel copies.

| test set | input | drone recall before | after | false alarms before | after |
|---|---|---|---|---|---|
| indoor held-out | clean | 0.78 | 0.64 | 0.000 | 0.000 |
| indoor held-out | through speaker + mic | 0.44 | 0.61 | 0.007 | 0.003 |
| outdoor held-out (Svanström) | clean | 0.15 | 0.13 | 0.002 | 0.000 |
| outdoor held-out (Svanström) | through speaker + mic | 0.06 | 0.12 | 0.005 | 0.002 |

The channel is simulated, not measured on a real speaker and microphone, so real playback may differ. It is lower on clean audio, so the live page keeps the original detector by default; `python -m console --mic-model speaker` switches the microphone to this one.
