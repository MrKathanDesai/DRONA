# Video script (2 minutes) and LinkedIn post

## Video: 2-minute screen recording

Record with the Windows Game Bar (`Win + Alt + R` to start and stop) or OBS.
Before recording, open three things: the dashboard
(`.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py`), a terminal in
`D:\DRONA`, and the slides PDF.

| Time | Show on screen | Say |
|---|---|---|
| 0:00–0:15 | Title slide | "We are Vidhi Rojasara and Kathan Desai from Ahmedabad University. Our AI Lab project detects drones from their sound and, as a research extension, from their 2.4 GHz radio signal." |
| 0:15–0:30 | Architecture slide | "Audio is cut into one-second windows and scored by a pretrained audio network with a trained head. RF spectra are scored by a forest. The two scores are calibrated and added, and every window gets one of four states: No Drone, Known Drone, Unknown Drone or Low Confidence." |
| 0:30–0:45 | Terminal: run `.\.venv\Scripts\python.exe -m pytest -q tests\test_status.py` | "The whole pipeline is tested: leakage between splits, the four-state logic and the dashboard itself. 182 tests pass." |
| 0:45–1:15 | Dashboard → Replay detector. Sidebar: Svanström outdoor → DRONE → pick a recording; RF feed → "DroneRF drone: bebop" | "Here we replay an outdoor drone recording with recorded radio windows beside it. The audio sensor, the RF sensor and the final fused decision are shown side by side, with a window-by-window strip and a Grad-CAM view of which sounds pushed the score towards drone. The audio and radio were not recorded together, so this is a what-if, and the dashboard says so." |
| 1:15–1:35 | Dashboard → Fusion tab, point at the RF-silent table | "Our ablation shows something important: a learned fusion layer misses drones with no radio link, at 0.2 percent recall, while a simple evidence sum keeps 18.4 percent. So we deploy the sum." |
| 1:35–1:50 | RF slide (v1 vs v2) | "We also found that a near-perfect RF benchmark model called pure noise 'drone' on a second receiver. We retrained it, and it now catches 9 of 10 drones it never saw." |
| 1:50–2:00 | Project briefing page, "do and do not show" | "Indoors we reach PR-AUC 0.97; outdoors we catch about a fifth of drone windows at 1 percent false alarms. The dashboard states clearly what is and is not validated. Thank you." |

## LinkedIn post

> Our AI Lab project at Ahmedabad University: detecting drones from their sound and radio signals 🚁🎧📡
>
> With Kathan Desai, I built a counter-drone detection system that listens for a drone's rotor noise and, as a research extension, looks at its 2.4 GHz radio link. Every second it reports one of four states: No Drone, Known Drone, Unknown Drone, or Low Confidence, which asks a human to review.
>
> What we learned:
> 🔹 Indoors, our acoustic detector reaches PR-AUC 0.97. Outdoors, on recordings it never trained on, it catches about a fifth of drone windows at a 1% false-alarm rate. That gap is the real challenge.
> 🔹 An RF model that scored almost 100% on a public benchmark called pure noise "drone" on a second receiver: it had learned the recording session, not the drone. We found the shortcut and retrained it. It now catches 9 of 10 drones it had never seen.
> 🔹 In our fusion ablation, a learned fusion layer missed drones with no radio link, while a simple evidence sum kept detecting them. Simpler won.
>
> Built with Python, PyTorch, scikit-learn and Streamlit, with 182 automated tests and an explainable dashboard (Grad-CAM).
>
> Thank you to Prof. Maitrik Shah for guidance in the AI Lab course.
>
> Code: https://github.com/MrKathanDesai/DRONA
>
> @Ahmedabad University @Maitrik Shah
> #AI #MachineLearning #DroneDetection #AudioAI #AhmedabadUniversity

When posting, type `@` and pick Ahmedabad University and Prof. Maitrik Shah from
LinkedIn's suggestions, so they are real tags rather than plain text.
