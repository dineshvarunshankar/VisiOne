# VisiOne: Giving Vision a Voice

VisiOne is an assistive-vision research prototype that detects objects, estimates their distance and closing speed, and provides spoken feedback.
Voice queries let users search for objects by name.

## How it works

- **Detection:** YOLO-World identifies objects from a configurable vocabulary.
- **Depth:** Depth Anything 3 estimates depth from RGB images; an alternative entry point uses an Intel RealSense camera.
- **Tracking:** Kalman filters estimate relative closing speed across frames.
- **Feedback:** Whisper transcribes queries; Coqui XTTS or system speech provides responses and obstacle alerts.

## Setup

Requires Python 3.10, a camera, microphone, and speakers.

```bash
git clone https://github.com/dineshvarunshankar/VisiOne.git
cd VisiOne
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r visione/requirements.txt
```

For monocular depth, install Depth Anything 3:

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
python -m pip install ./Depth-Anything-3 --no-deps
```

Place these assets under `visione/models/`:

- `yolov8m-worldv2.pt`: YOLO-World weights.
- `DA3/`: a pretrained metric Depth Anything 3 model, including its configuration and weights.
- `female_01.wav`: a speaker reference recording for XTTS.

Set the microphone, model paths, and voice engine in `visione/config.py`. Use `TTS_ENGINE = "system"` to use system speech without XTTS assets.

## Run

For an RGB camera:

```bash
python visione/main.py
```

For an Intel RealSense camera:

```bash
python visione/realsense/main.py
```

Say **“Hello, find my bottle”** to search for an object. Press **`q`** in the display window to exit.