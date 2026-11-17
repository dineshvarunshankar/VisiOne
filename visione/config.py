import torch
import os

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Models
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8m-worldv2.pt")
DEPTH_MODEL_VARIANT = os.path.join(PROJECT_ROOT, "models", "DA3")

# Audio
MIC_DEVICE_INDEX = 1
# Options: "ai" (XTTS) or "system" (pyttsx3).
TTS_ENGINE = "ai"
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_LANGUAGE = "en"
TTS_SPEAKER_WAV = os.path.join(PROJECT_ROOT, "models", "female_01.wav")
TTS_BASE_RATE = 150.0
TTS_RATE = 160
WAKE_WORD = "hello"
LISTENING_TIMEOUT = 3.0

# Safety logic
# Fraction of image width used for the center zone.
CENTER_ZONE_RATIO = 0.20
DISTANCE_WARN_NEAR = 0.5    # m
VELOCITY_THRESHOLD = 0.5    # m/s
SPEED_ALERT_THRESHOLD = 5.0 # m/s

MIN_TRACK_AGE = 15          # Minimum valid measurements
VELOCITY_SMOOTHING = 10      # Velocity averaging window (measurements)

COOLDOWN_SPEED   = 3.0      # seconds
COOLDOWN_GENERAL = 5.0      # seconds



# Tracking
TRACK_BUFFER = 30
TRACK_MATCH_THRESH = 0.8

# Tuning
INPUT_SIZE = (640, 384)
DEPTH_SCALE = 0.8
PERCENTILE_CLOSEST = 5
BACKGROUND_CUTOFF_RATIO = 1.20

# Vocabulary
SAFETY_VOCABULARY = [
    "person", "car", "truck", "bus", "motorcycle", "bicycle", "train",
    "rider", "police car", "ambulance",

    "stairs", "curb", "pothole", "pole", "tree", "traffic light", "stop sign",
    "fire hydrant", "wet floor sign", "construction barrier", "fence",

    "chair", "table", "couch", "bed", "desk", "cabinet", "door", "doorway",
    "window", "trash can", "box", "luggage", "backpack", "umbrella",
    "shoe", "toy", "wire", "glass door",

    "elevator", "escalator", "bench", "fountain", "atm", "vending machine",
    "counter", "sink", "toilet", "mirror",

    "cup", "bottle", "plate", "laptop", "monitor", "tv", "keyboard",
    "phone", "keys", "book", "bag", "lamp", "plant"
]

# Debug
SHOW_VISUALIZATION = True
WAIT_KEY_DELAY = 1
WINDOW_TITLE = "Visione | RGB + Depth"
SAVE_VIDEO = False
LOG_TO_FILE = False
