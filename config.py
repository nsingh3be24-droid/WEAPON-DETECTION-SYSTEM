# config.py — Weapon Detection System Configuration
# All tunable parameters in one place. Edit here; never hardcode elsewhere.

from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
LABELS_DIR      = DATA_DIR / "labels"
MODELS_DIR      = ROOT_DIR / "models"
LOGS_DIR        = ROOT_DIR / "logs"
EXPORTS_DIR     = ROOT_DIR / "exports"

# Create dirs if missing
for _dir in [RAW_DIR, PROCESSED_DIR, LABELS_DIR, MODELS_DIR, LOGS_DIR, EXPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
BASE_MODEL          = "yolov8m.pt"          # medium model; good accuracy/speed balance
POSE_MODEL          = "yolov8m-pose.pt"     # keypoint model for head/body orientation
TRAINED_MODEL_PATH  = MODELS_DIR / "weapon_detector.pt"
EXPORT_FORMAT       = "onnx"                # "onnx" | "engine" (TensorRT) | "tflite"
DEVICE              = "0"                 # "cpu" | "cuda" | "0" (GPU index)
HALF_PRECISION      = True                 # True only on CUDA + TensorRT

# ─────────────────────────────────────────────
# CLASSES  (must match your dataset labels)
# ─────────────────────────────────────────────
CLASS_NAMES = {
    0: "Handgun",
    1: "Knife",
    2: "Missile",
    3: "Rifle",
    4: "Shotgun",
    5: "Sword",
    6: "Tank",
}
WEAPON_CLASS_IDS = {0, 1, 2, 3, 4, 5, 6}
PERSON_CLASS_ID  = None

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
TRAIN_CFG = {
    "epochs":       150,
    "imgsz":        640,
    "batch":        4,
    "lr0":          0.01,
    "lrf":          0.001,
    "momentum":     0.937,
    "weight_decay": 0.0005,
    "warmup_epochs":3,
    "augment":      True,
    "mosaic":       1.0,
    "degrees":      5.0,
    "flipud":       0.0,
    "fliplr":       0.5,
    "workers":      4,
    "patience":     20,         # early-stop patience
    "project":      str(MODELS_DIR),
    "name":         "weapon_run",
    "exist_ok":     True,
    "device":       DEVICE,
}

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
INFER_CFG = {
    "conf":     0.45,       # minimum confidence
    "iou":      0.50,       # NMS IoU threshold
    "imgsz":    640,
    "device":   DEVICE,
    "half":     HALF_PRECISION,
    "max_det":  50,
    "verbose":  False,
}

# ─────────────────────────────────────────────
# BEHAVIOR / RULE ENGINE
# ─────────────────────────────────────────────
RULE_CFG = {
    # Proximity: fraction of frame diagonal at which weapon→person is "threatening"
    "proximity_threshold_ratio": 0.20,

    # Temporal: how many consecutive frames a rule must fire before an alert is raised
    "temporal_window_frames":    10,
    "alert_frame_count":          6,

    # Pose: angle (degrees) between weapon direction and nearest person's torso
    # below which we consider the weapon "pointed at" someone
    "pointing_angle_threshold":  35,

    # Occlusion: if weapon bbox overlaps desk/table region by this ratio → flag
    "occlusion_overlap_ratio":   0.40,

    # Cooldown: minimum seconds between repeated alerts for the same track pair
    "alert_cooldown_seconds":     5,
}

# ─────────────────────────────────────────────
# VIDEO / CAMERA
# ─────────────────────────────────────────────
CAMERA_SOURCE   = 0             # 0 = webcam; or RTSP URL string
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
TARGET_FPS      = 15            # processing FPS (skip frames above this)
DISPLAY_SCALE   = 1.0           # resize factor for on-screen display

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
PREPROCESS_CFG = {
    "target_size":      (640, 640),
    "normalize":        True,
    "mean":             (0.485, 0.456, 0.406),
    "std":              (0.229, 0.224, 0.225),
    "augment_train":    True,
    "val_split":        0.15,
    "test_split":       0.05,
    "seed":             42,
}

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_CFG = {
    "level":        "INFO",         # DEBUG | INFO | WARNING | ERROR
    "console":      True,
    "file":         True,
    "log_file":     str(LOGS_DIR / "weapon_detection.log"),
    "max_bytes":    10 * 1024 * 1024,   # 10 MB
    "backup_count": 5,
    "format":       "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    "datefmt":      "%Y-%m-%d %H:%M:%S",
}

# ─────────────────────────────────────────────
# ALERT OUTPUT
# ─────────────────────────────────────────────
ALERT_CFG = {
    "save_snapshots":   True,
    "snapshot_dir":     str(EXPORTS_DIR / "alerts"),
    "annotate_frame":   True,
    "beep_on_alert":    False,      # system beep (Linux: requires `beep`)
    "webhook_url":      None,       # POST JSON alert to this URL if set
}
