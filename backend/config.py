"""
AURA Backend – Configuration
All secrets are read from environment variables.
Never hard-code or log any API key.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env when present (dev only)


# ──────────────────────────────────────────────
# Roboflow – shared API key
# ──────────────────────────────────────────────
ROBOFLOW_API_KEY: str = os.environ.get("ROBOFLOW_API_KEY", "")

# ── Flood model 1 (existing: flood-detection/1) ──
ROBOFLOW_PROJECT_ID: str = os.environ.get("ROBOFLOW_PROJECT_ID", "flood-detection")
ROBOFLOW_MODEL_VERSION: int = int(os.environ.get("ROBOFLOW_MODEL_VERSION", "1"))
ROBOFLOW_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("ROBOFLOW_CONFIDENCE_THRESHOLD", "0.40")
)

# ── Flood model 2 (flood-detection-3susv/1) ───────
ROBOFLOW_FLOOD2_MODEL_ID: str = os.environ.get(
    "ROBOFLOW_FLOOD2_MODEL_ID", "flood-detection-3susv"
)
ROBOFLOW_FLOOD2_MODEL_VERSION: int = int(
    os.environ.get("ROBOFLOW_FLOOD2_MODEL_VERSION", "1")
)
ROBOFLOW_FLOOD2_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("ROBOFLOW_FLOOD2_CONFIDENCE_THRESHOLD", "0.40")
)

# ── Drainage model ─────────────────────────────
# model_id = 'drainage-6m7z9/1'  →  project=drainage-6m7z9, version=1
ROBOFLOW_DRAINAGE_MODEL_ID: str = os.environ.get(
    "ROBOFLOW_DRAINAGE_MODEL_ID", "drainage-6m7z9"
)
ROBOFLOW_DRAINAGE_MODEL_VERSION: int = int(
    os.environ.get("ROBOFLOW_DRAINAGE_MODEL_VERSION", "1")
)
ROBOFLOW_DRAINAGE_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("ROBOFLOW_DRAINAGE_CONFIDENCE_THRESHOLD", "0.40")
)

# ── Pipe-leak model ────────────────────────────
# model_id = 'pipe-leak/1'  →  project=pipe-leak, version=1
ROBOFLOW_PIPELEAK_MODEL_ID: str = os.environ.get(
    "ROBOFLOW_PIPELEAK_MODEL_ID", "pipe-leak"
)
ROBOFLOW_PIPELEAK_MODEL_VERSION: int = int(
    os.environ.get("ROBOFLOW_PIPELEAK_MODEL_VERSION", "1")
)
ROBOFLOW_PIPELEAK_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("ROBOFLOW_PIPELEAK_CONFIDENCE_THRESHOLD", "0.40")
)

# ── Accident model (accident-detection-uevan/2) ──
# model_id = 'accident-detection-uevan/2'  →  project=accident-detection-uevan, version=2
ROBOFLOW_ACCIDENT_MODEL_ID: str = os.environ.get(
    "ROBOFLOW_ACCIDENT_MODEL_ID", "accident-detection-uevan"
)
ROBOFLOW_ACCIDENT_MODEL_VERSION: int = int(
    os.environ.get("ROBOFLOW_ACCIDENT_MODEL_VERSION", "2")
)
ROBOFLOW_ACCIDENT_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("ROBOFLOW_ACCIDENT_CONFIDENCE_THRESHOLD", "0.40")
)

# ──────────────────────────────────────────────
# YOLO
# ──────────────────────────────────────────────
YOLO_MODEL_PATH: str = os.environ.get("YOLO_MODEL_PATH", "yolo11n.pt")

# Minimum confidence for YOLO vehicle detections
YOLO_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("YOLO_CONFIDENCE_THRESHOLD", "0.40")
)

# COCO class names we treat as accident-relevant vehicles
ACCIDENT_VEHICLE_CLASSES: set = {"car", "truck", "bus", "motorcycle"}

# ──────────────────────────────────────────────
# File upload
# ──────────────────────────────────────────────
MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS: set = {"jpg", "jpeg", "png", "webp", "bmp"}
