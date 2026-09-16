"""
AURA Backend – Configuration
All secrets are read from environment variables.
Never hard-code or log any API key.
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env relative to this file so it works from any launch directory ─────
# This means `python app.py` from the repo root, `backend/`, or anywhere else
# will all find backend/.env correctly.
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)  # loads backend/.env regardless of CWD

_log = logging.getLogger(__name__)


def _mask(key: str) -> str:
    """Return a safely-masked API key for logging (never logs the real value)."""
    if not key:
        return "<NOT SET>"
    if len(key) < 8:
        return "***"
    return key[:4] + "..." + key[-4:]


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
# Flask / Security
# ──────────────────────────────────────────────
SECRET_KEY: str = os.environ.get("SECRET_KEY", "change_me_in_production")

# ──────────────────────────────────────────────
# File upload
# ──────────────────────────────────────────────
MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS: set = {"jpg", "jpeg", "png", "webp", "bmp"}

# ──────────────────────────────────────────────
# Startup diagnostics – log masked key values
# ──────────────────────────────────────────────
# These run at import time so every server startup prints key status.
# Values are MASKED – only the first/last 4 chars are shown, never the real key.
_log.info("── AURA config loaded ──────────────────────────────────────────────────")
_log.info("  .env path resolved to : %s (exists=%s)", _env_path, _env_path.exists())
_log.info("  ROBOFLOW_API_KEY       : %s", _mask(ROBOFLOW_API_KEY))
_log.info("  YOLO_MODEL_PATH        : %s", YOLO_MODEL_PATH)
_log.info("  YOLO_CONFIDENCE        : %s", YOLO_CONFIDENCE_THRESHOLD)
_log.info("  FLASK_ENV              : %s", os.environ.get("FLASK_ENV", "<not set>"))
if not ROBOFLOW_API_KEY:
    _log.warning(
        "ROBOFLOW_API_KEY is empty! Roboflow models will be SKIPPED. "
        "Copy backend/.env.example to backend/.env and fill in your key."
    )
_log.info("────────────────────────────────────────────────────────────────────────")
