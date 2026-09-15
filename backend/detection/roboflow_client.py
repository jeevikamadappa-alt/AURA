"""
AURA - Roboflow Inference Client
=================================
Provides one function per Roboflow model:

  detect_flood()       -> flood-detection model
  detect_drainage()    -> drainage model  (drainage-6m7z9/1)
  detect_pipe_leak()   -> pipe-leak model (pipe-leak/1)

All three share the single ROBOFLOW_API_KEY from the environment.
The key is NEVER logged, stored in responses, or exposed.
"""

import base64
import logging
from typing import Optional

import requests

from config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_PROJECT_ID,
    ROBOFLOW_MODEL_VERSION,
    ROBOFLOW_CONFIDENCE_THRESHOLD,
    ROBOFLOW_FLOOD2_MODEL_ID,
    ROBOFLOW_FLOOD2_MODEL_VERSION,
    ROBOFLOW_FLOOD2_CONFIDENCE_THRESHOLD,
    ROBOFLOW_DRAINAGE_MODEL_ID,
    ROBOFLOW_DRAINAGE_MODEL_VERSION,
    ROBOFLOW_DRAINAGE_CONFIDENCE_THRESHOLD,
    ROBOFLOW_PIPELEAK_MODEL_ID,
    ROBOFLOW_PIPELEAK_MODEL_VERSION,
    ROBOFLOW_PIPELEAK_CONFIDENCE_THRESHOLD,
    ROBOFLOW_ACCIDENT_MODEL_ID,
    ROBOFLOW_ACCIDENT_MODEL_VERSION,
    ROBOFLOW_ACCIDENT_CONFIDENCE_THRESHOLD,
)

from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

# Connection pooling session to eliminate TLS handshake overhead across calls
_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=1)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def _prepare_image_for_roboflow(image_bytes: bytes, max_dim: int = 1024) -> bytes:
    """
    Resize image to max_dim (preserving aspect ratio) to drastically speed up
    Roboflow base64 encoding and network transmission.
    """
    try:
        import io
        from PIL import Image

        im = Image.open(io.BytesIO(image_bytes))
        w, h = im.size
        if max(w, h) > max_dim:
            im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            im.save(out, format="JPEG", quality=85, optimize=True)
            return out.getvalue()
    except Exception:
        pass
    return image_bytes


def _mask_key(key: str) -> str:
    """Return a safely-masked version of the key for logging."""
    if len(key) < 8:
        return "***"
    return key[:4] + "..." + key[-4:]


# -----------------------------------------------------------------------------
# Generic Roboflow inference helper
# -----------------------------------------------------------------------------

def _call_model(
    image_bytes: bytes,
    project_id: str,
    model_version: int,
    confidence_threshold: float,
) -> Optional[dict]:
    """
    POST image_bytes (base64-encoded) to any Roboflow hosted inference model.

    Returns
    -------
    dict | None
        On success  -> raw Roboflow JSON response.
        On failure  -> None (caller decides how to proceed).
    """
    if not ROBOFLOW_API_KEY:
        logger.warning(
            "ROBOFLOW_API_KEY is not set - skipping call for '%s/%s'.",
            project_id, model_version,
        )
        return None

    # Pre-shrink oversized images to reduce base64 upload payload from ~7MB to ~150KB
    ready_bytes = _prepare_image_for_roboflow(image_bytes)
    encoded = base64.b64encode(ready_bytes).decode("utf-8")

    url = (
        f"https://detect.roboflow.com/{project_id}"
        f"/{model_version}"
        f"?api_key={ROBOFLOW_API_KEY}"
        f"&confidence={int(confidence_threshold * 100)}"
        f"&overlap=30"
    )

    try:
        response = _session.post(
            url,
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        logger.error("Roboflow request timed out (project=%s).", project_id)
    except requests.exceptions.HTTPError as exc:
        logger.error(
            "Roboflow HTTP error %s for project '%s'.",
            exc.response.status_code, project_id,
        )
    except requests.exceptions.RequestException as exc:
        logger.error(
            "Roboflow request failed for project '%s': %s",
            project_id, type(exc).__name__,
        )
    except ValueError:
        logger.error(
            "Roboflow returned non-JSON response for project '%s'.", project_id
        )

    return None


# -----------------------------------------------------------------------------
# Per-model public functions (one per Roboflow model)
# -----------------------------------------------------------------------------

def detect_flood(image_bytes: bytes) -> Optional[dict]:
    """Call flood model 1 (flood-detection/1). Returns raw Roboflow JSON or None."""
    return _call_model(
        image_bytes,
        project_id=ROBOFLOW_PROJECT_ID,
        model_version=ROBOFLOW_MODEL_VERSION,
        confidence_threshold=ROBOFLOW_CONFIDENCE_THRESHOLD,
    )


def detect_flood2(image_bytes: bytes) -> Optional[dict]:
    """Call flood model 2 (flood-detection-3susv/1). Returns raw Roboflow JSON or None."""
    return _call_model(
        image_bytes,
        project_id=ROBOFLOW_FLOOD2_MODEL_ID,
        model_version=ROBOFLOW_FLOOD2_MODEL_VERSION,
        confidence_threshold=ROBOFLOW_FLOOD2_CONFIDENCE_THRESHOLD,
    )


def detect_accident_roboflow(image_bytes: bytes) -> Optional[dict]:
    """Call the Roboflow accident-detection model (accident-detection-uevan/2). Returns raw JSON or None."""
    return _call_model(
        image_bytes,
        project_id=ROBOFLOW_ACCIDENT_MODEL_ID,
        model_version=ROBOFLOW_ACCIDENT_MODEL_VERSION,
        confidence_threshold=ROBOFLOW_ACCIDENT_CONFIDENCE_THRESHOLD,
    )


def detect_drainage(image_bytes: bytes) -> Optional[dict]:
    """Call the drainage-6m7z9 model. Returns raw Roboflow JSON or None."""
    return _call_model(
        image_bytes,
        project_id=ROBOFLOW_DRAINAGE_MODEL_ID,
        model_version=ROBOFLOW_DRAINAGE_MODEL_VERSION,
        confidence_threshold=ROBOFLOW_DRAINAGE_CONFIDENCE_THRESHOLD,
    )


def detect_pipe_leak(image_bytes: bytes) -> Optional[dict]:
    """Call the pipe-leak model. Returns raw Roboflow JSON or None."""
    return _call_model(
        image_bytes,
        project_id=ROBOFLOW_PIPELEAK_MODEL_ID,
        model_version=ROBOFLOW_PIPELEAK_MODEL_VERSION,
        confidence_threshold=ROBOFLOW_PIPELEAK_CONFIDENCE_THRESHOLD,
    )


# -----------------------------------------------------------------------------
# Response parsers
# -----------------------------------------------------------------------------

def extract_best_prediction(
    roboflow_response: dict,
    confidence_threshold: float,
) -> Optional[dict]:
    """
    Return the highest-confidence prediction above confidence_threshold, or None.

    Returns
    -------
    dict | None
        {"confidence": float, "class": str}  if passes threshold, else None.
    """
    predictions = roboflow_response.get("predictions", [])
    if not predictions:
        return None

    best = max(predictions, key=lambda p: p.get("confidence", 0))
    confidence = best.get("confidence", 0)

    if confidence >= confidence_threshold:
        return {"confidence": confidence, "class": best.get("class", "unknown")}

    return None


def extract_flood_result(roboflow_response: dict) -> Optional[dict]:
    """Backward-compatibility alias for extract_best_prediction (flood threshold)."""
    return extract_best_prediction(roboflow_response, ROBOFLOW_CONFIDENCE_THRESHOLD)

