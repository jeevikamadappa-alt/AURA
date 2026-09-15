"""
AURA - YOLO Detector
====================
Uses a lightweight pretrained Ultralytics YOLO model (yolo11n.pt, COCO-trained).

IMPORTANT LIMITATION
--------------------
yolo11n.pt is trained on 80 COCO classes (person, car, truck, bicycle...).
It does NOT detect: flood, water, drainage, pipeline, pothole.

Functions
---------
detect_all()      -- Returns ALL detections from YOLO (every class, every confidence).
                     Used as raw data source for the pipeline and for debug display.
detect_vehicles() -- Filters to accident-relevant vehicle classes only (for YOLO+CLIP
                     accident detection).
get_model_classes() -- Returns the YOLO model's class ID → name mapping.
"""

import logging
import threading
from pathlib import Path
from typing import Optional

from config import (
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
    ACCIDENT_VEHICLE_CLASSES,
)

logger = logging.getLogger(__name__)

# Module-level cache so the model is loaded once per process
_yolo_model = None
_yolo_lock = threading.Lock()


def _load_model():
    """Lazily load the YOLO model (cached after first call)."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    try:
        from ultralytics import YOLO
        model_path = Path(YOLO_MODEL_PATH)
        logger.info("Loading YOLO model from: %s", model_path)
        _yolo_model = YOLO(str(model_path))
        logger.info(
            "YOLO model loaded successfully. Classes: %d (%s)",
            len(_yolo_model.names),
            ", ".join(list(_yolo_model.names.values())[:10]) + "...",
        )
        return _yolo_model

    except ImportError:
        logger.error("ultralytics package is not installed.")
        raise
    except Exception as exc:
        logger.error("Failed to load YOLO model: %s", exc)
        raise


def get_model_classes() -> dict:
    """Return the YOLO model's {class_id: class_name} mapping."""
    model = _load_model()
    return {int(k): v for k, v in model.names.items()}


def detect_all(image_bytes: bytes, min_conf: float = 0.10) -> dict:
    """
    Run YOLO inference and return ALL detections regardless of class.

    This is the raw YOLO output — all 80 COCO classes, all confidence levels
    above min_conf. Used for:
      - Debug / raw JSON display in the frontend
      - YOLO context enrichment (e.g., 'boat' detected = flood context)
      - Proving/disproving model capability on a given image

    Parameters
    ----------
    image_bytes : bytes
        Raw image bytes (JPEG/PNG/etc.)
    min_conf : float
        Minimum confidence to include in results (default: 0.10)

    Returns
    -------
    dict
        {
            "model_path":     str,
            "model_classes":  {int: str},          # full class map
            "image_width":    int,
            "image_height":   int,
            "total_detections": int,
            "all_detections": [                    # ALL classes above min_conf
                {
                    "class_id":   int,
                    "class_name": str,
                    "confidence": float,
                    "bbox_xyxy":  [x1, y1, x2, y2],
                    "bbox_xywh":  [cx, cy, w, h],
                    "area_ratio": float,           # fraction of image area
                    "accepted":   bool,            # True if above main threshold
                }
            ],
            "flood_relevant_detections": [str],    # classes that hint at flood
            "inference_time_s": float,
        }
    """
    import io, time
    from PIL import Image

    model = _load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = image.size
    img_area = img_w * img_h

    t0 = time.time()
    with _yolo_lock:
        results = model.predict(source=image, conf=min_conf, verbose=False)
    inference_time = round(time.time() - t0, 3)

    all_detections = []
    # COCO classes that could hint at flood/water context
    flood_hint_classes = {"boat", "surfboard", "umbrella"}

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id   = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            cx, cy, bw, bh  = [round(float(v), 2) for v in box.xywh[0].tolist()]
            box_area   = (x2 - x1) * (y2 - y1)
            area_ratio = round(box_area / img_area, 4) if img_area > 0 else 0.0

            all_detections.append({
                "class_id":   class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox_xyxy":  [x1, y1, x2, y2],
                "bbox_xywh":  [cx, cy, bw, bh],
                "area_ratio": area_ratio,
                "accepted":   confidence >= YOLO_CONFIDENCE_THRESHOLD,
            })

    # Sort by confidence descending
    all_detections.sort(key=lambda d: -d["confidence"])

    flood_relevant = [
        d["class_name"] for d in all_detections
        if d["class_name"].lower() in flood_hint_classes and d["accepted"]
    ]

    logger.info(
        "YOLO detect_all: %d detections (conf>=%.2f) | flood-hint classes: %s",
        len(all_detections), min_conf, flood_relevant or "none",
    )

    return {
        "model_path":                YOLO_MODEL_PATH,
        "model_classes":             {str(k): v for k, v in model.names.items()},
        "image_width":               img_w,
        "image_height":              img_h,
        "total_detections":          len(all_detections),
        "all_detections":            all_detections,
        "flood_relevant_detections": flood_relevant,
        "inference_time_s":          inference_time,
    }


def detect_vehicles(image_bytes: bytes) -> dict:
    """
    Filter YOLO detections to accident-relevant vehicle classes only.
    Used by the YOLO+CLIP accident detection stage.

    Returns
    -------
    dict
        {
            "detected":          bool,
            "confidence":        float,   # highest confidence among hits
            "detected_objects":  [...]
        }
    """
    import io
    from PIL import Image

    model = _load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with _yolo_lock:
        results = model.predict(
            source=image,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            verbose=False,
        )

    detected_objects = []
    max_confidence = 0.0

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id   = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])

            if class_name.lower() not in ACCIDENT_VEHICLE_CLASSES:
                continue

            xywh = box.xywh[0].tolist()
            detected_objects.append({
                "class":      class_name,
                "confidence": round(confidence, 4),
                "bbox":       [round(v, 2) for v in xywh],
            })
            if confidence > max_confidence:
                max_confidence = confidence

    return {
        "detected":         len(detected_objects) > 0,
        "confidence":       round(max_confidence, 4),
        "detected_objects": detected_objects,
    }
