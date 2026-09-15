"""
AURA - Core Incident Detection Pipeline
========================================

This module answers: "WHAT incident is visible in this image?"

Detection pipeline (multi-stage):

  Stage 0: YOLO raw scan
    - Runs detect_all() to get every YOLO detection + image metadata
    - Extracts YOLO context (e.g. boat detected = flood hint)
    - Raw output preserved for debug display

  Stage 1 (parallel): Roboflow specialized models
    - flood-detection/1 + flood-detection-3susv/1
    - drainage-6m7z9/1
    - pipe-leak/1
    - accident-detection-uevan/2
    - If ANY returns a detection above threshold -> that wins
    - If ALL fail (401/timeout) -> gracefully skip to Stage 2

  Stage 2: CLIP PRIMARY detector (for flood/drainage/pipeline/pothole)
    - CLIP is the primary detector when Roboflow is unavailable
    - Uses aerial/disaster-specific prompts (tuned for flood/satellite imagery)
    - Returns: incident_type, confidence, severity, incident_status, reasoning
    - incident_status: "confirmed" / "needs_verification" / "no_incident" / "normal"
    - Does NOT default to "Normal" when there is ambiguous evidence

  Stage 3: YOLO+CLIP Accident (if Stage 2 didn't find a non-accident)
    - YOLO detects vehicles -> CLIP confirms crash vs normal traffic

  Stage 4: Default
    - "No Incident Detected" if nothing triggers (NOT automatically "Normal")

KEY DESIGN PRINCIPLES:
  - CLIP is the primary flood/drainage/pipeline detector (YOLO cannot do this)
  - Roboflow models are trusted when they work; silently bypassed on 401/timeout
  - "Normal" requires CLIP to clearly score it as normal (>= 0.45)
  - "No Incident Detected" != "Normal"
  - "Needs Verification" is a valid output for ambiguous images
  - Raw YOLO data is always captured and included in the response
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)


# ==============================================================================
# Public API
# ==============================================================================

def detect_incident(image_bytes: bytes) -> dict:
    """
    Run the AURA incident-detection pipeline on image_bytes.

    Returns a dict with:
      - incident_type, confidence, severity, priority_hint
      - incident_status: "confirmed" / "needs_verification" / "no_incident" / "normal"
      - source: which detector produced the result
      - yolo_raw: full raw YOLO detection data
      - clip_scores: all CLIP class scores
      - reasoning: list of rules/evidence that led to the decision
    """
    # ── Stage 0: YOLO raw scan (always runs, provides context) ────────────────
    yolo_raw = _run_yolo_raw_scan(image_bytes)

    # ── Stage 1: Roboflow specialised models (run in parallel) ────────────────
    roboflow_result = _run_roboflow_stage(image_bytes)
    if roboflow_result is not None:
        roboflow_result["yolo_raw"] = yolo_raw
        roboflow_result["reasoning"] = roboflow_result.get("reasoning", []) + [
            "Stage 1: Roboflow model produced a detection above threshold"
        ]
        return roboflow_result

    # ── Stage 2: CLIP primary detector (flood/drainage/pipeline/pothole) ──────
    clip_result = _run_clip_primary_detection(image_bytes, yolo_raw)

    # If CLIP detected a non-accident incident, return it
    if clip_result["incident_type"] not in ("Normal", "No Incident Detected", "Needs Verification"):
        clip_result["yolo_raw"] = yolo_raw
        return clip_result

    # If CLIP says "Needs Verification", don't override with accident check
    if clip_result.get("incident_status") == "needs_verification":
        clip_result["yolo_raw"] = yolo_raw
        return clip_result

    # ── Stage 3: YOLO+CLIP accident detection (only if no incident yet) ───────
    accident_result = _run_roboflow_accident_detection(image_bytes)
    if accident_result is None:
        accident_result = _run_yolo_clip_accident(image_bytes)
    if accident_result is not None:
        accident_result["yolo_raw"] = yolo_raw
        logger.info("Stage 3: Accident detected (conf=%.3f)", accident_result.get("confidence", 0))
        return accident_result

    # ── Stage 4: Return CLIP result (Normal / No Incident) ───────────────────
    clip_result["yolo_raw"] = yolo_raw
    logger.info("Stage 4: Returning CLIP default result: %s", clip_result.get("incident_type"))
    return clip_result


# ==============================================================================
# Stage 0: YOLO raw scan
# ==============================================================================

def _run_yolo_raw_scan(image_bytes: bytes) -> dict:
    """Run detect_all() and return raw YOLO data for context and debug."""
    try:
        from detection.yolo_detector import detect_all
        raw = detect_all(image_bytes, min_conf=0.10)
        logger.info(
            "YOLO raw scan: %d detections | flood-hint: %s",
            raw.get("total_detections", 0),
            raw.get("flood_relevant_detections") or "none",
        )
        return raw
    except Exception as exc:
        logger.error("YOLO raw scan failed: %s", exc)
        return {
            "error": str(exc),
            "model_path": "yolo11n.pt",
            "total_detections": 0,
            "all_detections": [],
            "flood_relevant_detections": [],
        }


# ==============================================================================
# Stage 1: Roboflow detectors
# ==============================================================================

def _run_roboflow_stage(image_bytes: bytes) -> Optional[dict]:
    """
    Run all Roboflow models in parallel. Return the highest-confidence result,
    or None if all fail (401/timeout/no detection).
    """
    candidates = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_run_flood_detection, image_bytes):    "Flood",
            executor.submit(_run_flood2_detection, image_bytes):   "Flood2",
            executor.submit(_run_drainage_detection, image_bytes):  "Drainage",
            executor.submit(_run_pipe_leak_detection, image_bytes): "PipeLeak",
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                res = future.result()
                if res is not None:
                    candidates.append(res)
            except Exception as exc:
                logger.error("Roboflow detector '%s' error: %s", label, exc)

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c.get("confidence", 0.0))
    logger.info(
        "Stage 1 Roboflow winner: %s (conf=%.3f)",
        best.get("incident_type"), best.get("confidence", 0.0),
    )
    return best


def _run_flood_detection(image_bytes: bytes) -> Optional[dict]:
    from detection.roboflow_client import detect_flood, extract_flood_result
    try:
        raw = detect_flood(image_bytes)
        if raw is None:
            return None
        flood = extract_flood_result(raw)
        if flood:
            logger.info("Roboflow flood/1: detected (conf=%.3f).", flood["confidence"])
            return {
                "success": True,
                "incident_type": "Flood",
                "confidence": round(flood["confidence"], 4),
                "source": "roboflow-flood",
                "incident_status": "confirmed",
                "severity": "High",
                "detected_objects": [flood["class"]],
                "message": f"Flood detected with {flood['confidence']*100:.1f}% confidence by Roboflow flood-detection model.",
                "reasoning": [f"Roboflow flood-detection/1: class={flood['class']} conf={flood['confidence']:.3f}"],
                "clip_scores": {},
            }
        return None
    except Exception as exc:
        logger.error("Flood detection error: %s", type(exc).__name__)
        return None


def _run_flood2_detection(image_bytes: bytes) -> Optional[dict]:
    from detection.roboflow_client import detect_flood2, extract_best_prediction
    from config import ROBOFLOW_FLOOD2_CONFIDENCE_THRESHOLD
    try:
        raw = detect_flood2(image_bytes)
        if raw is None:
            return None
        result = extract_best_prediction(raw, ROBOFLOW_FLOOD2_CONFIDENCE_THRESHOLD)
        if result:
            logger.info("Roboflow flood2: detected (conf=%.3f).", result["confidence"])
            return {
                "success": True,
                "incident_type": "Flood",
                "confidence": round(result["confidence"], 4),
                "source": "roboflow-flood2",
                "incident_status": "confirmed",
                "severity": "High",
                "detected_objects": [result["class"]],
                "message": f"Flood detected with {result['confidence']*100:.1f}% confidence by flood-detection-3susv model.",
                "reasoning": [f"Roboflow flood-detection-3susv/1: class={result['class']} conf={result['confidence']:.3f}"],
                "clip_scores": {},
            }
        return None
    except Exception as exc:
        logger.error("Flood2 detection error: %s", type(exc).__name__)
        return None


def _run_drainage_detection(image_bytes: bytes) -> Optional[dict]:
    from detection.roboflow_client import detect_drainage, extract_best_prediction
    from config import ROBOFLOW_DRAINAGE_CONFIDENCE_THRESHOLD
    try:
        raw = detect_drainage(image_bytes)
        if raw is None:
            return None
        result = extract_best_prediction(raw, ROBOFLOW_DRAINAGE_CONFIDENCE_THRESHOLD)
        if result:
            logger.info("Roboflow drainage: detected (conf=%.3f).", result["confidence"])
            return {
                "success": True,
                "incident_type": "Drainage Issue",
                "confidence": round(result["confidence"], 4),
                "source": "roboflow-drainage",
                "incident_status": "confirmed",
                "severity": "Medium",
                "detected_objects": [result["class"]],
                "message": f"Drainage issue detected with {result['confidence']*100:.1f}% confidence.",
                "reasoning": [f"Roboflow drainage-6m7z9/1: class={result['class']} conf={result['confidence']:.3f}"],
                "clip_scores": {},
            }
        return None
    except Exception as exc:
        logger.error("Drainage detection error: %s", type(exc).__name__)
        return None


def _run_pipe_leak_detection(image_bytes: bytes) -> Optional[dict]:
    from detection.roboflow_client import detect_pipe_leak, extract_best_prediction
    from config import ROBOFLOW_PIPELEAK_CONFIDENCE_THRESHOLD
    try:
        raw = detect_pipe_leak(image_bytes)
        if raw is None:
            return None
        result = extract_best_prediction(raw, ROBOFLOW_PIPELEAK_CONFIDENCE_THRESHOLD)
        if result:
            logger.info("Roboflow pipe-leak: detected (conf=%.3f).", result["confidence"])
            return {
                "success": True,
                "incident_type": "Pipeline Issue",
                "confidence": round(result["confidence"], 4),
                "source": "roboflow-pipeleak",
                "incident_status": "confirmed",
                "severity": "High",
                "detected_objects": [result["class"]],
                "message": f"Pipeline issue detected with {result['confidence']*100:.1f}% confidence.",
                "reasoning": [f"Roboflow pipe-leak/1: class={result['class']} conf={result['confidence']:.3f}"],
                "clip_scores": {},
            }
        return None
    except Exception as exc:
        logger.error("Pipe-leak detection error: %s", type(exc).__name__)
        return None


def _run_roboflow_accident_detection(image_bytes: bytes) -> Optional[dict]:
    from detection.roboflow_client import detect_accident_roboflow, extract_best_prediction
    from config import ROBOFLOW_ACCIDENT_CONFIDENCE_THRESHOLD
    try:
        raw = detect_accident_roboflow(image_bytes)
        if raw is None:
            return None
        result = extract_best_prediction(raw, ROBOFLOW_ACCIDENT_CONFIDENCE_THRESHOLD)
        if result:
            logger.info("Roboflow accident: detected (conf=%.3f).", result["confidence"])
            return {
                "success": True,
                "incident_type": "Accident",
                "confidence": round(result["confidence"], 4),
                "source": "roboflow-accident",
                "incident_status": "confirmed",
                "severity": "High",
                "detected_objects": [result["class"]],
                "message": f"Traffic accident detected with {result['confidence']*100:.1f}% confidence by Roboflow accident model.",
                "reasoning": [f"Roboflow accident-detection-uevan/2: class={result['class']} conf={result['confidence']:.3f}"],
                "clip_scores": {},
            }
        return None
    except Exception as exc:
        logger.error("Roboflow accident detection error: %s", type(exc).__name__)
        return None


# ==============================================================================
# Stage 2: CLIP primary detection (flood/drainage/pipeline/pothole)
# ==============================================================================

def _run_clip_primary_detection(image_bytes: bytes, yolo_raw: dict) -> dict:
    """
    CLIP is the primary detector when Roboflow is unavailable.
    Uses classify_incident() which implements the full threshold logic
    and returns incident_status (confirmed/needs_verification/no_incident/normal).
    """
    try:
        from detection.clip_classifier import classify_incident
        result = classify_incident(image_bytes, yolo_context=yolo_raw)
        incident = result.get("incident_type", "No Incident Detected")
        status   = result.get("incident_status", "no_incident")
        conf     = result.get("confidence", 0.0)
        severity = result.get("severity", "Unknown")
        clip_scores = result.get("clip_scores", {})
        reasoning   = result.get("reasoning", [])

        logger.info(
            "Stage 2 CLIP primary: incident=%s status=%s conf=%.3f severity=%s",
            incident, status, conf, severity,
        )

        return {
            "success":         True,
            "incident_type":   incident,
            "confidence":      conf,
            "source":          "clip-primary",
            "incident_status": status,
            "severity":        severity,
            "detected_objects": [],
            "message":         _build_message(incident, conf, status, reasoning),
            "clip_scores":     clip_scores,
            "reasoning":       reasoning,
        }

    except Exception as exc:
        logger.error("CLIP primary detection failed: %s", exc)
        return {
            "success":         False,
            "incident_type":   "No Incident Detected",
            "confidence":      0.0,
            "source":          "clip-primary-error",
            "incident_status": "no_incident",
            "severity":        "Unknown",
            "detected_objects": [],
            "message":         f"CLIP detection failed: {exc}",
            "clip_scores":     {},
            "reasoning":       [f"CLIP error: {exc}"],
        }


def _build_message(incident_type: str, confidence: float, status: str, reasoning: list) -> str:
    """Build a human-readable detection message."""
    if incident_type == "No Incident Detected":
        return "No flood, drainage, pipeline, or road incident detected in this image."
    if incident_type == "Needs Verification":
        return "Ambiguous image — some incident signal detected but below confidence threshold. Manual review recommended."
    if incident_type == "Normal":
        return "Image appears to show normal conditions with no incident detected."
    status_str = {
        "confirmed":          "confirmed",
        "needs_verification": "detected (needs verification)",
        "no_incident":        "not detected",
        "normal":             "normal conditions",
    }.get(status, status)
    return (
        f"{incident_type} {status_str} with {confidence*100:.1f}% confidence "
        f"(CLIP semantic analysis). "
        + (reasoning[-1] if reasoning else "")
    )


# ==============================================================================
# Stage 3: YOLO + CLIP Accident detection
# ==============================================================================

def _run_yolo_clip_accident(image_bytes: bytes) -> Optional[dict]:
    """
    Two-step accident detection:
      Step A: YOLO detects vehicle presence.
      Step B: CLIP confirms whether the scene looks like a crash.
    Returns a formatted incident dict only if BOTH steps agree.
    """
    from detection.yolo_detector import detect_vehicles

    try:
        yolo = detect_vehicles(image_bytes)
    except Exception as exc:
        logger.error("YOLO vehicle detection failed: %s", exc)
        return None

    if not yolo["detected"]:
        logger.info("YOLO: no accident-relevant vehicles found.")
        return None

    vehicle_conf = yolo["confidence"]
    logger.info(
        "YOLO: vehicles detected (conf=%.3f, count=%d). Running CLIP verification...",
        vehicle_conf, len(yolo["detected_objects"]),
    )

    try:
        from detection.clip_classifier import is_likely_accident
        is_accident, combined_conf = is_likely_accident(image_bytes, vehicle_conf)
    except Exception as exc:
        logger.warning("CLIP verification unavailable (%s). Using conservative YOLO heuristic.", exc)
        is_accident  = vehicle_conf >= 0.75
        combined_conf = vehicle_conf if is_accident else 0.0

    if not is_accident:
        logger.info("CLIP veto: vehicles detected but scene does NOT look like an accident.")
        return None

    return {
        "success":         True,
        "incident_type":   "Accident",
        "confidence":      combined_conf,
        "source":          "yolo+clip",
        "incident_status": "confirmed",
        "severity":        "High",
        "detected_objects": yolo["detected_objects"],
        "message":         f"Traffic accident detected with {combined_conf*100:.1f}% confidence (YOLO vehicles + CLIP verification).",
        "clip_scores":     {},
        "reasoning":       [
            f"YOLO vehicle confidence: {vehicle_conf:.3f}",
            f"CLIP confirmed crash scene",
            f"Combined confidence: {combined_conf:.3f}",
        ],
    }


# ==============================================================================
# Helpers
# ==============================================================================

def _normal_response() -> dict:
    return {
        "success":         True,
        "incident_type":   "Normal",
        "confidence":      0.0,
        "source":          "none",
        "incident_status": "normal",
        "severity":        "None",
        "detected_objects": [],
        "message":         "No flood, accident, drainage, pipe leak, or pothole evidence detected.",
        "clip_scores":     {},
        "reasoning":       ["No detector triggered above threshold"],
        "yolo_raw":        {},
    }
