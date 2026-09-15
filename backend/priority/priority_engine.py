"""
AURA Priority Engine – Core Logic
===================================

Takes the incident detection JSON (output of detection/incident_detector.py)
and produces a structured priority response.

Scope
-----
This module answers ONLY:
  "How urgent is this incident, and what resources should respond?"

It does NOT:
  - Re-analyse the image
  - Call any external API
  - Perform any ML inference

Input contract (detection JSON schema)
---------------------------------------
{
    "success":          bool,
    "incident_type":    str,     # "Flood" | "Accident" | "Drainage Issue" | "Pipeline Issue" | ...
    "confidence":       float,   # 0.0 - 1.0
    "source":           str,     # "roboflow" | "clip-primary" | "yolo+clip" | "none"
    "incident_status":  str,     # "confirmed" | "needs_verification" | "no_incident" | "normal"
    "severity":         str,     # "High" | "Medium" | "Low" | "None"
    "detected_objects": list,
    "message":          str,
    "clip_scores":      dict,    # {class_name: float} - CLIP class scores
    "reasoning":        list,    # list of reasoning strings
    "yolo_raw":         dict,    # full raw YOLO detection data
}

Output contract
---------------
{
    "success": bool,
    "evaluated_at": str,          # ISO-8601 UTC timestamp
    "demo_disclaimer": str,       # Always present – resources are mock data
    "incidents": [
        {
            "incident_type": str,
            "confidence": float,
            "source": str,
            "priority": str,      # "P1" – "P4"
            "severity": str,      # "Critical" | "High" | "Medium" | "Low"
            "response_target": str,
            "recommended_action": str,
            "detected_objects": list,
            "detection_message": str,
            "resources": {
                "ambulance": {...},
                "hospital": {...},
                "rescue_crew": {...}
            }
        }
    ]
}
"""

import logging
from datetime import datetime, timezone
from typing import Union

from priority.priority_matrix import classify, PRIORITY_MATRIX
from priority.mock_resources import select_resource

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Resource selection rules per priority level
# ──────────────────────────────────────────────────────────────────────────────
# Maps priority code → which resource types to recommend.
# "none" means that resource type is not dispatched at that level.

_RESOURCE_PLAN = {
    "P1": {"ambulance": True,  "hospital": True,  "rescue_crew": True},
    "P2": {"ambulance": True,  "hospital": True,  "rescue_crew": True},
    "P3": {"ambulance": False, "hospital": False,  "rescue_crew": True},
    "P4": {"ambulance": False, "hospital": False,  "rescue_crew": False},
}

_DEMO_DISCLAIMER = (
    "⚠️  DEMO DATA: All resource entries (ambulances, hospitals, rescue crews) "
    "are fictional and created for demonstration purposes only. "
    "They do not reflect real-time availability or actual distances."
)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def prioritize(detection_result: Union[dict, list]) -> dict:
    """
    Assign priority and allocate mock resources to one or more detection results.

    Parameters
    ----------
    detection_result : dict | list
        A single detection JSON dict, or a list of dicts (batch mode).
        Each dict must conform to the incident detection output schema.

    Returns
    -------
    dict   Priority engine response (see module docstring for schema).
    """
    # Normalise to list for uniform processing
    if isinstance(detection_result, dict):
        detections = [detection_result]
    else:
        detections = list(detection_result)

    incidents = []
    for det in detections:
        if not isinstance(det, dict):
            logger.warning("Skipping non-dict detection entry: %s", type(det))
            continue
        incident = _process_single(det)
        if incident:
            incidents.append(incident)

    # Sort: P1 first (lowest response_target_minutes = highest urgency)
    incidents.sort(key=lambda i: PRIORITY_MATRIX[i["priority"]].response_target_minutes)

    return {
        "success": True,
        "evaluated_at": _utc_now(),
        "demo_disclaimer": _DEMO_DISCLAIMER,
        "incidents": incidents,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _process_single(det: dict) -> dict | None:
    """
    Convert a single detection dict into a prioritised incident dict.
    Returns None if the detection reports success=False.
    """
    if not det.get("success", True):
        logger.info("Skipping failed detection entry.")
        return None

    incident_type = det.get("incident_type", "Normal")
    confidence    = float(det.get("confidence", 0.0))
    source        = det.get("source", "none")
    detected_objs = det.get("detected_objects", [])
    det_message   = det.get("message", "")

    # ── Classify ──────────────────────────────────────────────────────────────
    level = classify(incident_type, confidence)

    # ── Allocate resources ────────────────────────────────────────────────────
    resource_plan = _RESOURCE_PLAN.get(level.code, {})
    resources = {}

    if resource_plan.get("ambulance"):
        resources["ambulance"] = select_resource("ambulance")
    else:
        resources["ambulance"] = _not_dispatched("ambulance", level.code)

    if resource_plan.get("hospital"):
        resources["hospital"] = select_resource("hospital")
    else:
        resources["hospital"] = _not_dispatched("hospital", level.code)

    if resource_plan.get("rescue_crew"):
        resources["rescue_crew"] = select_resource("rescue_crew")
    else:
        resources["rescue_crew"] = _not_dispatched("rescue_crew", level.code)

    logger.info(
        "Incident '%s' (conf=%.2f) → %s (%s)",
        incident_type, confidence, level.code, level.severity,
    )

    return {
        "incident_type":      incident_type,
        "confidence":         round(confidence, 4),
        "source":             source,
        "priority":           level.code,
        "severity":           level.severity,
        "response_target":    level.response_target,
        "recommended_action": level.recommended_action,
        "detected_objects":   detected_objs,
        "detection_message":  det_message,
        "resources":          resources,
        # Pass-through for debug/transparency
        "incident_status":    det.get("incident_status", "unknown"),
        "clip_scores":        det.get("clip_scores", {}),
        "reasoning":          det.get("reasoning", []),
        "yolo_raw":           det.get("yolo_raw", {}),
    }


def _not_dispatched(resource_type: str, priority_code: str) -> dict:
    """Return a placeholder dict for resources not required at this priority."""
    return {
        "dispatched": False,
        "reason": (
            f"Not required for {priority_code} incidents. "
            f"Manual escalation possible if situation changes."
        ),
        "demo_data": True,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
