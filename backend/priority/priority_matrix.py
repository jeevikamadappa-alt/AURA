"""
AURA Priority Engine – Priority Matrix Definitions
===================================================
Defines the P1–P4 priority levels and maps (incident_type, confidence)
to a priority tier.

This module is PURE DATA — no I/O, no HTTP, no image processing.
"""

from dataclasses import dataclass, field
from typing import List


# ──────────────────────────────────────────────────────────────────────────────
# Priority level descriptors
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PriorityLevel:
    code: str                    # "P1", "P2", "P3", "P4"
    severity: str                # "Critical", "High", "Medium", "Low"
    response_target: str         # Human-readable SLA string
    response_target_minutes: int # Numeric upper bound (for sorting)
    recommended_action: str
    example_incidents: List[str] = field(default_factory=list)


PRIORITY_MATRIX = {
    "P1": PriorityLevel(
        code="P1",
        severity="Critical",
        response_target="Under 10 minutes",
        response_target_minutes=10,
        recommended_action=(
            "Immediate emergency response required. "
            "Dispatch ambulance, rescue crew, and police/fire support."
        ),
        example_incidents=[
            "Flood causing major danger",
            "Severe accident with likely casualties",
        ],
    ),
    "P2": PriorityLevel(
        code="P2",
        severity="High",
        response_target="Under 1 hour",
        response_target_minutes=60,
        recommended_action=(
            "Rapid emergency response. "
            "Dispatch emergency/rapid-response crew immediately."
        ),
        example_incidents=[
            "Significant flooding",
            "Major accident without confirmed severe injuries",
        ],
    ),
    "P3": PriorityLevel(
        code="P3",
        severity="Medium",
        response_target="24–48 hours",
        response_target_minutes=2880,
        recommended_action=(
            "Schedule public works or maintenance crew within 24–48 hours."
        ),
        example_incidents=[
            "Pothole",
            "Minor road obstruction",
            "Low-confidence flood signal",
        ],
    ),
    "P4": PriorityLevel(
        code="P4",
        severity="Low",
        response_target="3–7 days",
        response_target_minutes=10080,
        recommended_action=(
            "Log and add to normal maintenance queue."
        ),
        example_incidents=[
            "Minor civic issue",
            "No actionable incident detected",
        ],
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Incident → Priority mapping rules
# ──────────────────────────────────────────────────────────────────────────────
#
# Each rule is: (incident_type, min_confidence) → priority_code
# Rules are evaluated top-to-bottom; first match wins.
# ──────────────────────────────────────────────────────────────────────────────

CLASSIFICATION_RULES = [
    # ── Flood ─────────────────────────────────────────────────────────────────
    # High-confidence flood → life-threatening danger → P1
    {"incident_type": "Flood",    "min_confidence": 0.75, "priority": "P1"},
    # Moderate flood signal → significant but uncertain → P2
    {"incident_type": "Flood",    "min_confidence": 0.40, "priority": "P2"},
    # Low-confidence flood signal → treat as medium concern → P3
    {"incident_type": "Flood",    "min_confidence": 0.00, "priority": "P3"},

    # ── Accident ──────────────────────────────────────────────────────────────
    # High-confidence vehicle cluster → likely severe → P1
    {"incident_type": "Accident", "min_confidence": 0.75, "priority": "P1"},
    # Moderate vehicle detection → possible accident → P2
    {"incident_type": "Accident", "min_confidence": 0.40, "priority": "P2"},
    # Low-confidence → treat cautiously as P3
    {"incident_type": "Accident", "min_confidence": 0.00, "priority": "P3"},

    # ── Drought ───────────────────────────────────────────────────────────────
    # High-confidence drought signal → severe resource crisis → P1
    {"incident_type": "Drought",  "min_confidence": 0.75, "priority": "P1"},
    # Moderate drought signal → significant but not yet critical → P2
    {"incident_type": "Drought",  "min_confidence": 0.40, "priority": "P2"},
    # Low-confidence drought signal → monitor, escalate if confirmed → P3
    {"incident_type": "Drought",  "min_confidence": 0.00, "priority": "P3"},

    # ── Drainage ──────────────────────────────────────────────────────────────
    # High-confidence drainage failure → significant infrastructure risk → P2
    {"incident_type": "Drainage", "min_confidence": 0.75, "priority": "P2"},
    # Moderate drainage signal → schedule maintenance → P3
    {"incident_type": "Drainage", "min_confidence": 0.40, "priority": "P3"},
    # Low-confidence → log and monitor → P4
    {"incident_type": "Drainage", "min_confidence": 0.00, "priority": "P4"},

    # ── PipeLeak ──────────────────────────────────────────────────────────────
    # High-confidence pipe break → structural/service risk → P2
    {"incident_type": "PipeLeak", "min_confidence": 0.75, "priority": "P2"},
    # Moderate signal → inspect within 24 hours → P3
    {"incident_type": "PipeLeak", "min_confidence": 0.40, "priority": "P3"},
    # Low-confidence → add to maintenance queue → P4
    {"incident_type": "PipeLeak", "min_confidence": 0.00, "priority": "P4"},

    # ── Pothole ───────────────────────────────────────────────────────────────
    # High-confidence pothole -> road safety risk -> P3
    {"incident_type": "Pothole", "min_confidence": 0.55, "priority": "P3"},
    # Lower-confidence pothole -> log and schedule inspection -> P4
    {"incident_type": "Pothole", "min_confidence": 0.00, "priority": "P4"},

    # ── Drainage Issue (CLIP-detected) ────────────────────────────────────────
    # High-confidence drainage failure -> P2
    {"incident_type": "Drainage Issue", "min_confidence": 0.50, "priority": "P2"},
    # Moderate -> P3
    {"incident_type": "Drainage Issue", "min_confidence": 0.30, "priority": "P3"},
    # Low -> P4
    {"incident_type": "Drainage Issue", "min_confidence": 0.00, "priority": "P4"},

    # ── Pipeline Issue (CLIP-detected) ────────────────────────────────────────
    # High-confidence pipeline break -> P1 (infrastructure risk)
    {"incident_type": "Pipeline Issue", "min_confidence": 0.55, "priority": "P1"},
    # Moderate -> P2
    {"incident_type": "Pipeline Issue", "min_confidence": 0.30, "priority": "P2"},
    # Low -> P3
    {"incident_type": "Pipeline Issue", "min_confidence": 0.00, "priority": "P3"},

    # ── Needs Verification ────────────────────────────────────────────────────
    # Ambiguous detection - requires manual review -> P3
    {"incident_type": "Needs Verification", "min_confidence": 0.00, "priority": "P3"},

    # ── No Incident Detected ──────────────────────────────────────────────────
    # Distinct from Normal - no evidence found -> P4
    {"incident_type": "No Incident Detected", "min_confidence": 0.00, "priority": "P4"},

    # ── Normal ────────────────────────────────────────────────────────────────
    {"incident_type": "Normal",   "min_confidence": 0.00, "priority": "P4"},
]


def classify(incident_type: str, confidence: float) -> PriorityLevel:
    """
    Return the appropriate PriorityLevel for a given incident type and
    confidence score, using the CLASSIFICATION_RULES table.

    Parameters
    ----------
    incident_type : str
        One of "Flood", "Accident", "Drainage Issue", "Pipeline Issue",
        "Drainage", "PipeLeak", "Pothole", "Needs Verification",
        "No Incident Detected", "Normal"
    confidence    : float  0.0 - 1.0

    Returns
    -------
    PriorityLevel  Always returns a valid level (defaults to P4).
    """
    for rule in CLASSIFICATION_RULES:
        if (
            rule["incident_type"].lower() == incident_type.lower()
            and confidence >= rule["min_confidence"]
        ):
            return PRIORITY_MATRIX[rule["priority"]]

    # Fallback – should never be reached with current rules
    return PRIORITY_MATRIX["P4"]
