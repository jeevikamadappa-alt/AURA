"""
AURA - CLIP Semantic Incident Classifier
==========================================
Uses OpenAI CLIP (via open-clip-torch) to score an image against carefully
crafted text prompts for each incident type.

Design Role
-----------
CLIP is the PRIMARY detector for:
  * Flood / Inundation
  * Drainage overflow / blockage
  * Pipeline / pipe-leak issues
  * Pothole / road damage

CLIP is the SECONDARY (confirmation) step for:
  * Accident (primary: Roboflow accident model + YOLO vehicle detection)

Performance
-----------
  - Model is loaded once at startup and cached.
  - Text features are pre-computed and cached after the first call.
  - Subsequent inference: ~0.1-0.5s per image on CPU.

Aerial Imagery Note
-------------------
Many disaster images are aerial/drone/satellite views.
The prompts below are specifically tuned for both ground-level AND aerial views.
"""

import io
import logging
import threading
import time
from typing import Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Model cache
_model = None
_preprocess = None
_tokenizer = None
_model_lock = threading.Lock()

_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "openai"

# Confidence thresholds (configurable)
THRESHOLD_HIGH   = 0.55
THRESHOLD_MEDIUM = 0.35
THRESHOLD_LOW    = 0.25

# Per-class minimum score to trigger that incident label
CLASS_MIN_SCORE = {
    "Flood":    0.28,
    "Drainage": 0.30,
    "Pipeline": 0.30,
    "Pothole":  0.32,
    "Accident": 0.35,
    "Normal":   0.45,
}

# Text prompts optimised for aerial, ground-level, and satellite disaster imagery
_PROMPT_SETS = {
    "Flood": [
        "an aerial photograph of a flooded area with water covering land",
        "satellite view of inundated neighborhoods and submerged buildings",
        "aerial view of floodwater spreading across roads and fields",
        "drone footage of severe flooding with houses partially underwater",
        "overhead view of a flood disaster with water-covered urban areas",
        "aerial image of large-scale inundation with brown or grey floodwater",
        "a photograph of a flooded street with water covering the road",
        "a photo of deep floodwater flowing through a residential area",
        "a photo of submerged vehicles and buildings during a flood",
        "an image of an area underwater after heavy rainfall flooding",
    ],
    "Drainage": [
        "a photograph of a blocked or clogged drain overflowing with water",
        "an image of drainage overflow causing waterlogging on a road",
        "a photo of a broken drain with sewage water spilling out",
        "an aerial view of waterlogging caused by failed drainage systems",
        "a photograph of a clogged storm drain with accumulated debris",
        "an image of an overflowing drainage channel or ditch",
        "a photo of stagnant pooled water from a blocked drain",
    ],
    "Pipeline": [
        "a photograph of a burst water pipe with water spraying out",
        "an image of a broken pipeline with leaking water or gas",
        "a photo of an exposed or damaged underground pipeline",
        "a photograph of a corroded or cracked pipe infrastructure failure",
        "an image of water gushing from a ruptured water main",
        "a photo of a pipeline leak with water pooling on the ground",
        "aerial view of a damaged pipeline with visible rupture",
    ],
    "Pothole": [
        "a photograph of a large pothole in a road surface",
        "an image of a broken road with deep holes and cracks",
        "a photo of road surface damage with potholes and crumbling asphalt",
        "a close-up of multiple potholes on a deteriorated road",
        "an aerial view of a road with visible surface damage and potholes",
    ],
    "Accident": [
        "a photograph of a traffic accident with damaged or crashed vehicles",
        "an image of a road collision with overturned or wrecked cars",
        "a photo of a vehicle pile-up accident on a highway",
        "an image showing emergency responders at a serious car accident",
        "a photo of a vehicle crash with visible damage and debris on the road",
    ],
    "Normal": [
        "a photograph of a normal road with vehicles driving safely",
        "an aerial view of a city or town in normal conditions with no damage",
        "an image of a clear dry road with no flooding or damage",
        "a photo of normal infrastructure in good condition",
        "an overhead view of a well-maintained road with no incidents",
    ],
}

_text_features_cache = None
_text_labels_cache = None
_text_cache_lock = threading.Lock()


def _load_model():
    """Load and cache the CLIP model (thread-safe, loads once per process)."""
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return _model, _preprocess, _tokenizer
    with _model_lock:
        if _model is not None:
            return _model, _preprocess, _tokenizer
        import open_clip
        logger.info("Loading CLIP model (%s / %s)...", _MODEL_NAME, _PRETRAINED)
        model, _, preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED
        )
        tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
        model.eval()
        _model = model
        _preprocess = preprocess
        _tokenizer = tokenizer
        logger.info("CLIP model loaded successfully.")
        return _model, _preprocess, _tokenizer


def _get_text_features(model, tokenizer, device):
    """Compute and cache text feature embeddings for all prompts."""
    global _text_features_cache, _text_labels_cache
    if _text_features_cache is not None:
        return _text_features_cache, _text_labels_cache
    with _text_cache_lock:
        if _text_features_cache is not None:
            return _text_features_cache, _text_labels_cache
        all_prompts, all_labels = [], []
        for label, prompts in _PROMPT_SETS.items():
            for p in prompts:
                all_prompts.append(p)
                all_labels.append(label)
        tokens = tokenizer(all_prompts).to(device)
        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        _text_features_cache = features.cpu()
        _text_labels_cache = all_labels
        logger.info("CLIP text features cached (%d prompts, %d classes).", len(all_prompts), len(_PROMPT_SETS))
        return _text_features_cache, _text_labels_cache


def classify_image(image_bytes: bytes) -> dict:
    """
    Run CLIP classification. Returns:
      {incident_type, confidence, all_scores, reasoning}
    """
    t0 = time.time()
    model, preprocess, tokenizer = _load_model()
    device = "cpu"

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    text_features, text_labels = _get_text_features(model, tokenizer, device)
    text_features = text_features.to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    logits = (image_features @ text_features.T).squeeze(0)
    probs = torch.softmax(logits * 100.0, dim=0).cpu().numpy()

    class_scores, class_counts = {}, {}
    for i, label in enumerate(text_labels):
        class_scores[label] = class_scores.get(label, 0.0) + float(probs[i])
        class_counts[label] = class_counts.get(label, 0) + 1
    for label in class_scores:
        class_scores[label] /= class_counts[label]

    total = sum(class_scores.values()) or 1.0
    class_scores = {k: v / total for k, v in class_scores.items()}

    best_class = max(class_scores, key=lambda k: class_scores[k])
    best_score = class_scores[best_class]
    elapsed = time.time() - t0

    logger.info(
        "CLIP inference %.1fs | top=%s (%.3f) | scores=%s",
        elapsed, best_class, best_score,
        {k: f"{v:.3f}" for k, v in sorted(class_scores.items(), key=lambda x: -x[1])},
    )

    return {
        "incident_type": best_class,
        "confidence":    round(best_score, 4),
        "all_scores":    {k: round(v, 4) for k, v in class_scores.items()},
        "reasoning":     [
            f"CLIP top class: {best_class} ({best_score*100:.1f}%)",
            f"Inference time: {elapsed:.1f}s",
        ],
    }


def _compute_severity(incident_type: str, confidence: float) -> str:
    """Derive severity from incident type and CLIP confidence score."""
    if incident_type in ("Normal", "No Incident Detected"):
        return "None"
    if incident_type == "Needs Verification":
        return "Unknown"
    if confidence >= THRESHOLD_HIGH:
        return "High" if incident_type in ("Flood", "Pipeline", "Drainage", "Accident") else "Medium"
    elif confidence >= THRESHOLD_MEDIUM:
        return "Medium" if incident_type in ("Flood", "Pipeline", "Drainage", "Accident") else "Low"
    return "Low"


def classify_incident(image_bytes: bytes, yolo_context: Optional[dict] = None) -> dict:
    """
    Full incident classification: CLIP + optional YOLO context boosts.

    incident_status values:
      "confirmed"           - clear evidence of incident
      "needs_verification"  - weak signal, below high threshold
      "no_incident"         - no meaningful evidence
      "normal"              - clearly normal conditions
    """
    result = classify_image(image_bytes)
    scores = dict(result["all_scores"])
    top_class = result["incident_type"]
    reasoning = list(result["reasoning"])

    # Apply YOLO context boosts
    if yolo_context:
        boat_detected = any(
            d.get("class_name", "").lower() == "boat"
            for d in yolo_context.get("all_detections", [])
        )
        if boat_detected:
            scores["Flood"] = min(1.0, scores.get("Flood", 0) + 0.20)
            reasoning.append("YOLO detected 'boat' - flood context boost +0.20 applied")
            total = sum(scores.values()) or 1.0
            scores = {k: v / total for k, v in scores.items()}
            top_class = max(scores, key=lambda k: scores[k])

    flood_score    = scores.get("Flood",    0.0)
    drainage_score = scores.get("Drainage", 0.0)
    pipeline_score = scores.get("Pipeline", 0.0)
    pothole_score  = scores.get("Pothole",  0.0)
    accident_score = scores.get("Accident", 0.0)
    normal_score   = scores.get("Normal",   0.0)

    ordered = sorted(scores.items(), key=lambda x: -x[1])

    incident_type   = None
    incident_status = None

    # 1. Check top incident class against threshold
    for class_name, score in ordered:
        if class_name == "Normal":
            continue
        min_score = CLASS_MIN_SCORE.get(class_name, 0.30)
        if score >= min_score and class_name == top_class:
            incident_type = class_name
            if score >= THRESHOLD_MEDIUM:
                incident_status = "confirmed"
                reasoning.append(
                    f"{class_name} confirmed: score {score:.3f} >= MEDIUM threshold {THRESHOLD_MEDIUM}"
                )
            else:
                incident_status = "needs_verification"
                reasoning.append(
                    f"{class_name} needs verification: score {score:.3f} above minimum {min_score} "
                    f"but below MEDIUM threshold {THRESHOLD_MEDIUM}"
                )
            break

    # 2. If not top class, check flood/drainage/pipeline for notable signal
    if incident_type is None:
        for class_name, score in [
            ("Flood",    flood_score),
            ("Drainage", drainage_score),
            ("Pipeline", pipeline_score),
        ]:
            min_score = CLASS_MIN_SCORE.get(class_name, 0.30)
            if score >= min_score:
                incident_type   = class_name
                incident_status = "needs_verification"
                reasoning.append(
                    f"{class_name} flagged for verification: score {score:.3f} >= minimum {min_score} "
                    f"(not top class, but notable signal)"
                )
                break

    # 3. No incident class meets threshold
    if incident_type is None:
        max_incident_score = max(
            flood_score, drainage_score, pipeline_score, pothole_score, accident_score
        )
        if max_incident_score < THRESHOLD_LOW:
            if normal_score >= CLASS_MIN_SCORE.get("Normal", 0.45):
                incident_type   = "Normal"
                incident_status = "normal"
                reasoning.append(
                    f"Normal confirmed: score {normal_score:.3f} dominant, "
                    f"no incident class above LOW threshold {THRESHOLD_LOW}"
                )
            else:
                incident_type   = "No Incident Detected"
                incident_status = "no_incident"
                reasoning.append(
                    f"Insufficient evidence: all scores below thresholds. "
                    f"Max incident score={max_incident_score:.3f}, Normal={normal_score:.3f}"
                )
        else:
            incident_type   = "Needs Verification"
            incident_status = "needs_verification"
            reasoning.append(
                f"Ambiguous: incident signal present (max={max_incident_score:.3f}) "
                f"but below minimum thresholds - manual review recommended"
            )

    confidence = scores.get(incident_type, scores.get(top_class, 0.0)) \
        if incident_type not in ("Normal", "No Incident Detected", "Needs Verification") \
        else scores.get(top_class, 0.0)

    severity = _compute_severity(incident_type, confidence)

    reasoning.append(f"Severity: {severity} (confidence={confidence:.3f})")
    reasoning.extend([
        f"All CLIP scores: " + ", ".join(
            f"{k}={v:.3f}" for k, v in sorted(scores.items(), key=lambda x: -x[1])
        )
    ])

    logger.info(
        "CLIP decision: incident=%s status=%s conf=%.3f severity=%s",
        incident_type, incident_status, confidence, severity,
    )

    return {
        "incident_type":   incident_type,
        "confidence":      round(confidence, 4),
        "incident_status": incident_status,
        "severity":        severity,
        "clip_scores":     {k: round(v, 4) for k, v in scores.items()},
        "clip_top_class":  top_class,
        "reasoning":       reasoning,
        "source":          "clip",
    }


def is_likely_accident(image_bytes: bytes, vehicle_conf: float) -> Tuple[bool, float]:
    """
    YOLO+CLIP combined accident verification.
    Returns (is_accident, combined_confidence).
    """
    try:
        result = classify_image(image_bytes)
    except Exception as exc:
        logger.warning("CLIP classification failed, defaulting to no-accident: %s", exc)
        return False, 0.0

    scores = result.get("all_scores", {})
    accident_score = scores.get("Accident", 0.0)
    normal_score   = scores.get("Normal",   0.0)
    flood_score    = scores.get("Flood",    0.0)
    top_class      = result.get("incident_type", "Normal")

    logger.info(
        "CLIP accident-check: accident=%.3f normal=%.3f flood=%.3f top=%s",
        accident_score, normal_score, flood_score, top_class,
    )

    # Veto if flood/normal/drainage/pipeline is clearly dominant
    if top_class in ("Flood", "Normal", "Drainage", "Pipeline") and accident_score < 0.35:
        logger.info("CLIP veto: top class '%s', refusing Accident.", top_class)
        return False, 0.0

    if accident_score <= normal_score + 0.05:
        logger.info(
            "CLIP veto: accident_score (%.3f) not > normal (%.3f) + margin.",
            accident_score, normal_score,
        )
        return False, 0.0

    combined_confidence = (accident_score * vehicle_conf) ** 0.5
    is_accident = combined_confidence >= 0.30
    logger.info("CLIP accident decision: %s (combined_conf=%.3f)", is_accident, combined_confidence)
    return is_accident, round(combined_confidence, 4)
