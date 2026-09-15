"""
AURA - Priority Engine Smoke Tests
Tests /api/prioritize with all three incident types at varying confidence levels.
Also tests the full /api/detect-and-prioritize pipeline.
"""

import io
import json
import sys
import urllib.request
from PIL import Image, ImageDraw

BASE = "http://127.0.0.1:5000"
PASS = "[PASS]"
FAIL = "[FAIL]"

errors = 0


def post_json(path: str, body) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def assert_equal(label, got, expected):
    global errors
    if got == expected:
        print(f"  {PASS} {label}: {got!r}")
    else:
        print(f"  {FAIL} {label}: expected {expected!r}, got {got!r}")
        errors += 1


def section(title):
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# /api/prioritize – single incident classification
# ─────────────────────────────────────────────────────────────────────────────

CASES = [
    (
        "Flood @ 0.94 -> P1 Critical",
        {"success": True, "incident_type": "Flood",    "confidence": 0.94, "source": "roboflow", "detected_objects": ["flood"], "message": "Test"},
        "P1", "Critical",
    ),
    (
        "Flood @ 0.55 -> P2 High",
        {"success": True, "incident_type": "Flood",    "confidence": 0.55, "source": "roboflow", "detected_objects": ["flood"], "message": "Test"},
        "P2", "High",
    ),
    (
        "Flood @ 0.20 -> P3 Medium",
        {"success": True, "incident_type": "Flood",    "confidence": 0.20, "source": "roboflow", "detected_objects": [],        "message": "Test"},
        "P3", "Medium",
    ),
    (
        "Accident @ 0.85 -> P1 Critical",
        {"success": True, "incident_type": "Accident", "confidence": 0.85, "source": "yolo26",   "detected_objects": [{"class": "car"}], "message": "Test"},
        "P1", "Critical",
    ),
    (
        "Accident @ 0.50 -> P2 High",
        {"success": True, "incident_type": "Accident", "confidence": 0.50, "source": "yolo26",   "detected_objects": [{"class": "truck"}], "message": "Test"},
        "P2", "High",
    ),
    (
        "Normal @ 0.00 -> P4 Low",
        {"success": True, "incident_type": "Normal",   "confidence": 0.00, "source": "none",     "detected_objects": [],        "message": "Test"},
        "P4", "Low",
    ),
]

section("/api/prioritize - single incident")
for label, payload, exp_priority, exp_severity in CASES:
    try:
        result = post_json("/api/prioritize", payload)
        inc = result["incidents"][0]
        assert_equal(f"{label} | priority", inc["priority"],  exp_priority)
        assert_equal(f"{label} | severity", inc["severity"],  exp_severity)
    except Exception as exc:
        print(f"  {FAIL} {label}: EXCEPTION - {exc}")
        errors += 1


# ─────────────────────────────────────────────────────────────────────────────
# Batch mode – output sorted P1 first
# ─────────────────────────────────────────────────────────────────────────────

section("/api/prioritize - batch mode (sorted output)")
batch = [
    {"success": True, "incident_type": "Normal",   "confidence": 0.00, "source": "none",     "detected_objects": [], "message": ""},
    {"success": True, "incident_type": "Flood",    "confidence": 0.92, "source": "roboflow", "detected_objects": [], "message": ""},
    {"success": True, "incident_type": "Accident", "confidence": 0.55, "source": "yolo26",   "detected_objects": [], "message": ""},
]
try:
    result = post_json("/api/prioritize", batch)
    priorities = [i["priority"] for i in result["incidents"]]
    assert_equal("Batch sorted order", priorities, ["P1", "P2", "P4"])
except Exception as exc:
    print(f"  {FAIL} Batch test EXCEPTION - {exc}")
    errors += 1


# ─────────────────────────────────────────────────────────────────────────────
# Resource allocation per priority tier
# ─────────────────────────────────────────────────────────────────────────────

section("Resource allocation per priority tier")


def check_resources(label, payload, expect_amb, expect_hosp, expect_resc):
    global errors
    try:
        result = post_json("/api/prioritize", payload)
        inc = result["incidents"][0]
        res = inc["resources"]
        # Not-dispatched entries have dispatched=False; real resources don't have it
        amb_disp  = res["ambulance"].get("dispatched", True)
        hosp_disp = res["hospital"].get("dispatched", True)
        resc_disp = res["rescue_crew"].get("dispatched", True)
        assert_equal(f"{label} ambulance dispatched",  amb_disp,  expect_amb)
        assert_equal(f"{label} hospital dispatched",   hosp_disp, expect_hosp)
        assert_equal(f"{label} rescue dispatched",     resc_disp, expect_resc)
    except Exception as exc:
        print(f"  {FAIL} {label}: EXCEPTION - {exc}")
        errors += 1


check_resources(
    "P1 Flood",
    {"success": True, "incident_type": "Flood",    "confidence": 0.90, "source": "roboflow", "detected_objects": [], "message": ""},
    True, True, True,
)
check_resources(
    "P2 Accident",
    {"success": True, "incident_type": "Accident", "confidence": 0.55, "source": "yolo26",  "detected_objects": [], "message": ""},
    True, True, True,
)
check_resources(
    "P3 Flood (low conf)",
    {"success": True, "incident_type": "Flood",    "confidence": 0.20, "source": "roboflow","detected_objects": [], "message": ""},
    False, False, True,
)
check_resources(
    "P4 Normal",
    {"success": True, "incident_type": "Normal",   "confidence": 0.00, "source": "none",    "detected_objects": [], "message": ""},
    False, False, False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: /api/detect-and-prioritize with a synthetic image
# ─────────────────────────────────────────────────────────────────────────────

section("/api/detect-and-prioritize - full pipeline (synthetic image)")

img = Image.new("RGB", (320, 240), color=(30, 60, 180))
d   = ImageDraw.Draw(img)
d.rectangle([40, 80, 280, 160], fill=(20, 20, 20))
buf = io.BytesIO()
img.save(buf, format="JPEG")
img_bytes = buf.getvalue()

boundary = "PRIORITY_TEST_BOUND"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="image"; filename="test.jpg"\r\n'
    f"Content-Type: image/jpeg\r\n\r\n"
).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(
    BASE + "/api/detect-and-prioritize",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)
try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())

    assert_equal("success flag",       result["success"],                      True)
    assert_equal("incidents is list",  isinstance(result["incidents"], list),  True)
    assert_equal("disclaimer present", "demo_disclaimer" in result,            True)
    assert_equal("evaluated_at set",   bool(result.get("evaluated_at")),       True)

    print("\n  First incident result:")
    print("  " + json.dumps(result["incidents"][0], indent=4).replace("\n", "\n  "))

except Exception as exc:
    print(f"  {FAIL} Full pipeline EXCEPTION - {exc}")
    errors += 1


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
if errors == 0:
    print(f"  {PASS} All tests passed!")
else:
    print(f"  {FAIL} {errors} test(s) failed.")
    sys.exit(1)
print("=" * 60 + "\n")
