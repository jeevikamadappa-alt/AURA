"""Quick end-to-end smoke test for POST /api/detect"""
import io
import json
import urllib.request

from PIL import Image, ImageDraw

# --- Build a synthetic test image ---
img = Image.new("RGB", (320, 240), color=(100, 149, 237))
d = ImageDraw.Draw(img)
d.rectangle([60, 80, 260, 160], fill=(50, 50, 50))
buf = io.BytesIO()
img.save(buf, format="JPEG")
img_bytes = buf.getvalue()
print(f"Test image: {len(img_bytes)} bytes")

# --- POST multipart/form-data ---
boundary = "AURA_TEST_BOUNDARY"
crlf = b"\r\n"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="image"; filename="test.jpg"\r\n'
    f"Content-Type: image/jpeg\r\n\r\n"
).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(
    "http://127.0.0.1:5000/api/detect",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())

print(json.dumps(result, indent=2))
