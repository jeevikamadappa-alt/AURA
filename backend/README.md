# AURA – Incident Detection Backend

> **Scope**: This service answers **"WHAT incident happened?"**
> It does NOT calculate priority, hospital availability, ambulance routing, or response time.
> Those responsibilities belong to the Priority Engine (separate module).

---

## Architecture

```
backend/
├── app.py                        # Flask entry point + API routes
├── config.py                     # Env-var config (zero hard-coded secrets)
├── detection/
│   ├── __init__.py
│   ├── incident_detector.py      # Core pipeline (Flood → Accident → Normal)
│   ├── roboflow_client.py        # Roboflow flood-detection wrapper
│   └── yolo_detector.py          # YOLO vehicle fallback detector
├── requirements.txt
├── .env.example                  # Safe-to-commit template
└── .gitignore
```

---

## Detection Pipeline

| Stage | Model | Trigger | Returns |
|-------|-------|---------|---------|
| 1 | Roboflow (custom flood model) | Always runs first | `incident_type: "Flood"` |
| 2 | Ultralytics YOLO (yolo11n.pt) | Flood not found | `incident_type: "Accident"` |
| 3 | None | Neither found | `incident_type: "Normal"` |

> **YOLO disclaimer**: YOLO detects *vehicle presence* (car, truck, bus, motorcycle)
> as an indirect accident-signal – not a direct accident classification.

---

## API

### `POST /api/detect`
Upload an image and receive incident classification.

**Request** – `multipart/form-data`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `image` | File | ✅ | jpg, jpeg, png, webp, bmp ≤ 16 MB |

**Response** – `application/json`

```json
{
  "success": true,
  "incident_type": "Flood | Accident | Normal",
  "confidence": 0.94,
  "source": "roboflow | yolo26 | none",
  "detected_objects": [],
  "message": "Human-readable explanation"
}
```

### `GET /health`
Liveness probe – returns `{"status": "ok"}`.

### `GET /test`
Browser-based upload form for quick manual testing.

---

## Setup

### 1. Create virtual environment

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy .env.example .env
# Edit .env and fill in:
#   ROBOFLOW_API_KEY=your_actual_key
#   ROBOFLOW_PROJECT_ID=your_project_id
#   ROBOFLOW_MODEL_VERSION=1
```

### 4. Run the server

```bash
python app.py
# Server starts at http://localhost:5000
```

Open **http://localhost:5000/test** in your browser to use the visual test form.

---

## cURL example

```bash
curl -X POST http://localhost:5000/api/detect \
  -F "image=@/path/to/your/image.jpg"
```

---

## Security

- `ROBOFLOW_API_KEY` is **never** logged, returned in responses, or hard-coded.
- `.env` is in `.gitignore` – only `.env.example` is committed.
- YOLO model weights (`.pt`) are also gitignored.

---

## Production

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
Set `FLASK_DEBUG=0` and use a reverse proxy (nginx) in front of gunicorn.
