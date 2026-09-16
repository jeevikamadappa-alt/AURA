# AURA – After Cloning: Getting Model Weights
# ============================================

## Why `yolo11n.pt` is not in this repo

The YOLO model weights file (`yolo11n.pt`) is ~5.4 MB and is excluded from
git tracking (via `.gitignore *.pt` rule) to keep the repository lightweight.

## Step 1: Download the model

After cloning, run this ONE command from the `backend/` directory:

```bash
# Option A – Python (works everywhere)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

The Ultralytics library will automatically download `yolo11n.pt` from the
official Ultralytics servers and cache it. The file will appear in your
current working directory.

OR download directly:

```bash
# Option B – curl (Linux/Mac/Git Bash)
curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -o backend/yolo11n.pt
```

Place the file at: `backend/yolo11n.pt`

## Step 2: Set up your `.env`

```bash
cd backend
cp .env.example .env
# Then edit .env and fill in your ROBOFLOW_API_KEY
```

## Step 3: Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

## Step 4: Start the server

```bash
# Windows
backend\start.bat

# PowerShell
backend\start.ps1

# Or directly
cd backend && python app.py
```

## Verifying the setup works

Open http://localhost:5000/test in your browser.
The terminal should show on startup:
  - `ROBOFLOW_API_KEY : rf_...xxxx` (masked, not empty)
  - `YOLO model loaded successfully`
  - `CLIP model loaded successfully`

If `ROBOFLOW_API_KEY` shows `<NOT SET>`, your `.env` is missing or not loaded.
