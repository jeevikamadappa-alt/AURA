"""
AURA Backend – Flask Application

Endpoints
---------
GET  /health                        → liveness check
POST /api/detect                    → incident detection (image upload → JSON)
POST /api/prioritize                → priority engine  (detection JSON → priority JSON)
POST /api/detect-and-prioritize     → full pipeline    (image upload → priority JSON)
GET  /test                          → browser-based test UI
"""

import logging
import os
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Bootstrap path so imports resolve regardless of CWD ───────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import config  # noqa: E402  (must come after path fix)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Allow cross-origin requests from the AURA frontend (adjust origins in prod)
CORS(app, resources={r"/api/*": {"origins": "*"}})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _allowed_file(filename: str) -> bool:
    """Return True if the file extension is in the allow-list."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS


def _error(message: str, status: int = 400) -> tuple:
    """Return a standardised JSON error response."""
    return (
        jsonify(
            {
                "success": False,
                "incident_type": None,
                "confidence": 0.0,
                "source": None,
                "detected_objects": [],
                "message": message,
            }
        ),
        status,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    """Liveness probe – returns 200 when the service is up."""
    return jsonify({"status": "ok", "service": "AURA Incident Detection"}), 200


@app.post("/api/detect")
def detect():
    """
    POST /api/detect

    Accepts a multipart/form-data request with a single image field named
    'image'. Runs the incident-detection pipeline and returns JSON.

    Returns
    -------
    200  – detection result (success or normal)
    400  – bad request (missing / invalid file)
    500  – internal detection failure
    """
    # ── Validate request ──────────────────────────────────────────────────────
    if "image" not in request.files:
        return _error("No image field found in the request. Use field name 'image'.")

    file = request.files["image"]

    if file.filename == "" or file.filename is None:
        return _error("No file selected.")

    if not _allowed_file(file.filename):
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        return _error(
            f"Unsupported file type. Allowed extensions: {allowed}."
        )

    # ── Read image bytes ──────────────────────────────────────────────────────
    try:
        image_bytes = file.read()
    except Exception:
        logger.exception("Failed to read uploaded file.")
        return _error("Could not read the uploaded file.", 400)

    if len(image_bytes) == 0:
        return _error("Uploaded file is empty.")

    # ── Quick sanity-check: verify it's a parseable image ─────────────────────
    try:
        import io
        from PIL import Image

        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        return _error("Uploaded file is not a valid image.")

    # ── Run detection pipeline ────────────────────────────────────────────────
    try:
        from detection.incident_detector import detect_incident

        result = detect_incident(image_bytes)
        return jsonify(result), 200

    except RuntimeError as exc:
        # YOLO or pipeline-level failure
        logger.error("Detection pipeline error: %s", exc)
        return (
            jsonify(
                {
                    "success": False,
                    "incident_type": None,
                    "confidence": 0.0,
                    "source": None,
                    "detected_objects": [],
                    "message": str(exc),
                }
            ),
            500,
        )
    except Exception:
        logger.exception("Unexpected error in /api/detect.")
        return _error("An unexpected error occurred during detection.", 500)


@app.get("/test")
def test_ui():
    """
    Multi-image upload form.
    Accepts up to 6 images, runs /api/detect-and-prioritize on each,
    then displays all results in one priority-ranked table (P1 first).
    Drop-zone is hidden after images are selected; a reset link restores it.
    """
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AURA &ndash; Multi-Incident Detection &amp; Priority</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#0a0d13;--surface:#111620;--card:#161d2b;--border:#1e2a3a;
  --accent:#4f9cf9;--accent2:#38d9a9;
  --text:#e8eef7;--muted:#6b7fa3;--font:'Inter',system-ui,sans-serif;
  --p1:#ef4444;--p2:#f59e0b;--p3:#3b82f6;--p4:#22c55e;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font);
  min-height:100vh;display:flex;flex-direction:column;
  align-items:center;padding:2rem 1rem 4rem}
.hdr{width:100%;max-width:960px;display:flex;align-items:center;
  gap:.75rem;margin-bottom:2rem}
.hdr-icon{font-size:2rem}
.hdr h1{font-size:1.6rem;font-weight:800;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hdr p{font-size:.82rem;color:var(--muted);margin-top:.15rem}
.main{width:100%;max-width:960px}

/* Drop zone */
#upload-phase{width:100%}
.drop-zone{border:2px dashed var(--border);border-radius:16px;
  padding:3.5rem 2rem;text-align:center;cursor:pointer;
  transition:border-color .25s,background .25s;position:relative;
  background:var(--card)}
.drop-zone.drag-over{border-color:var(--accent);background:rgba(79,156,249,.06)}
.drop-zone input[type=file]{position:absolute;inset:0;opacity:0;
  cursor:pointer;width:100%;height:100%}
.dz-icon{font-size:3rem;margin-bottom:.75rem}
.dz-title{font-size:1.1rem;font-weight:600;margin-bottom:.3rem}
.dz-sub{color:var(--muted);font-size:.84rem}
.dz-sub span{color:var(--accent);text-decoration:underline}
.dz-limit{margin-top:.8rem;font-size:.74rem;color:var(--muted);
  background:rgba(79,156,249,.1);display:inline-block;
  padding:.2rem .75rem;border-radius:999px}

/* Preview phase */
#preview-phase{display:none;width:100%}
.preview-header{display:flex;align-items:center;justify-content:space-between;
  margin-bottom:1rem}
.preview-header h2{font-size:.95rem;font-weight:600}
.reset-link{font-size:.8rem;color:var(--accent);cursor:pointer;
  text-decoration:none;background:none;border:none;padding:.3rem .6rem;
  border-radius:6px;border:1px solid rgba(79,156,249,.3);
  transition:background .15s}
.reset-link:hover{background:rgba(79,156,249,.1)}
.thumb-grid{display:grid;
  grid-template-columns:repeat(auto-fill,minmax(140px,1fr));
  gap:.75rem;margin-bottom:1.5rem}
.thumb-item{border-radius:10px;overflow:hidden;border:1px solid var(--border);
  background:var(--card);position:relative}
.thumb-item img{width:100%;height:100px;object-fit:cover;display:block}
.thumb-label{font-size:.72rem;color:var(--muted);padding:.35rem .5rem;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.thumb-rm{position:absolute;top:5px;right:5px;
  background:rgba(0,0,0,.7);border:none;color:#fff;border-radius:50%;
  width:20px;height:20px;font-size:.68rem;cursor:pointer;
  display:flex;align-items:center;justify-content:center}
.thumb-rm:hover{background:rgba(239,68,68,.8)}

/* Button */
.btn-primary{width:100%;padding:.9rem;
  background:linear-gradient(135deg,var(--accent) 0%,#2d7dd2 100%);
  color:#fff;font-weight:700;font-size:1rem;border:none;
  border-radius:10px;cursor:pointer;letter-spacing:.02em;
  transition:opacity .2s,transform .1s}
.btn-primary:hover{opacity:.88}
.btn-primary:active{transform:scale(.98)}
.btn-primary:disabled{opacity:.3;cursor:not-allowed}

/* Spinner */
#spinner{display:none;text-align:center;margin:2rem 0}
.spin-ring{display:inline-block;width:36px;height:36px;
  border:3px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;animation:spin .8s linear infinite;margin-bottom:.6rem}
@keyframes spin{to{transform:rotate(360deg)}}
#spinner-msg{color:var(--muted);font-size:.88rem}

/* Results */
#result-section{display:none;width:100%;margin-top:2rem}
.section-hdr{display:flex;align-items:center;gap:.6rem;margin-bottom:1rem}
.section-hdr h2{font-size:1rem;font-weight:700}
.count-pill{background:var(--accent);color:#fff;font-size:.7rem;
  font-weight:800;padding:.12rem .55rem;border-radius:999px}

/* Priority table */
.tbl-wrap{overflow-x:auto;border-radius:12px;border:1px solid var(--border);
  margin-bottom:1.5rem}
table.pri{width:100%;border-collapse:collapse;font-size:.83rem}
.pri th{background:var(--card);color:var(--muted);font-weight:600;
  font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;
  padding:.6rem 1rem;border-bottom:1px solid var(--border);
  text-align:left;white-space:nowrap}
.pri td{padding:.65rem 1rem;border-bottom:1px solid var(--border);
  vertical-align:middle}
.pri tr:last-child td{border-bottom:none}
.pri tr:hover td{background:rgba(255,255,255,.02)}
.tbl-thumb{width:46px;height:37px;object-fit:cover;
  border-radius:5px;border:1px solid var(--border)}

/* Row left-border by tier */
.row-p1 td:first-child{border-left:3px solid var(--p1)}
.row-p2 td:first-child{border-left:3px solid var(--p2)}
.row-p3 td:first-child{border-left:3px solid var(--p3)}
.row-p4 td:first-child{border-left:3px solid var(--p4)}

/* Badges */
.badge{display:inline-flex;align-items:center;padding:.16rem .6rem;
  border-radius:999px;font-size:.7rem;font-weight:800;
  letter-spacing:.05em;white-space:nowrap}
.b-p1{background:rgba(239,68,68,.2);color:#fca5a5;border:1px solid rgba(239,68,68,.4)}
.b-p2{background:rgba(245,158,11,.2);color:#fcd34d;border:1px solid rgba(245,158,11,.4)}
.b-p3{background:rgba(59,130,246,.2);color:#93c5fd;border:1px solid rgba(59,130,246,.4)}
.b-p4{background:rgba(34,197,94,.2);color:#86efac;border:1px solid rgba(34,197,94,.4)}
.b-flood{background:rgba(59,130,246,.18);color:#93c5fd;border:1px solid rgba(59,130,246,.35)}
.b-accident{background:rgba(249,115,22,.18);color:#fdba74;border:1px solid rgba(249,115,22,.35)}
.b-drought{background:rgba(120,53,15,.35);color:#fbbf24;border:1px solid rgba(120,53,15,.5)}
.b-drainage{background:rgba(20,184,166,.2);color:#5eead4;border:1px solid rgba(20,184,166,.4)}
.b-pipeleak{background:rgba(168,85,247,.2);color:#d8b4fe;border:1px solid rgba(168,85,247,.4)}
.b-normal{background:rgba(34,197,94,.15);color:#86efac;border:1px solid rgba(34,197,94,.3)}
.b-err{background:rgba(239,68,68,.15);color:#fca5a5;border:1px solid rgba(239,68,68,.3)}
.tbl-fname{font-weight:600;font-size:.8rem;color:var(--text);max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* Resource mini-chips */
.chip{display:inline-block;font-size:.68rem;padding:.1rem .42rem;
  border-radius:5px;margin:.1rem .08rem 0 0;
  background:rgba(255,255,255,.05);border:1px solid var(--border)}
.chip-ok{border-color:rgba(34,197,94,.4);color:#86efac}
.chip-busy{border-color:rgba(245,158,11,.4);color:#fcd34d}
.chip-no{color:var(--muted)}

/* Detail accordion cards */
.details-wrap{display:grid;gap:.85rem}
.d-card{background:var(--card);border:1px solid var(--border);
  border-radius:12px;overflow:hidden}
.d-hdr{display:flex;align-items:center;gap:.55rem;
  padding:.72rem 1rem;cursor:pointer;user-select:none;
  border-bottom:1px solid transparent;transition:background .15s}
.d-hdr:hover{background:rgba(255,255,255,.03)}
.d-hdr.open{border-bottom-color:var(--border)}
.d-chev{margin-left:auto;color:var(--muted);font-size:.75rem;
  transition:transform .2s}
.d-chev.open{transform:rotate(180deg)}
.d-body{display:none;padding:1rem;
  display:none;grid-template-columns:1fr 1fr;gap:.6rem}
.d-body.open{display:grid}
.dl dt{color:var(--muted);font-size:.7rem;font-weight:600;margin-bottom:.18rem;
  text-transform:uppercase;letter-spacing:.05em}
.dl dd{font-size:.82rem;color:var(--text);line-height:1.4}
.span2{grid-column:1/-1}
.r-table{width:100%;border-collapse:collapse;font-size:.77rem}
.r-table th{color:var(--muted);font-weight:600;font-size:.68rem;
  padding:.26rem .55rem;border-bottom:1px solid var(--border);text-align:left}
.r-table td{padding:.3rem .55rem;border-bottom:1px solid rgba(30,42,58,.7)}
.r-table tr:last-child td{border-bottom:none}
.st-ok{color:#86efac;font-weight:600}
.st-busy{color:#fcd34d;font-weight:600}
.st-no{color:var(--muted)}

/* Disclaimer */
.disclaimer{font-size:.72rem;color:var(--muted);margin-top:1.5rem;
  background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.2);
  border-radius:8px;padding:.6rem .95rem;line-height:1.5}
details.raw summary{cursor:pointer;color:var(--muted);font-size:.78rem;
  user-select:none;margin-top:1.2rem;padding:.25rem 0;display:block}
details.raw summary:hover{color:var(--accent)}
pre{background:#070b11;border:1px solid var(--border);border-radius:8px;
  padding:1rem;font-size:.72rem;overflow-x:auto;line-height:1.6;
  color:#7a90b8;white-space:pre-wrap;word-break:break-all;margin-top:.4rem}
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-icon">&#128752;</div>
  <div>
    <h1>AURA Detection</h1>
    <p>Upload up to 6 images &mdash; each is analysed and all results are ranked by priority</p>
  </div>
</div>

<div class="main">

  <!-- ── Upload phase ── -->
  <div id="upload-phase">
    <div class="drop-zone" id="drop-zone">
      <input type="file" id="image-input" multiple
             accept=".jpg,.jpeg,.png,.webp,.bmp">
      <div class="dz-icon">&#128444;&#65039;</div>
      <div class="dz-title">Drop images here</div>
      <p class="dz-sub">or <span>browse your files</span></p>
      <span class="dz-limit">Up to 6 images &middot; JPG &middot; PNG &middot; WebP &middot; BMP</span>
    </div>
  </div>

  <!-- ── Preview phase ── -->
  <div id="preview-phase">
    <div class="preview-header">
      <h2 id="preview-title">0 images selected</h2>
      <button class="reset-link" id="reset-btn">&#8593; Upload different images</button>
    </div>
    <div class="thumb-grid" id="thumb-grid"></div>
    <button class="btn-primary" id="submit-btn" disabled>
      &#9889;&nbsp; Detect &amp; Prioritize All
    </button>
  </div>

  <!-- ── Spinner ── -->
  <div id="spinner">
    <div class="spin-ring"></div>
    <div id="spinner-msg">Analysing&hellip;</div>
  </div>

  <!-- ── Results ── -->
  <div id="result-section">
    <div class="section-hdr">
      <h2>Priority Ranking</h2>
      <span class="count-pill" id="result-count">0</span>
    </div>

    <div class="tbl-wrap">
      <table class="pri">
        <thead><tr>
          <th>Rank</th><th>Image &amp; File</th><th>Incident</th>
          <th>Priority</th><th>Severity</th>
          <th>Response target</th><th>Confidence</th><th>Resources</th>
        </tr></thead>
        <tbody id="result-tbody"></tbody>
      </table>
    </div>

    <div class="details-wrap" id="detail-cards"></div>
    <div class="disclaimer" id="disclaimer-box" style="display:none"></div>

    <details class="raw">
      <summary>&#128196; Raw JSON responses</summary>
      <pre id="raw-json"></pre>
    </details>
  </div>

</div>

<script>
/* ── DOM ── */
const uploadPhase  = document.getElementById('upload-phase');
const previewPhase = document.getElementById('preview-phase');
const dropZone     = document.getElementById('drop-zone');
const imageInput   = document.getElementById('image-input');
const previewTitle = document.getElementById('preview-title');
const thumbGrid    = document.getElementById('thumb-grid');
const resetBtn     = document.getElementById('reset-btn');
const submitBtn    = document.getElementById('submit-btn');
const spinner      = document.getElementById('spinner');
const spinnerMsg   = document.getElementById('spinner-msg');
const resultSec    = document.getElementById('result-section');
const resultTbody  = document.getElementById('result-tbody');
const resultCount  = document.getElementById('result-count');
const detailCards  = document.getElementById('detail-cards');
const disclaimerBox= document.getElementById('disclaimer-box');
const rawJson      = document.getElementById('raw-json');

const MAX = 6;
// Each entry: { file: File, thumbUrl: string, id: number }
let selectedFiles = [];
let _nextId = 0;

/* ── Lookup maps ── */
const TIER = {P1:'b-p1',P2:'b-p2',P3:'b-p3',P4:'b-p4'};
const ITYPE= {flood:'b-flood',accident:'b-accident',drought:'b-drought',drainage:'b-drainage',pipeleak:'b-pipeleak',normal:'b-normal'};
const RROW = {P1:'row-p1',P2:'row-p2',P3:'row-p3',P4:'row-p4'};
const PORD = {P1:1,P2:2,P3:3,P4:4};
const tb   = p => TIER[p]||'b-err';
const ib   = t => ITYPE[(t||'').toLowerCase()]||'b-err';
const rr   = p => RROW[p]||'';

function avSt(v){
  if(!v) return '<span class="st-no">&#8212;</span>';
  const s=v.toLowerCase();
  if(['available','open'].includes(s)) return '<span class="st-ok">'+v+'</span>';
  if(['busy','limited'].includes(s))   return '<span class="st-busy">'+v+'</span>';
  return '<span class="st-no">'+v+'</span>';
}
function chip(r){
  if(!r||r.dispatched===false) return '<span class="chip chip-no">&#8212;</span>';
  const a=(r.availability||'').toLowerCase();
  const c=(['available','open'].includes(a))?'chip-ok':(['busy','limited'].includes(a))?'chip-busy':'chip-no';
  const eta=r.estimated_response_minutes!=null?r.estimated_response_minutes+'min':'?';
  return '<span class="chip '+c+'">'+(r.name||'?')+' &bull; '+eta+'</span>';
}
function rtRow(icon,label,r){
  if(!r||r.dispatched===false)
    return '<tr><td>'+icon+' '+label+'</td><td colspan="4" class="st-no">Not dispatched</td></tr>';
  return '<tr><td>'+icon+' '+label+'</td><td>'+(r.name||'&#8212;')+'</td><td>'+avSt(r.availability)+'</td>'
    +'<td>'+(r.distance_km!=null?r.distance_km+' km':'&#8212;')+'</td>'
    +'<td>'+(r.estimated_response_minutes!=null?r.estimated_response_minutes+' min':'&#8212;')+'</td></tr>';
}

/* ── File handling ── */
function handleFiles(files){
  const arr = Array.from(files);
  if(!arr.length) return;

  // Revoke previous blob URLs
  selectedFiles.forEach(e => URL.revokeObjectURL(e.thumbUrl));

  // Store new entry objects with unique id
  selectedFiles = arr.slice(0, MAX).map(f => ({
    id: 'img_' + (++_nextId) + '_' + Math.random().toString(36).slice(2, 7),
    file: f,
    fileName: f.name,
    thumbUrl: URL.createObjectURL(f)
  }));

  renderPreviews();
}

function renderPreviews(){
  thumbGrid.innerHTML = '';
  selectedFiles.forEach(entry => {
    const el = document.createElement('div');
    el.className = 'thumb-item';
    el.innerHTML = '<img src="' + entry.thumbUrl + '" alt="' + entry.fileName + '">'
      + '<div class="thumb-label" title="' + entry.fileName + '">' + entry.fileName + '</div>'
      + '<button class="thumb-rm" data-id="' + entry.id + '" title="Remove">&#10005;</button>';
    thumbGrid.appendChild(el);
  });

  thumbGrid.querySelectorAll('.thumb-rm').forEach(b => {
    b.addEventListener('click', e => {
      const id = e.currentTarget.dataset.id;
      const idx = selectedFiles.findIndex(en => en.id === id);
      if(idx !== -1){
        URL.revokeObjectURL(selectedFiles[idx].thumbUrl);
        selectedFiles.splice(idx, 1);
      }
      if(!selectedFiles.length){
        showUpload();
        return;
      }
      renderPreviews();
    });
  });

  previewTitle.textContent = selectedFiles.length + ' image' + (selectedFiles.length > 1 ? 's' : '') + ' selected';
  submitBtn.disabled = selectedFiles.length === 0;
  uploadPhase.style.display = 'none';
  previewPhase.style.display = 'block';
}

function showUpload(){
  selectedFiles.forEach(e => URL.revokeObjectURL(e.thumbUrl));
  selectedFiles = [];
  uploadPhase.style.display = 'block';
  previewPhase.style.display = 'none';
  resultSec.style.display = 'none';
  imageInput.value = '';
}

imageInput.addEventListener('change',()=>handleFiles(imageInput.files));
dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.classList.add('drag-over');});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop',e=>{
  e.preventDefault();dropZone.classList.remove('drag-over');
  handleFiles(e.dataTransfer.files);
});
resetBtn.addEventListener('click',showUpload);

/* ── Submit (Concurrent Multi-Image Analysis) ── */
submitBtn.addEventListener('click', async () => {
  if(!selectedFiles.length) return;
  submitBtn.disabled = true;
  spinner.style.display = 'block';
  resultSec.style.display = 'none';

  // Snapshot the current entries so UI changes cannot desync ongoing analysis
  const snapshot = selectedFiles.map(e => ({...e}));
  let completed = 0;
  spinnerMsg.textContent = 'Analysing ' + snapshot.length + ' image' + (snapshot.length > 1 ? 's' : '') + ' concurrently...';

  // Process all images concurrently for maximum speed
  const analysisPromises = snapshot.map(async (entry, idx) => {
    const fd = new FormData();
    fd.append('image', entry.file);
    fd.append('image_id', entry.id);
    fd.append('client_file_name', entry.fileName);

    try {
      const resp = await fetch('/api/detect-and-prioritize', {
        method: 'POST',
        body: fd
      });
      const json = await resp.json();
      completed++;
      spinnerMsg.textContent = 'Analysed ' + completed + ' of ' + snapshot.length + ' images...';

      let inc;
      if (json.success && Array.isArray(json.incidents) && json.incidents.length > 0) {
        inc = json.incidents[0];
      } else if (json.success) {
        // Detection returned success but no incidents array — use direct fields
        inc = {
          incident_type: json.incident_type || 'No Incident Detected',
          confidence: json.confidence || 0,
          priority: json.priority || 'P4',
          severity: json.severity || 'None',
          response_target: json.response_target || '3-7 days',
          recommended_action: json.recommended_action || json.message || 'No incident detected',
          detected_objects: json.detected_objects || [],
          source: json.source || 'none',
          detection_message: json.message || '',
          incident_status: json.incident_status || 'no_incident',
          clip_scores: json.clip_scores || {},
          reasoning: json.reasoning || [],
          yolo_raw: json.yolo_raw || {},
          resources: { ambulance: {dispatched: false}, hospital: {dispatched: false}, rescue_crew: {dispatched: false} }
        };
      } else {
        inc = {
          incident_type: 'Error',
          confidence: 0,
          priority: 'ERR',
          severity: 'Error',
          response_target: '--',
          recommended_action: json.message || 'Detection failed',
          detected_objects: [],
          source: 'none',
          detection_message: json.message || '',
          incident_status: 'error',
          clip_scores: {},
          reasoning: [],
          yolo_raw: {},
          resources: { ambulance: {dispatched: false}, hospital: {dispatched: false}, rescue_crew: {dispatched: false} }
        };
      }

      // Explicitly bundle the result with the EXACT entry's metadata
      return {
        id: entry.id,
        fileName: entry.fileName,
        thumbUrl: entry.thumbUrl,
        origIdx: idx + 1,
        inc: inc,
        disclaimer: json.demo_disclaimer || '',
        raw: { file: entry.fileName, response: json }
      };

    } catch (err) {
      completed++;
      spinnerMsg.textContent = 'Analysed ' + completed + ' of ' + snapshot.length + ' images...';
      return {
        id: entry.id,
        fileName: entry.fileName,
        thumbUrl: entry.thumbUrl,
        origIdx: idx + 1,
        inc: {
          incident_type: 'Error',
          confidence: 0,
          priority: 'ERR',
          severity: 'Error',
          response_target: '--',
          recommended_action: 'Network / server error: ' + err.message,
          detected_objects: [],
          source: 'none',
          detection_message: '',
          resources: {}
        },
        disclaimer: '',
        raw: { file: entry.fileName, error: err.message }
      };
    }
  });

  const analyzedItems = await Promise.all(analysisPromises);
  spinner.style.display = 'none';

  // Sort by Priority P1 -> P4, then by confidence descending
  analyzedItems.sort((a, b) => {
    const pa = PORD[a.inc.priority] || 99, pb = PORD[b.inc.priority] || 99;
    return pa !== pb ? pa - pb : (b.inc.confidence || 0) - (a.inc.confidence || 0);
  });

  const rawAll = analyzedItems.map(item => item.raw);
  renderResults(analyzedItems, rawAll);
});

/* ── Render ── */
function renderResults(rows, rawAll){
  resultTbody.innerHTML = '';
  detailCards.innerHTML = '';
  resultCount.textContent = rows.length;

  let disc = '';
  rows.forEach((item, rank) => {
    const {inc, thumbUrl, fileName, origIdx} = item;
    const res = inc.resources || {};
    const conf = ((inc.confidence || 0) * 100).toFixed(1);

    /* Table row */
    const tr = document.createElement('tr');
    tr.className = rr(inc.priority);
    tr.innerHTML =
      '<td style="color:var(--muted);font-weight:700;font-size:.82rem">#' + (rank + 1) + '</td>'
      + '<td>'
      +   '<div style="display:flex;align-items:center;gap:.65rem">'
      +     '<img class="tbl-thumb" src="' + thumbUrl + '" alt="' + fileName + '">'
      +     '<span class="tbl-fname" title="' + fileName + '">' + fileName + '</span>'
      +   '</div>'
      + '</td>'
      + '<td><span class="badge ' + ib(inc.incident_type) + '">' + (inc.incident_type || '?') + '</span></td>'
      + '<td><span class="badge ' + tb(inc.priority) + '">' + (inc.priority || '?') + '</span></td>'
      + '<td style="color:var(--muted);font-size:.8rem">' + (inc.severity || '--') + '</td>'
      + '<td style="font-size:.8rem"><strong>' + (inc.response_target || '--') + '</strong></td>'
      + '<td style="font-size:.8rem">' + conf + '%</td>'
      + '<td style="font-size:.74rem">' + chip(res.ambulance) + ' ' + chip(res.hospital) + ' ' + chip(res.rescue_crew) + '</td>';
    resultTbody.appendChild(tr);

    /* Detail accordion card */
    const objs = Array.isArray(inc.detected_objects) && inc.detected_objects.length
      ? inc.detected_objects.map(o => typeof o === 'object' ? (o.class || JSON.stringify(o)) : String(o)).join(', ')
      : 'None';
    const cid = 'dc-' + rank;
    const card = document.createElement('div');
    card.className = 'd-card';

    // Build CLIP scores HTML
    const clipScores = inc.clip_scores || {};
    const clipEntries = Object.entries(clipScores).sort((a,b) => b[1]-a[1]);
    let clipHtml = '';
    if (clipEntries.length > 0) {
      clipHtml = '<div class="clip-scores"><div style="font-size:.72rem;color:var(--muted);font-weight:600;margin-bottom:.4rem;text-transform:uppercase;letter-spacing:.05em">CLIP Semantic Scores</div>';
      clipEntries.forEach(([cls, score]) => {
        const pct = (score * 100).toFixed(1);
        const isTop = cls === inc.clip_top_class || cls === inc.incident_type;
        clipHtml += '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.25rem">'
          + '<span style="font-size:.72rem;min-width:90px;color:' + (isTop ? 'var(--accent)' : 'var(--muted)') + ';font-weight:' + (isTop ? '700' : '400') + '">' + cls + '</span>'
          + '<div style="flex:1;height:6px;background:var(--border);border-radius:3px"><div style="width:' + pct + '%;height:100%;background:' + (isTop ? 'var(--accent)' : 'var(--muted)') + ';border-radius:3px;transition:width .5s"></div></div>'
          + '<span style="font-size:.72rem;color:' + (isTop ? 'var(--accent)' : 'var(--muted)') + ';min-width:38px;text-align:right">' + pct + '%</span>'
          + '</div>';
      });
      clipHtml += '</div>';
    }

    // Build reasoning HTML
    const reasoning = Array.isArray(inc.reasoning) ? inc.reasoning : [];
    let reasoningHtml = '';
    if (reasoning.length > 0) {
      reasoningHtml = '<dl class="dl span2"><dt>Detection Reasoning</dt><dd><ul style="margin:0;padding-left:1.2rem;font-size:.76rem;opacity:.85">'
        + reasoning.map(r => '<li>' + r + '</li>').join('')
        + '</ul></dd></dl>';
    }

    // Build YOLO section
    const yoloRaw = inc.yolo_raw || {};
    const yoloDetCount = yoloRaw.total_detections || 0;
    const yoloDetections = yoloRaw.all_detections || [];
    const yoloFloodHints = yoloRaw.flood_relevant_detections || [];
    const yoloModelPath = yoloRaw.model_path || 'yolo11n.pt';
    let yoloHtml = '<dl class="dl span2"><dt>YOLO Model (Raw Detections)</dt><dd>';
    yoloHtml += '<div style="font-size:.72rem;color:var(--muted);margin-bottom:.3rem">'
      + 'Model: <strong>' + yoloModelPath + '</strong> — COCO-pretrained (80 classes: person, car, truck…)<br>'
      + '<span style="color:#f59e0b">⚠ This YOLO model is NOT trained to detect floods, drainage, or pipeline issues.</span>'
      + '</div>';
    if (yoloDetCount === 0) {
      yoloHtml += '<span style="font-size:.76rem;opacity:.7">No YOLO detections above conf=0.10 threshold.</span>';
    } else {
      yoloHtml += '<span style="font-size:.76rem">' + yoloDetCount + ' detection(s) found:</span><br>';
      yoloDetections.slice(0, 8).forEach(d => {
        const acc = d.accepted ? '✓' : '✗';
        const col = d.accepted ? 'var(--accent2)' : 'var(--muted)';
        yoloHtml += '<div style="font-size:.72rem;color:' + col + '">'
          + acc + ' ' + d.class_name + ' (id=' + d.class_id + ') conf=' + (d.confidence*100).toFixed(0) + '% area=' + (d.area_ratio*100).toFixed(1) + '%'
          + '</div>';
      });
      if (yoloDetections.length > 8) yoloHtml += '<div style="font-size:.72rem;opacity:.6">...and ' + (yoloDetections.length-8) + ' more</div>';
    }
    if (yoloFloodHints.length > 0) {
      yoloHtml += '<div style="font-size:.72rem;color:var(--accent);margin-top:.3rem">Flood context hint: ' + yoloFloodHints.join(', ') + ' detected</div>';
    }
    yoloHtml += '</dd></dl>';

    // Status badge
    const statusMap = {
      confirmed: {c:'#22c55e',t:'Confirmed'},
      needs_verification: {c:'#f59e0b',t:'Needs Verification'},
      no_incident: {c:'#6b7fa3',t:'No Incident'},
      normal: {c:'#6b7fa3',t:'Normal'},
      error: {c:'#ef4444',t:'Error'},
    };
    const st = statusMap[inc.incident_status] || {c:'#6b7fa3',t:inc.incident_status||'Unknown'};
    const statusBadge = '<span style="display:inline-block;padding:.15rem .5rem;border-radius:4px;font-size:.7rem;font-weight:700;background:' + st.c + '22;color:' + st.c + ';border:1px solid ' + st.c + '55">' + st.t + '</span>';

    card.innerHTML =
      '<div class="d-hdr" id="h-' + cid + '">'
        + '<span style="font-weight:700;font-size:.8rem;color:var(--muted);min-width:24px">#' + (rank + 1) + '</span>'
        + '<img class="tbl-thumb" style="width:34px;height:28px" src="' + thumbUrl + '" alt="' + fileName + '">'
        + '<span class="badge ' + tb(inc.priority) + '">' + (inc.priority || '?') + '</span>'
        + '<span class="badge ' + ib(inc.incident_type) + '">' + (inc.incident_type || '?') + '</span>'
        + statusBadge
        + '<span style="font-size:.84rem;font-weight:600;color:var(--text)">' + fileName + '</span>'
        + '<span class="d-chev" id="cv-' + cid + '">&#9660;</span>'
      + '</div>'
      + '<div class="d-body" id="b-' + cid + '">'
        + '<dl class="dl"><dt>Confidence</dt><dd>' + conf + '% via ' + (inc.source || 'none') + '</dd></dl>'
        + '<dl class="dl"><dt>Incident Status</dt><dd>' + statusBadge + '</dd></dl>'
        + '<dl class="dl"><dt>Response target</dt><dd><strong>' + (inc.response_target || '--') + '</strong></dd></dl>'
        + '<dl class="dl span2"><dt>Recommended action</dt><dd>' + (inc.recommended_action || '--') + '</dd></dl>'
        + '<dl class="dl span2"><dt>Detected objects (YOLO vehicle classes)</dt><dd>' + objs + '</dd></dl>'
        + '<dl class="dl span2"><dt>Detection note</dt><dd style="font-style:italic;opacity:.75;font-size:.8rem">' + (inc.detection_message || '--') + '</dd></dl>'
        + (clipHtml ? '<dl class="dl span2"><dt></dt><dd>' + clipHtml + '</dd></dl>' : '')
        + reasoningHtml
        + yoloHtml
        + '<div class="span2"><table class="r-table">'
          + '<thead><tr><th>Resource</th><th>Name</th><th>Status</th><th>Distance</th><th>ETA</th></tr></thead>'
          + '<tbody>'
            + rtRow('&#128657;', 'Ambulance', res.ambulance)
            + rtRow('&#127973;', 'Hospital', res.hospital)
            + rtRow('&#128736;&#65039;', 'Rescue Crew', res.rescue_crew)
          + '</tbody></table></div>'
      + '</div>';
    detailCards.appendChild(card);

    document.getElementById('h-' + cid).addEventListener('click', () => {
      const body = document.getElementById('b-' + cid);
      const chev = document.getElementById('cv-' + cid);
      const hdr = document.getElementById('h-' + cid);
      const open = body.classList.toggle('open');
      hdr.classList.toggle('open', open);
      chev.classList.toggle('open', open);
    });

    if(!disc && item.disclaimer) disc = item.disclaimer;
  });

  if(disc){
    disclaimerBox.textContent = '\u26a0\ufe0f ' + disc;
    disclaimerBox.style.display = 'block';
  } else {
    disclaimerBox.style.display = 'none';
  }

  rawJson.textContent = JSON.stringify(rawAll, null, 2);
  resultSec.style.display = 'block';
  submitBtn.disabled = false;
}
</script>
</body>
</html>"""
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}




# ─────────────────────────────────────────────────────────────────────────────
# Priority Engine endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/api/prioritize")
def prioritize():
    """
    POST /api/prioritize

    Accepts the JSON output of /api/detect (or a list of such objects for
    batch mode) and returns priority classifications with mock resource
    allocations.

    Does NOT re-analyse any image.

    Request body: application/json
    {
        "success": true,
        "incident_type": "Flood",
        "confidence": 0.94,
        "source": "roboflow",
        "detected_objects": [...],
        "message": "..."
    }
    OR a JSON array of the above for batch priority ranking.

    Returns
    -------
    200  – priority response
    400  – invalid / missing JSON body
    500  – internal error
    """
    body = request.get_json(silent=True)
    if body is None:
        return (
            jsonify(
                {
                    "success": False,
                    "message": (
                        "Request body must be JSON. "
                        "Send the detection result from /api/detect."
                    ),
                }
            ),
            400,
        )

    # Accept both a single dict and a list of dicts
    if not isinstance(body, (dict, list)):
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Body must be a detection JSON object or a list of them.",
                }
            ),
            400,
        )

    try:
        from priority.priority_engine import prioritize as run_priority

        result = run_priority(body)
        return jsonify(result), 200

    except Exception:
        logger.exception("Unexpected error in /api/prioritize.")
        return (
            jsonify({"success": False, "message": "Priority engine error."}),
            500,
        )


@app.post("/api/detect-and-prioritize")
def detect_and_prioritize():
    """
    POST /api/detect-and-prioritize

    Convenience endpoint: accepts a multipart image upload, runs incident
    detection, then immediately pipes the result through the priority engine.

    The detection and priority modules remain fully separate internally;
    this route is just a thin orchestration layer.

    Returns the full priority JSON response.
    """
    # ── Reuse the same image validation as /api/detect ────────────────────────
    if "image" not in request.files:
        return _error("No image field found. Use field name 'image'.")

    file = request.files["image"]
    if file.filename == "" or file.filename is None:
        return _error("No file selected.")
    if not _allowed_file(file.filename):
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        return _error(f"Unsupported file type. Allowed: {allowed}.")

    try:
        image_bytes = file.read()
    except Exception:
        logger.exception("Failed to read uploaded file.")
        return _error("Could not read the uploaded file.", 400)

    if not image_bytes:
        return _error("Uploaded file is empty.")

    try:
        import io
        from PIL import Image
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        return _error("Uploaded file is not a valid image.")

    image_id = request.form.get("image_id", "")
    client_file_name = request.form.get("client_file_name", file.filename or "")

    # ── Step 1: Detect ────────────────────────────────────────────────────────
    try:
        from detection.incident_detector import detect_incident
        detection_result = detect_incident(image_bytes)
    except RuntimeError as exc:
        logger.error("Detection failed in detect-and-prioritize: %s", exc)
        return (
            jsonify({"success": False, "message": str(exc), "image_id": image_id, "filename": client_file_name}),
            500,
        )
    except Exception:
        logger.exception("Unexpected detection error in detect-and-prioritize.")
        return _error("Detection step failed.", 500)

    # ── Step 2: Prioritize ────────────────────────────────────────────────────
    try:
        from priority.priority_engine import prioritize as run_priority
        priority_result = run_priority(detection_result)
        if isinstance(priority_result, dict):
            priority_result["image_id"] = image_id
            priority_result["filename"] = client_file_name or file.filename
        return jsonify(priority_result), 200
    except Exception:
        logger.exception("Priority step failed in detect-and-prioritize.")
        return (
            jsonify({"success": False, "message": "Priority engine error.", "image_id": image_id, "filename": client_file_name}),
            500,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────


@app.errorhandler(413)
def too_large(_e):
    mb = config.MAX_CONTENT_LENGTH // (1024 * 1024)
    return _error(f"File too large. Maximum upload size is {mb} MB.", 413)


@app.errorhandler(404)
def not_found(_e):
    return _error("Endpoint not found.", 404)


@app.errorhandler(405)
def method_not_allowed(_e):
    return _error("Method not allowed.", 405)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    logger.info("Starting AURA Incident Detection service on port %d …", port)

    # Pre-warm the CLIP model so the first request is fast.
    # Also runs a dummy inference to cache the text feature embeddings,
    # which reduces subsequent CLIP inference from ~11s to ~0.1s.
    try:
        import io as _io
        from PIL import Image as _Image
        from detection.clip_classifier import _load_model, classify_image
        logger.info("Pre-warming CLIP model at startup...")
        _load_model()
        # Run a dummy inference to cache text features
        _dummy = _Image.new("RGB", (224, 224), color=(128, 128, 128))
        _buf = _io.BytesIO()
        _dummy.save(_buf, format="JPEG")
        classify_image(_buf.getvalue())
        logger.info("CLIP model ready (text features cached).")
    except Exception as exc:
        logger.warning(
            "CLIP model pre-warm failed (%s). "
            "First request may be slow or CLIP features may be unavailable.",
            exc,
        )

    # use_reloader=False: prevents Werkzeug from auto-reloading the process
    # when source files change (which would kill long-running CLIP inference).
    # Restart the server manually after code changes.
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)

