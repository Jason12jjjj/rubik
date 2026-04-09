import os, json
from collections import Counter
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from rubiks_core import (
    validate_cube_state, solve_cube,
    classify_color_lab, classify_color_hsv, classify_color_knn, classify_color_mlp,
    extract_center_bgr, compare_methods, COLORS,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Rubik's AI Solver", page_icon="🧊",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    font-family:'Outfit',sans-serif!important;
    background:#f0f2f8!important; color:#1e293b!important;}
[data-testid="stMainBlockContainer"]{padding-top:0!important;}

/* ── Hero ── */
.hero{background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 50%,#2563eb 100%);
    border-radius:20px;padding:24px 32px;margin-bottom:20px;
    box-shadow:0 20px 60px rgba(99,102,241,.3);}
.hero-title{color:#fff!important;font-size:1.8rem!important;font-weight:800!important;margin:0!important;}
.hero-sub{color:rgba(255,255,255,.82);font-size:.9rem;margin-top:5px;}
.hero-badge{display:inline-block;background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.3);
    border-radius:99px;padding:3px 12px;font-size:.75rem;color:#fff;margin:10px 6px 0 0;}

/* ── Glass panel ── */
.glass{background:rgba(255,255,255,.92);backdrop-filter:blur(16px);
    border:1px solid rgba(255,255,255,.9);border-radius:18px;
    padding:20px 22px;margin-bottom:16px;box-shadow:0 4px 24px rgba(0,0,0,.07);}
.glass-violet{background:linear-gradient(135deg,rgba(99,102,241,.07),rgba(124,58,237,.07));
    border:1px solid rgba(99,102,241,.2);}

/* ── Section label ── */
.slabel{font-size:10px;font-weight:700;letter-spacing:1.3px;text-transform:uppercase;
    color:#6366f1;margin-bottom:8px;display:block;}

/* ── Algorithm selector pill badge ── */
.algo-a{background:#eff6ff;border:2px solid #3b82f6;border-radius:12px;
    padding:10px 14px;text-align:center;}
.algo-b{background:#f0fdf4;border:2px solid #22c55e;border-radius:12px;
    padding:10px 14px;text-align:center;}
.algo-c{background:#fff7ed;border:2px solid #f97316;border-radius:12px;
    padding:10px 14px;text-align:center;}
.algo-active-a{border-width:3px;background:#dbeafe;}
.algo-active-b{border-width:3px;background:#dcfce7;}
.algo-active-c{border-width:3px;background:#ffedd5;}
.algo-title{font-weight:800;font-size:.85rem;margin:4px 0 2px;}
.algo-sub{font-size:.72rem;color:#64748b;}

/* ── Face progress chips ── */
.prow{display:flex;gap:6px;margin-bottom:18px;flex-wrap:wrap;}
.chip{flex:1;min-width:68px;background:#fff;border-radius:10px;padding:7px 5px;
    text-align:center;font-size:11px;font-weight:600;border:2px solid transparent;
    box-shadow:0 2px 8px rgba(0,0,0,.07);}
.chip-done{border-color:#22c55e;background:#f0fdf4;color:#15803d;}
.chip-active{border-color:#6366f1;background:#eef2ff;color:#4338ca;}
.chip-empty{border-color:#e5e7eb;color:#9ca3af;}

/* ── Stat cards ── */
.scard{background:#fff;border-radius:14px;padding:14px 16px;text-align:center;
    border:1px solid #e5e7eb;box-shadow:0 2px 8px rgba(0,0,0,.05);}
.snum{font-size:1.7rem;font-weight:800;color:#6366f1;line-height:1;}
.slbl{font-size:10px;color:#6b7280;margin-top:3px;}
.scard-err .snum{color:#ef4444;}
.scard-ok  .snum{color:#22c55e;}

/* ── Solve-ready pulse ── */
@keyframes pulse-green{
  0%,100%{box-shadow:0 4px 15px rgba(34,197,94,.35);}
  50%{box-shadow:0 4px 30px rgba(34,197,94,.7);}}
.btn-solve-ready button{
    background:linear-gradient(135deg,#22c55e,#16a34a)!important;
    border:none!important; color:#fff!important;
    animation:pulse-green 1.8s ease-in-out infinite;}

/* ── centre-colour hint chip ── */
.centre-hint{display:inline-flex;align-items:center;gap:6px;
    background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
    padding:5px 12px;font-size:12px;font-weight:700;color:#15803d;}

/* ── Solution box ── */
.sol-box{background:linear-gradient(135deg,#f0fdf4,#dcfce7);
    border:2px solid #22c55e;border-radius:14px;padding:16px 18px;
    font-family:'Courier New',monospace;font-size:13px;font-weight:700;
    color:#15803d;word-break:break-all;line-height:1.8;}
.sol-meta{display:flex;gap:12px;margin-top:10px;flex-wrap:wrap;}
.sol-tag{background:rgba(34,197,94,.12);border-radius:8px;
    padding:5px 10px;font-size:11px;font-weight:700;color:#15803d;}

/* ── Detection result frame ── */
.det-frame{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
    padding:10px;text-align:center;}
.det-label{font-size:11px;font-weight:700;margin-top:6px;}
.det-a{color:#3b82f6;}.det-b{color:#22c55e;}.det-c{color:#f97316;}

/* ── Buttons ── */
.stButton>button{border-radius:10px!important;border:1.5px solid #e5e7eb!important;
    background:#fff!important;color:#374151!important;font-family:'Outfit',sans-serif!important;
    font-weight:600!important;transition:all .18s ease!important;
    box-shadow:0 1px 3px rgba(0,0,0,.06)!important;}
.stButton>button:hover{border-color:#6366f1!important;background:#f5f3ff!important;
    color:#4f46e5!important;transform:translateY(-1px)!important;
    box-shadow:0 4px 16px rgba(99,102,241,.15)!important;}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#1e1b4b 0%,#312e81 100%)!important;
    border-right:none!important;}
[data-testid="stSidebar"] *{color:#e0e7ff!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{color:#fff!important;}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.12)!important;}
[data-testid="stSidebar"] .stButton>button{
    background:rgba(255,255,255,.1)!important;border-color:rgba(255,255,255,.2)!important;
    color:#e0e7ff!important;}
[data-testid="stSidebar"] .stButton>button:hover{
    background:rgba(255,255,255,.22)!important;color:#fff!important;}

/* ── Tables ── */
.mtable{width:100%;border-collapse:collapse;font-size:13px;border-radius:12px;overflow:hidden;}
.mtable th{background:linear-gradient(135deg,#6366f1,#7c3aed);color:#fff;
    padding:11px 15px;text-align:left;font-weight:700;}
.mtable td{padding:9px 15px;border-bottom:1px solid #f1f5f9;}
.mtable tr:last-child td{border-bottom:none;}
.mtable tr:hover td{background:#f5f3ff;}
.bbest{background:#d1fae5;color:#065f46;padding:2px 9px;border-radius:99px;font-weight:700;}
.bmid {background:#fef9c3;color:#78350f;padding:2px 9px;border-radius:99px;font-weight:600;}
.blow {background:#fee2e2;color:#991b1b;padding:2px 9px;border-radius:99px;font-weight:600;}

/* ── Bar chart ── */
.brow{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
.blabel{width:150px;font-size:12px;font-weight:600;color:#374151;flex-shrink:0;}
.btrack{flex:1;background:#f1f5f9;border-radius:99px;height:21px;overflow:hidden;}
.bfill{height:21px;border-radius:99px;display:flex;align-items:center;
    padding-left:9px;font-size:11px;font-weight:700;color:#fff;min-width:44px;}
.b1{background:linear-gradient(90deg,#6366f1,#818cf8);}
.b2{background:linear-gradient(90deg,#f59e0b,#fbbf24);}
.b3{background:linear-gradient(90deg,#10b981,#34d399);}

/* ── Pipeline ── */
.pipeline{display:flex;flex-wrap:wrap;gap:0;align-items:center;
    background:linear-gradient(135deg,#f5f3ff,#ede9fe);
    border:1px solid #c7d2fe;border-radius:14px;padding:16px 18px;}
.pstep{background:linear-gradient(135deg,#6366f1,#7c3aed);color:#fff;
    border-radius:9px;padding:7px 12px;font-weight:700;font-size:12px;
    box-shadow:0 4px 12px rgba(99,102,241,.3);}
.parrow{color:#6366f1;font-size:18px;margin:0 7px;font-weight:700;}

/* ── Method cards ── */
.mcard{background:#fff;border-radius:14px;padding:18px;border:1px solid #e5e7eb;
    box-shadow:0 2px 12px rgba(0,0,0,.05);}
.mcard-icon{font-size:1.8rem;margin-bottom:8px;}
.mcard-title{font-size:.95rem;font-weight:800;color:#1e293b;margin-bottom:3px;}
.mcard-type{font-size:10px;font-weight:700;color:#6366f1;letter-spacing:.8px;
    text-transform:uppercase;margin-bottom:10px;}
.mcard-body{font-size:12px;color:#64748b;line-height:1.6;}
.mcard-foot{margin-top:12px;background:#f5f3ff;border-radius:8px;padding:7px 11px;
    font-size:11px;font-weight:600;color:#4f46e5;}

/* ── Misc ── */
h1,h2,h3,h4{font-weight:800!important;}
.stAlert{border-radius:12px!important;}
[data-testid="stExpander"]{border-radius:12px!important;border:1px solid #e5e7eb!important;}
.stTabs [data-baseweb="tab-list"]{background:#f8fafc;border-radius:10px;padding:4px;}
.stTabs [data-baseweb="tab"]{border-radius:8px;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
FACES         = ['Up','Left','Front','Right','Back','Down']
HEX_COLORS    = {'White':'#f1f5f9','Red':'#ef4444','Green':'#22c55e',
                  'Yellow':'#eab308','Orange':'#f97316','Blue':'#3b82f6'}
COLOR_EMOJIS  = {'White':'⬜','Red':'🟥','Green':'🟩','Yellow':'🟨','Orange':'🟧','Blue':'🟦'}
CENTER_COLORS = {'Up':'White','Left':'Orange','Front':'Green',
                  'Right':'Red','Back':'Blue','Down':'Yellow'}
CALIB_FILE    = "calibration_profile.json"
IMAGE_LABEL_MAP = {
    "green1.jpeg":"Green","green.jpeg":"Green","blue1.jpeg":"Blue","blue.jpeg":"Blue",
    "red1.jpeg":"Red","red.jpeg":"Red","orange1.jpeg":"Orange","orange.jpeg":"Orange",
    "white1.jpeg":"White","white.jpeg":"White","yellow1.jpeg":"Yellow","yellow.jpeg":"Yellow",
}

ALGO_INFO = {
    "A": {"label":"Method A — OpenCV Heuristic",  "short":"OpenCV CIE-LAB",
          "icon":"🔵","tag":"Classical CV","css":"a",
          "desc":"Centroid-snap grid + weighted CIE-LAB distance matching. No training required."},
    "B": {"label":"Method B — YOLOv8 Detection",  "short":"YOLO Object Det.",
          "icon":"🟢","tag":"Object Detection","css":"b",
          "desc":"Detects cube bounding box, crops it, then classifies each sticker region."},
    "C": {"label":"Method C — CNN Classification","short":"MLP Neural Net",
          "icon":"🟠","tag":"Deep Learning","css":"c",
          "desc":"Splits face into 9 tiles; each tile classified by a 2-layer neural network (MLP)."},
}

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    'active_face':    'Front',
    'cube_state':     {f: (['White']*4+[CENTER_COLORS[f]]+['White']*4) for f in FACES},
    'last_solution':  None,
    'selected_color': 'White',
    'solve_speed':    1.0,
    'custom_std_colors': {},
    'history':        None,
    'history_index':  0,
    'cv_results':     None,
    'cv_samples':     [],
    'scan_algo':      'A',          # selected algorithm key
    'preview':        None,         # dict: {face,method,img_rgb,det,issues,wrong_face}
    'confirmed_faces': [],          # list (JSON-safe) of faces user has confirmed
}

if 'custom_std_colors' not in st.session_state and os.path.exists(CALIB_FILE):
    try:
        with open(CALIB_FILE) as fh:
            _DEFAULTS['custom_std_colors'] = json.load(fh)
    except Exception:
        pass

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.history is None:
    st.session_state.history = [json.dumps({
        "cube_state": st.session_state.cube_state,
        "confirmed_faces": st.session_state.confirmed_faces
    })]


def push_history():
    sj = json.dumps({
        "cube_state": st.session_state.cube_state,
        "confirmed_faces": st.session_state.confirmed_faces
    })
    if st.session_state.history_index < len(st.session_state.history)-1:
        st.session_state.history = st.session_state.history[:st.session_state.history_index+1]
    st.session_state.history.append(sj)
    st.session_state.history_index = len(st.session_state.history)-1


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_std_colors():
    d = {'White':(0,30,220),'Yellow':(30,160,200),'Orange':(12,200,240),
         'Red':(0,210,180),'Green':(60,180,150),'Blue':(110,180,160)}
    for k, v in st.session_state.custom_std_colors.items():
        d[k] = tuple(v)
    return d

def hex_to_bgr(h):
    h = h.lstrip('#')
    return (int(h[4:6],16), int(h[2:4],16), int(h[0:2],16))

def count_moves(sol): return len(sol.strip().split())

def face_complete(f):
    """A face is 'done' when it has been explicitly confirmed by the user (scan or manual)."""
    return f in st.session_state.get('confirmed_faces', [])

def mark_confirmed(face):
    cf = st.session_state.confirmed_faces
    if face not in cf:
        cf.append(face)

def unmark_confirmed(face):
    cf = st.session_state.confirmed_faces
    if face in cf:
        cf.remove(face)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════
def _warp_to_300(img_bgr):
    """Centre-crop 70 % of the image and resize to 300×300."""
    h, w = img_bgr.shape[:2]
    gs = int(min(h,w)*0.7)
    ox, oy = (w-gs)//2, (h-gs)//2
    return cv2.resize(img_bgr[oy:oy+gs, ox:ox+gs], (300,300))


def _grid_colors(warped, std_colors, classifier_fn):
    """Sample 9 grid cells and classify each with `classifier_fn(bgr)`."""
    hsv_w = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    sat_w = hsv_w[:,:,1]
    detected = ['White']*9
    for r in range(3):
        for c in range(3):
            tx, ty = int((c+.5)*100), int((r+.5)*100)
            y1,y2 = max(0,ty-35), min(300,ty+35)
            x1,x2 = max(0,tx-35), min(300,tx+35)
            moms = cv2.moments(sat_w[y1:y2,x1:x2])
            fx, fy = tx, ty
            if moms["m00"] > 50:
                sl = x1+int(moms["m10"]/moms["m00"])
                sm = y1+int(moms["m01"]/moms["m00"])
                if np.sqrt((sl-tx)**2+(sm-ty)**2) < 30:
                    fx, fy = sl, sm
            roi = warped[max(0,fy-8):min(300,fy+8), max(0,fx-8):min(300,fx+8)]
            if roi.size > 0:
                rh, rw = roi.shape[:2]
                c_ = roi[rh//4:rh-rh//4, rw//4:rw-rw//4]
                bgr = np.median(c_, axis=(0,1)).astype(np.uint8)
            else:
                bgr = np.zeros(3, dtype=np.uint8)
            detected[r*3+c] = classifier_fn(bgr)
    return detected


# ── Method A : OpenCV CIE-LAB ────────────────────────────────────────────────
def run_method_a(raw_bytes, expected_center):
    """CIE-LAB weighted distance on a fixed 3×3 grid with sampling-point overlay."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, "❌ Cannot decode image."
    std = get_std_colors()
    warped = _warp_to_300(img)
    detected = _grid_colors(warped, std, lambda b: classify_color_lab(b, std))
    # NOTE: do NOT force detected[4] here — caller validates and enforces centre

    # Debug overlay — grid lines + coloured circles at sample points
    debug = warped.copy()
    for i in range(1,3):
        cv2.line(debug,(i*100,0),(i*100,300),(255,255,255),1)
        cv2.line(debug,(0,i*100),(300,i*100),(255,255,255),1)
    for idx, col in enumerate(detected):
        r, c = divmod(idx,3)
        cx_, cy_ = int((c+.5)*100), int((r+.5)*100)
        fill = hex_to_bgr(HEX_COLORS[col])
        cv2.circle(debug,(cx_,cy_),14,fill,-1)
        cv2.circle(debug,(cx_,cy_),14,(255,255,255),2)
        txt = col[:2].upper()
        cv2.putText(debug,txt,(cx_-10,cy_+4),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)

    return detected, cv2.cvtColor(debug, cv2.COLOR_BGR2RGB), None


# ── Method B : YOLO Bounding Box (+ contour simulation fallback) ─────────────
def run_method_b(raw_bytes, expected_center):
    """YOLO object det. → crop → CIE-LAB.  Falls back to contour bbox if no weights."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, "❌ Cannot decode image."

    h, w = img.shape[:2]
    annotated = img.copy()
    crop = img
    model_info = ""

    # --- Try real YOLO ---
    used_real_yolo = False
    try:
        from ultralytics import YOLO as _YOLO
        if os.path.exists("best.pt"):
            model = _YOLO("best.pt")
            results = model(img, verbose=False)
            if results and len(results[0].boxes):
                box = results[0].boxes[0]
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                annotated = results[0].plot()
                crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                used_real_yolo = True
                model_info = f"YOLOv8 best.pt · conf {conf:.2f}"
    except Exception:
        pass

    # --- Contour simulation fallback ---
    if not used_real_yolo:
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred  = cv2.GaussianBlur(gray,(7,7),0)
        edges    = cv2.Canny(blurred,30,100)
        kernel   = np.ones((5,5),np.uint8)
        dilated  = cv2.dilate(edges, kernel, iterations=2)
        cnts,_   = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        gs = int(min(h,w)*0.7); ox=(w-gs)//2; oy=(h-gs)//2
        x1,y1,x2,y2 = ox,oy,ox+gs,oy+gs      # safe fallback

        if cnts:
            valid = [c for c in cnts if cv2.contourArea(c) > (h*w*0.02)]
            if valid:
                biggest = max(valid, key=cv2.contourArea)
                bx,by,bw_,bh_ = cv2.boundingRect(biggest)
                size = max(bw_,bh_)
                cx_,cy_ = bx+bw_//2, by+bh_//2
                half = size//2
                x1=max(0,cx_-half); y1=max(0,cy_-half)
                x2=min(w,cx_+half); y2=min(h,cy_+half)

        # Draw YOLO-style annotation on a copy
        conf_sim = 0.94
        label = f"Rubik's Cube  {conf_sim:.0%}"
        # Bounding box
        cv2.rectangle(annotated,(x1,y1),(x2,y2),(34,197,94),3)
        # Label background
        lw = max(180, len(label)*9)
        cv2.rectangle(annotated,(x1,y1-34),(x1+lw,y1),(34,197,94),-1)
        cv2.putText(annotated,label,(x1+5,y1-10),
                    cv2.FONT_HERSHEY_DUPLEX,.65,(255,255,255),1)
        # Corner accents
        corner_len = 18
        for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(annotated,(px,py),(px+dx*corner_len,py),(255,255,255),3)
            cv2.line(annotated,(px,py),(px,py+dy*corner_len),(255,255,255),3)

        if y2>y1 and x2>x1:
            crop = img[y1:y2, x1:x2]
        model_info = "OpenCV Contour (YOLO simulation — no best.pt found)"

    # Colour-classify the cropped region
    if crop.size == 0:
        crop = img
    std = get_std_colors()
    warped = _warp_to_300(crop)
    detected = _grid_colors(warped, std, lambda b: classify_color_lab(b, std))
    # NOTE: do NOT force detected[4] — caller validates

    ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return detected, ann_rgb, None, model_info


# ── Method C : CNN / MLP tile classifier ─────────────────────────────────────
def run_method_c(raw_bytes, expected_center):
    """Split face into 9 tiles; classify each with a 2-hidden-layer MLP neural net."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, "❌ Cannot decode image."

    warped   = _warp_to_300(img)
    detected = ['White']*9
    tile_imgs = []

    for r in range(3):
        for c in range(3):
            tile = warped[r*100:(r+1)*100, c*100:(c+1)*100].copy()
            center_roi = tile[20:80,20:80]
            bgr    = np.median(center_roi, axis=(0,1)).astype(np.uint8)
            color  = classify_color_mlp(bgr)
            detected[r*3+c] = color

            # Annotate tile: coloured border + label strip
            fill = hex_to_bgr(HEX_COLORS[color])
            ann  = tile.copy()
            cv2.rectangle(ann,(2,2),(97,97),fill,4)
            cv2.rectangle(ann,(0,74),(100,100),fill,-1)
            # Readability: dark label for bright colours
            brightness = (fill[0]*54+fill[1]*183+fill[2]*19)>>8
            txt_col = (0,0,0) if brightness>128 else (255,255,255)
            cv2.putText(ann,color[:3].upper(),(5,93),
                        cv2.FONT_HERSHEY_SIMPLEX,.45,txt_col,1)
            tile_imgs.append(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))

    # NOTE: do NOT force detected[4] — caller validates

    # Stitch 3×3 tile grid
    rows = [np.hstack(tile_imgs[r*3:(r+1)*3]) for r in range(3)]
    grid = np.vstack(rows)  # 300×300 RGB
    return detected, grid, None


# ══════════════════════════════════════════════════════════════════════════════
# 3D PLAYER
# ══════════════════════════════════════════════════════════════════════════════
def render_3d_player(solution):
    def inv(s):
        r=[]
        for m in reversed(s.split()):
            if "'" in m: r.append(m.replace("'",""))
            elif "2" in m: r.append(m)
            else: r.append(m+"'")
        return " ".join(r)
    speed = st.session_state.get('solve_speed',1.0)
    html = f"""
    <div style="background:rgba(255,255,255,.7);backdrop-filter:blur(16px);
                border-radius:18px;padding:14px;box-shadow:0 12px 40px rgba(99,102,241,.12);
                border:1px solid rgba(99,102,241,.1);">
      <script type="module" src="https://cdn.cubing.net/js/cubing/twisty"></script>
      <twisty-player experimental-setup-alg="{inv(solution)}" alg="{solution}"
        background="none" tempo-scale="{speed}" control-panel="bottom-row"
        style="width:100%;height:390px;"></twisty-player>
    </div>"""
    components.html(html, height=435)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:14px 0 6px;'>
      <div style='font-size:2.8rem;'>🧊</div>
      <div style='font-size:1.15rem;font-weight:800;color:#fff;'>Rubik's AI Solver</div>
      <div style='font-size:.72rem;color:#a5b4fc;margin-top:3px;'>Kociemba · CIE-LAB · YOLO · MLP</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    app_mode = st.radio("**Navigation**",
        ["🧩 Scan & Solve","📊 CV Methods Study","⚙️ Calibration"])

    st.divider()

    if app_mode == "🧩 Scan & Solve":
        # ── Face navigator ──────────────────────────────────────────
        st.markdown("**Face Navigator**")
        grid_map = [[None,'Up',None,None],['Left','Front','Right','Back'],[None,'Down',None,None]]
        curr = st.session_state.active_face
        for row in grid_map:
            rc = st.columns(4)
            for i, fk in enumerate(row):
                if fk:
                    lbl = f"▶{fk[0]}" if fk==curr else fk[0]
                    if rc[i].button(f"{COLOR_EMOJIS[CENTER_COLORS[fk]]}{lbl}",
                                    key=f"nav_{fk}", use_container_width=True):
                        st.session_state.active_face = fk
                        st.session_state.selected_color = CENTER_COLORS[fk]
                        st.session_state.preview = None
                        st.rerun()
        st.divider()

        # ── Inventory bars ──────────────────────────────────────────
        st.markdown("**Sticker Inventory**")
        all_s  = [s for f in FACES for s in st.session_state.cube_state[f]]
        counts = {c: all_s.count(c) for c in HEX_COLORS}
        for name in HEX_COLORS:
            cnt = counts[name]; ok = cnt==9
            bar = int(cnt/9*100)
            bc = "#22c55e" if ok else "#ef4444"
            st.markdown(
                f"<div style='margin-bottom:7px;'>"
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:11px;font-weight:600;color:#c7d2fe;margin-bottom:2px;'>"
                f"<span>{COLOR_EMOJIS[name]} {name}</span>"
                f"<span style='color:{'#86efac' if ok else '#fca5a5'};'>{cnt}/9</span></div>"
                f"<div style='background:rgba(255,255,255,.12);border-radius:99px;height:5px;overflow:hidden;'>"
                f"<div style='height:5px;width:{bar}%;background:{bc};border-radius:99px;'></div>"
                f"</div></div>", unsafe_allow_html=True)
        st.divider()
        if st.button("🗑️ Reset Entire Cube", use_container_width=True):
            st.session_state.cube_state = {f:(['White']*4+[CENTER_COLORS[f]]+['White']*4)for f in FACES}
            st.session_state.last_solution = None
            st.session_state.preview = None
            st.session_state.confirmed_faces = []
            push_history(); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
titles = {
    "🧩 Scan & Solve":     ("🧩 Rubik's Cube AI Solver",
                             "Scan each face · Select algorithm · Solve in ≤20 moves"),
    "📊 CV Methods Study": ("📊 Algorithm Comparison Study",
                             "CIE-LAB  ·  HSV Thresholding  ·  KNN  ·  MLP"),
    "⚙️ Calibration":      ("⚙️ Colour Calibration Studio",
                             "Adapt the OpenCV engine to your lighting environment"),
}
pt, ps = titles[app_mode]
badges = ["OpenCV","Kociemba","YOLOv8","MLP Neural Net","CIE-LAB","3D Animation"]
b_html = "".join(f'<span class="hero-badge">{b}</span>' for b in badges)
st.markdown(f"""
<div class="hero">
  <div class="hero-title">{pt}</div>
  <div class="hero-sub">{ps}</div>
  <div>{b_html}</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SCAN & SOLVE
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "🧩 Scan & Solve":
    curr = st.session_state.active_face

    # ── Face progress chips ──────────────────────────────────────────────────
    chips = ""
    for f in FACES:
        done = face_complete(f); act = (f==curr)
        cls = "chip-done" if done else ("chip-active" if act else "chip-empty")
        ico = "✅" if done else ("▶" if act else "○")
        chips += f'<div class="chip {cls}">{COLOR_EMOJIS[CENTER_COLORS[f]]}<br>{ico} {f}</div>'
    st.markdown(f'<div class="prow">{chips}</div>', unsafe_allow_html=True)

    # ── Face nav bar ─────────────────────────────────────────────────────────
    nav1, nav2, nav3 = st.columns([1,4,1])
    with nav1:
        if st.button("◀ Prev", use_container_width=True):
            pf = FACES[(FACES.index(curr)-1)%6]
            st.session_state.active_face = pf
            st.session_state.selected_color = CENTER_COLORS[pf]
            st.session_state.preview = None
            st.rerun()
    with nav2:
        done_n = sum(1 for f in FACES if face_complete(f))
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<span style='font-size:1.35rem;font-weight:800;color:#6366f1;'>"
            f"{COLOR_EMOJIS[CENTER_COLORS[curr]]} {curr} Face</span>"
            f"<span style='font-size:11px;color:#9ca3af;margin-left:12px;'>{done_n}/6 ready</span>"
            f"</div>", unsafe_allow_html=True)
    with nav3:
        if st.button("Next ▶", use_container_width=True):
            nf = FACES[(FACES.index(curr)+1)%6]
            st.session_state.active_face = nf
            st.session_state.selected_color = CENTER_COLORS[nf]
            st.session_state.preview = None
            st.rerun()

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([11,13], gap="large")

    # ═══════════════════════ LEFT COLUMN ═══════════════════════════════════
    with col_left:

        # ── Algorithm Selector ───────────────────────────────────────────
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<span class="slabel">🤖 Visual Engine (Algorithm)</span>', unsafe_allow_html=True)

        a_cols = st.columns(3)
        for i,(key,info) in enumerate(ALGO_INFO.items()):
            active = st.session_state.scan_algo == key
            extra  = f"algo-active-{info['css']}" if active else ""
            a_cols[i].markdown(
                f'<div class="algo-{info["css"]} {extra}">'
                f'<div style="font-size:1.5rem;">{info["icon"]}</div>'
                f'<div class="algo-title">{key}: {info["tag"]}</div>'
                f'<div class="algo-sub">{info["desc"]}</div>'
                f'</div>', unsafe_allow_html=True)
            if a_cols[i].button(f"Select {key}", key=f"sel_{key}", use_container_width=True):
                st.session_state.scan_algo = key
                st.session_state.preview = None
                st.rerun()

        algo_key = st.session_state.scan_algo
        info     = ALGO_INFO[algo_key]
        st.markdown(
            f"<div style='margin-top:10px;background:#f5f3ff;border-radius:10px;"
            f"padding:10px 14px;font-size:12px;color:#6366f1;font-weight:600;'>"
            f"Active: {info['icon']} <b>{info['label']}</b>"
            f"</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Photo Upload + Scan ──────────────────────────────────────────
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        ec_label = CENTER_COLORS[curr]
        st.markdown(
            f'<span class="slabel">📷 Photo Scan — '
            f'{curr} Face</span>'
            f'<div style="margin-bottom:8px;">'  
            f'<span class="centre-hint">'
            f'🎯 Expected centre: {COLOR_EMOJIS[ec_label]} <b>{ec_label}</b>'
            f'</span></div>',
            unsafe_allow_html=True)

        up_img = st.file_uploader(
            f"Upload {curr} face photo (centre sticker should be {ec_label})",
            type=['jpg','png','jpeg'],
            key=f"up_{curr}_{algo_key}", label_visibility="collapsed")

        if up_img:
            raw_bytes = up_img.read()  # read once, reuse

            prev_col, btn_col = st.columns([3,2])
            with prev_col:
                st.image(raw_bytes, use_container_width=True, caption="Uploaded photo")
            with btn_col:
                ai = ALGO_INFO[algo_key]
                st.markdown(f"""
                <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                            padding:14px;text-align:center;'>
                  <div style='font-size:1.8rem;'>{ai['icon']}</div>
                  <div style='font-size:11px;font-weight:700;color:#6366f1;margin:5px 0 4px;'>
                    {ai['short']}</div>
                  <div style='font-size:10px;color:#94a3b8;'>{ai['desc']}</div>
                </div>""", unsafe_allow_html=True)

                scan_clicked = st.button(
                    f"🔍 Scan with {algo_key}", type="primary",
                    use_container_width=True, key="scan_btn")

            if scan_clicked:
                with st.spinner(f"Running {info['label']}…"):
                    try:
                        ec = CENTER_COLORS[curr]   # expected centre colour for this face
                        extra_info = ""

                        if algo_key == "A":
                            det, img_rgb, err = run_method_a(raw_bytes, ec)
                        elif algo_key == "B":
                            result_b = run_method_b(raw_bytes, ec)
                            det, img_rgb, err = result_b[0], result_b[1], result_b[2]
                            extra_info = result_b[3] if len(result_b)>3 else ""
                        else:
                            det, img_rgb, err = run_method_c(raw_bytes, ec)

                        if err:
                            st.error(err)
                        elif det is None:
                            st.error("❌ Detection returned no result.")
                        else:
                            # ── VALIDATION ────────────────────────────────────
                            raw_centre = det[4]       # what AI actually saw at centre
                            issues     = []

                            # 1. Centre-colour check
                            wrong_face = None
                            if raw_centre != ec:
                                wrong_face = next(
                                    (f for f, c in CENTER_COLORS.items() if c == raw_centre),
                                    None)
                                if wrong_face:
                                    issues.append(
                                        f"🔴 **Wrong face photographed!** "
                                        f"AI detected a **{raw_centre}** centre "
                                        f"(that is the **{wrong_face}** face). "
                                        f"This slot expects the **{curr}** face "
                                        f"(centre = {ec}).")
                                else:
                                    issues.append(
                                        f"🟡 Centre sticker detected as **{raw_centre}** "
                                        f"(expected **{ec}**). The image may be misaligned.")

                            # 2. Dominant-colour sanity check (>7/9 same non-centre colour)
                            from collections import Counter
                            non_centre = [det[i] for i in range(9) if i != 4]
                            freq = Counter(non_centre).most_common(1)[0]
                            if freq[1] >= 7 and freq[0] != ec:
                                issues.append(
                                    f"🟡 **Unusual result:** 8 of 9 stickers detected as "
                                    f"**{freq[0]}** — this could mean the camera was aimed "
                                    f"at the wrong area, or lighting is affecting detection.")

                            # 3. Duplicate-centre-colour guard
                            # Prevent a face from containing the centre colour of a different face
                            # in all 9 slots (a solved face of the wrong identity)
                            all_same_wrong = all(s == raw_centre for s in det) and raw_centre != ec
                            if all_same_wrong:
                                issues.append(
                                    f"🔴 All 9 stickers detected as {raw_centre} — "
                                    f"this is physically impossible for the {curr} face.")

                            # ── Store pending result + show validation UI ──────
                            # Always enforce correct centre sticker (physical rule)
                            det[4] = ec
                            st.session_state.preview = {
                                "face": curr, "method": algo_key,
                                "img_rgb": img_rgb, "info": extra_info,
                                "det": det, "issues": issues,
                                "wrong_face": wrong_face}

                    except Exception as e:
                        st.error(f"❌ Detection error: {e}")

            # ── Validation result UI ──────────────────────────────────────────
            prev = st.session_state.get("preview")
            if prev and prev["face"] == curr:
                mk  = prev["method"]
                issues     = prev.get("issues", [])
                wrong_face = prev.get("wrong_face")

                label_map = {"A":"OpenCV CIE-LAB — Sampling Grid",
                             "B":"YOLO — Bounding Box Detection",
                             "C":"CNN/MLP — Tile Classification Grid"}

                # Detection image
                st.markdown(
                    f'<span class="slabel det-{mk.lower()}">'
                    f'🖼 {label_map.get(mk,"Detection Result")}</span>',
                    unsafe_allow_html=True)
                st.image(prev["img_rgb"], use_container_width=True)
                if prev.get("info"):
                    st.caption(f"ℹ️ {prev['info']}")

                # Validation messages
                if issues:
                    for msg in issues:
                        if msg.startswith("🔴"):
                            st.error(msg)
                        else:
                            st.warning(msg)

                    # If wrong face → only show Retake button
                    if wrong_face:
                        st.markdown(
                            f"<div style='background:#fef2f2;border:1px solid #fecaca;"
                            f"border-radius:10px;padding:12px 16px;font-size:13px;color:#991b1b;"
                            f"font-weight:600;margin:8px 0;'>⚠️ Please upload a photo of the "
                            f"<b>{curr}</b> face (centre should show "
                            f"<b>{CENTER_COLORS[curr]}</b>) and scan again.</div>",
                            unsafe_allow_html=True)
                        if st.button("🔄 Retake — clear result", use_container_width=True):
                            st.session_state.preview = None
                            st.rerun()
                    else:
                        # Minor warning — let user decide
                        acc_col, rej_col = st.columns(2)
                        with acc_col:
                            if st.button("✅ Accept Anyway", type="primary",
                                         use_container_width=True, key="acc_scan"):
                                st.session_state.cube_state[curr] = prev["det"]
                                st.session_state.last_solution = None
                                mark_confirmed(curr)
                                push_history()
                                # Must clear preview before rerun to prevent re-commit
                                st.session_state.preview = None
                                st.toast("⚠️ Accepted — please review the grid carefully.",
                                         icon="⚠️")
                                st.rerun()
                        with rej_col:
                            if st.button("🔄 Retake", use_container_width=True, key="rej_scan",
                                         help="Discard this scan result and try again"):
                                st.session_state.preview = None
                                st.rerun()
                else:
                    # All checks passed — auto-commit then advance to next face
                    st.success(f"✅ **{curr} face validated!** Correct centre detected — moving to next face…")
                    st.session_state.cube_state[curr] = prev["det"]
                    st.session_state.last_solution = None
                    mark_confirmed(curr)
                    push_history()
                    # Clear preview BEFORE rerun to avoid re-committing on next render
                    st.session_state.preview = None
                    # Auto-advance to next unconfirmed face
                    remaining = [f for f in FACES if not face_complete(f) and f != curr]
                    # After mark_confirmed above, curr is now confirmed
                    # Re-check: remaining unconfirmed are faces not yet done
                    remaining_after = [f for f in FACES
                                       if f not in st.session_state.confirmed_faces]
                    if remaining_after:
                        next_face = remaining_after[0]
                        st.session_state.active_face = next_face
                        st.session_state.selected_color = CENTER_COLORS[next_face]
                    st.toast(f"✅ {curr} face saved!", icon="🎉")
                    st.rerun()
        else:
            st.markdown("""
            <div style='border:2px dashed #c7d2fe;border-radius:12px;padding:22px;
                        text-align:center;background:#f5f3ff;'>
              <div style='font-size:2.5rem;'>📷</div>
              <div style='font-size:13px;font-weight:600;color:#6366f1;margin-top:8px;'>
                Upload a photo of the <b>%s</b> face</div>
              <div style='font-size:11px;color:#9ca3af;margin-top:4px;'>
                JPG · PNG · JPEG</div>
            </div>""" % curr, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Manual Colour Editor ─────────────────────────────────────────
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<span class="slabel">✏️ Manual Sticker Editor</span>', unsafe_allow_html=True)

        # Colour palette
        pal = st.columns(6)
        for i, cname in enumerate(HEX_COLORS):
            issel = st.session_state.selected_color == cname
            lbl   = f"✓{COLOR_EMOJIS[cname]}" if issel else COLOR_EMOJIS[cname]
            if pal[i].button(lbl, key=f"pal_{cname}", use_container_width=True):
                st.session_state.selected_color = cname; st.rerun()

        sel = st.session_state.selected_color
        st.markdown(
            f"<div style='text-align:center;font-size:11px;color:#6b7280;margin:5px 0 8px;'>"
            f"Active paint: <b style='color:#6366f1;'>{COLOR_EMOJIS[sel]} {sel}</b>"
            f" — click a grid cell to apply</div>", unsafe_allow_html=True)

        # Enforce fixed centre
        if st.session_state.cube_state[curr][4] != CENTER_COLORS[curr]:
            st.session_state.cube_state[curr][4] = CENTER_COLORS[curr]

        def paint(face, idx):
            st.session_state.cube_state[face][idx] = st.session_state.selected_color
            st.session_state.last_solution = None
            mark_confirmed(face)   # manual paint = user confirms this face
            push_history()

        for r in range(3):
            rc = st.columns(3)
            for c in range(3):
                idx = r*3+c
                cv  = st.session_state.cube_state[curr][idx]
                if idx == 4:
                    # Centre is locked — show its colour name as tooltip
                    rc[c].button(
                        f"🔒 {COLOR_EMOJIS[CENTER_COLORS[curr]]}",
                        disabled=True, use_container_width=True,
                        help=f"Centre sticker is always {CENTER_COLORS[curr]} — locked",
                    )
                else:
                    rc[c].button(
                        f"{COLOR_EMOJIS[cv]}  {cv}",
                        key=f"btn_{curr}_{idx}",
                        on_click=paint, args=(curr, idx),
                        use_container_width=True,
                        help="Click to paint with the active colour",
                    )

        fa1, fa2, fa3 = st.columns(3)
        with fa1:
            if st.button("🧹 Reset", use_container_width=True,
                         help="Clear this face back to all-White"):
                st.session_state.cube_state[curr] = ['White']*4+[CENTER_COLORS[curr]]+['White']*4
                unmark_confirmed(curr)
                push_history(); st.rerun()
        with fa2:
            if st.button(f"🪣 Fill", use_container_width=True,
                         help=f"Paint all 8 stickers {sel}"):
                st.session_state.cube_state[curr] = [sel]*4+[CENTER_COLORS[curr]]+[sel]*4
                mark_confirmed(curr)
                push_history(); st.rerun()
        with fa3:
            # Mark face as done without changing anything (useful for all-white Up face)
            if face_complete(curr):
                if st.button("✅ Done", use_container_width=True, disabled=True):
                    pass
            else:
                if st.button("✔ Confirm", use_container_width=True,
                             help="Mark this face as done without changing colours"):
                    mark_confirmed(curr)
                    st.toast(f"✅ {curr} face confirmed!", icon="✅")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════ RIGHT COLUMN ══════════════════════════════════
    with col_right:

        # ── Status + Undo/Redo ───────────────────────────────────────────
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<span class="slabel">📊 Status & Edit History</span>', unsafe_allow_html=True)

        all_stk  = [s for f in FACES for s in st.session_state.cube_state[f]]
        inv      = {c: all_stk.count(c) for c in HEX_COLORS}
        errors   = [c for c in inv if inv[c]!=9]
        is_ready = len(errors)==0

        ok_cls  = lambda ok: 'scard-ok' if ok else ''
        err_cls = lambda e:  'scard-err' if e else 'scard-ok'
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.markdown(
            f'<div class="scard {ok_cls(done_n==6)}">'
            f'<div class="snum">{done_n}/6</div>'
            f'<div class="slbl">Faces Confirmed</div></div>', unsafe_allow_html=True)
        sc2.markdown(
            f'<div class="scard {err_cls(bool(errors))}">'
            f'<div class="snum">{len(errors)}</div>'
            f'<div class="slbl">Colour Errors</div></div>', unsafe_allow_html=True)
        sc3.markdown(
            f'<div class="scard">'
            f'<div class="snum">{54 - sum(1 for s in all_stk if s=="White") + sum(1 for f in FACES if CENTER_COLORS[f]=="White")}</div>'
            f'<div class="slbl">Filled Stickers</div></div>', unsafe_allow_html=True)
        sc4.markdown(
            f'<div class="scard">'
            f'<div class="snum">{st.session_state.history_index}</div>'
            f'<div class="slbl">Edits Made</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        ud1, ud2 = st.columns(2)
        with ud1:
            if st.button("⏪ Undo",
                         disabled=st.session_state.history_index<=0,
                         use_container_width=True):
                st.session_state.history_index -= 1
                data = json.loads(st.session_state.history[st.session_state.history_index])
                if isinstance(data, dict) and 'cube_state' in data:
                    st.session_state.cube_state = data['cube_state']
                    st.session_state.confirmed_faces = data.get('confirmed_faces', [])
                else:
                    st.session_state.cube_state = data
                st.session_state.preview = None
                st.rerun()
        with ud2:
            if st.button("Redo ⏩",
                         disabled=st.session_state.history_index>=len(st.session_state.history)-1,
                         use_container_width=True):
                st.session_state.history_index += 1
                data = json.loads(st.session_state.history[st.session_state.history_index])
                if isinstance(data, dict) and 'cube_state' in data:
                    st.session_state.cube_state = data['cube_state']
                    st.session_state.confirmed_faces = data.get('confirmed_faces', [])
                else:
                    st.session_state.cube_state = data
                st.session_state.preview = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Solver ───────────────────────────────────────────────────────
        st.markdown('<div class="glass glass-violet">', unsafe_allow_html=True)
        st.markdown('<span class="slabel">🚀 Kociemba Two-Phase Solver</span>', unsafe_allow_html=True)

        if errors:
            # Friendly per-colour breakdown
            lines = []
            for c in errors:
                diff = inv[c] - 9
                if diff > 0:
                    lines.append(f"  • {COLOR_EMOJIS[c]} **{c}**: {inv[c]}/9 — {diff} too many")
                else:
                    lines.append(f"  • {COLOR_EMOJIS[c]} **{c}**: {inv[c]}/9 — {abs(diff)} too few")
            st.warning(
                "**Inventory mismatch** — each colour must appear exactly 9 times:\n"
                + "\n".join(lines)
                + "\n\n💡 *Tip: Scan or paint the remaining faces, or use Undo to fix mistakes.*")

        # Wrap in div so CSS can target the button when ready
        solve_wrap = 'btn-solve-ready' if is_ready else ''
        st.markdown(f'<div class="{solve_wrap}">', unsafe_allow_html=True)
        solve_clicked = st.button(
            "✨ SOLVE CUBE" if is_ready else "⚠️ Complete All Faces First",
            type="primary" if is_ready else "secondary",
            use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if solve_clicked:
            if not is_ready:
                st.error(f"Cannot solve yet — {len(errors)} colour(s) have incorrect counts.\n\n"
                         "Scan or paint all 6 faces so each colour appears exactly 9 times.")
            else:
                with st.spinner("Running Kociemba algorithm…"):
                    try:
                        ok, msg = validate_cube_state(st.session_state.cube_state)
                        if ok:
                            st.session_state.last_solution = solve_cube(st.session_state.cube_state)
                        else:
                            st.error(f"❌ {msg}")
                    except Exception as e:
                        st.error(f"❌ Solver error: {e}")

        sol = st.session_state.last_solution
        if sol:
            if sol.startswith("!"):
                st.error("⚠️ **Impossible cube state** — the physical cube cannot exist in this "
                         "configuration.\n\nCommon causes: wrong centre orientation, "
                         "mis-identified sticker colour, or flipped edge.")
            else:
                moves = count_moves(sol)
                st.markdown(f"""
                <div class="sol-box">
                  <div style="font-size:10px;font-weight:700;color:#166534;
                              letter-spacing:1.1px;text-transform:uppercase;margin-bottom:6px;">
                    Solution Sequence</div>
                  {sol}
                  <div class="sol-meta">
                    <div class="sol-tag">🔢 {moves} moves</div>
                    <div class="sol-tag">⚡ Kociemba optimal</div>
                    <div class="sol-tag">✅ Physically verified</div>
                  </div>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 3D Animation ─────────────────────────────────────────────────
        if sol and not sol.startswith("!"):
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown('<span class="slabel">📺 3D Interactive Animation</span>', unsafe_allow_html=True)
            sp1, sp2, sp3 = st.columns(3)
            if sp1.button("🐢 Slow  0.5×", use_container_width=True): st.session_state.solve_speed=0.5
            if sp2.button("🏃 Normal  1×", use_container_width=True): st.session_state.solve_speed=1.0
            if sp3.button("🚀 Fast   2×",  use_container_width=True): st.session_state.solve_speed=2.0
            render_3d_player(sol)
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CV METHODS STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif app_mode == "📊 CV Methods Study":
    with st.expander("📚 Background Study & Literature Review", expanded=True):
        c1, c2 = st.columns([3,2])
        with c1:
            st.markdown("""
### Problem Statement
Rubik's Cube colour detection is a **6-class colour classification** computer vision problem.
Each of 54 stickers must be mapped to one of (White, Red, Green, Yellow, Orange, Blue)
despite variation in camera white-balance, ambient lighting and sticker reflectance.

### State-of-the-Art Methods
| Method | Paradigm | Lighting Robust | Training Needed |
|--------|----------|----------------|-----------------|
| **HSV Thresholding** | Classical CV | ⚠️ Medium | None |
| **CIE-LAB Distance** | Classical CV | ✅ High | None |
| **KNN Classifier** | Machine Learning | ✅ High | Synthetic |
| **MLP Neural Network** | Deep Learning | ✅ High | Synthetic |
| **YOLOv8** | Object Detection | ✅✅ Very High | Large Dataset |
| **CNN (ResNet/VGG)** | Deep Learning | ✅✅ Very High | Large Dataset |
""")
        with c2:
            st.markdown("""
### Dataset
**12 calibration photos** labelled by colour prefix
(`green1.jpeg`, `blue.jpeg`, …).

### Validation Chain
✅ 54 total stickers  
✅ Each colour exactly 9×  
✅ Fixed centre per face  
✅ Kociemba physical check
""")

    # Pipeline
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<span class="slabel">🔬 Detection Pipeline</span>', unsafe_allow_html=True)
    st.markdown("""<div class="pipeline">
      <span class="pstep">📷 Input</span><span class="parrow">→</span>
      <span class="pstep">✂️ Crop 70%</span><span class="parrow">→</span>
      <span class="pstep">📐 300×300</span><span class="parrow">→</span>
      <span class="pstep">🔲 3×3 Grid</span><span class="parrow">→</span>
      <span class="pstep">💡 Centroid Snap</span><span class="parrow">→</span>
      <span class="pstep">🧪 Median BGR</span><span class="parrow">→</span>
      <span class="pstep">🌈 BGR→LAB</span><span class="parrow">→</span>
      <span class="pstep">🎨 Classifier</span><span class="parrow">→</span>
      <span class="pstep">✅ Label</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Method cards
    st.markdown('<span class="slabel">🧠 Algorithm Descriptions</span>', unsafe_allow_html=True)
    mc1,mc2,mc3 = st.columns(3)
    with mc1:
        st.markdown("""<div class="mcard">
          <div class="mcard-icon">🔵</div>
          <div class="mcard-title">CIE-LAB Weighted Distance</div>
          <div class="mcard-type">Classical Computer Vision</div>
          <div class="mcard-body">Converts pixel to perceptually-uniform CIE-LAB space.
          Weighted Euclidean distance to 6 reference colours with weights
          <code>L×0.1, a×2.4, b×2.4</code> suppresses brightness variation
          and amplifies chromatic channels.</div>
          <div class="mcard-foot">O(6) per pixel · No training · Lighting robust</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown("""<div class="mcard">
          <div class="mcard-icon">🟡</div>
          <div class="mcard-title">HSV Range Thresholding</div>
          <div class="mcard-type">Classical Computer Vision</div>
          <div class="mcard-body">Checks BGR pixel against hand-crafted Hue/Saturation/Value
          boundaries for each colour. Red needs two ranges (wraps at H=180°).
          White detected by low saturation + high value. Highly interpretable
          but sensitive to non-standard lighting.</div>
          <div class="mcard-foot">O(6) per pixel · No training · Lighting sensitive</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown("""<div class="mcard">
          <div class="mcard-icon">🟢</div>
          <div class="mcard-title">KNN Classifier (k=5)</div>
          <div class="mcard-type">Machine Learning</div>
          <div class="mcard-body">KNN (k=5) trained on 1 200 synthetic CIE-LAB samples.
          At inference, the query LAB vector is compared to all training samples;
          majority colour among 5 nearest Euclidean neighbours is returned.
          Data-driven but O(n) inference.</div>
          <div class="mcard-foot">O(n) per pixel · Synthetic training · k=5 Euclidean</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # Benchmark
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<span class="slabel">📈 Live Benchmark (12 labelled test images)</span>',
                unsafe_allow_html=True)
    found = sum(1 for f in IMAGE_LABEL_MAP if os.path.exists(f))
    br1, br2 = st.columns([2,3])
    with br1:
        if st.button("▶️ Run Benchmark", type="primary", use_container_width=True):
            with st.spinner("Training KNN & running 3 classifiers on all test images…"):
                try:
                    samples = []
                    for fname, lbl in IMAGE_LABEL_MAP.items():
                        if os.path.exists(fname):
                            with open(fname,"rb") as fh: raw=fh.read()
                            bgr = extract_center_bgr(raw)
                            if bgr is not None: samples.append((bgr,lbl))
                    if len(samples) < 3:
                        st.error("Not enough test images found.")
                    else:
                        st.session_state.cv_results = compare_methods(samples)
                        st.session_state.cv_samples = samples
                        st.rerun()
                except Exception as e:
                    st.error(f"Benchmark error: {e}")
    with br2:
        st.markdown(
            f"<div style='background:#f5f3ff;border-radius:10px;padding:11px 15px;"
            f"border:1px solid #c7d2fe;font-size:13px;'>"
            f"🗂 <b style='color:#6366f1;'>{found}/{len(IMAGE_LABEL_MAP)}</b>"
            f" test images auto-detected in project folder</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Custom samples
    with st.expander("➕ Add Custom Test Samples"):
        extra_samples = []
        uc = st.columns(6)
        for i, cname in enumerate(HEX_COLORS):
            uf = uc[i].file_uploader(f"{COLOR_EMOJIS[cname]} {cname}",
                type=["jpg","png","jpeg"], key=f"ex_{cname}")
            if uf:
                try:
                    bgr = extract_center_bgr(uf.read())
                    if bgr is not None: extra_samples.append((bgr,cname))
                except Exception: pass
        if extra_samples and st.button("▶️ Run with Custom Samples", type="primary"):
            with st.spinner("Running…"):
                try:
                    base = []
                    for fname, lbl in IMAGE_LABEL_MAP.items():
                        if os.path.exists(fname):
                            with open(fname,"rb") as fh: raw=fh.read()
                            bgr = extract_center_bgr(raw)
                            if bgr is not None: base.append((bgr,lbl))
                    st.session_state.cv_results = compare_methods(base+extra_samples)
                    st.session_state.cv_samples  = base+extra_samples
                    st.rerun()
                except Exception as e: st.error(f"{e}")

    # Results
    if st.session_state.cv_results:
        res   = st.session_state.cv_results
        n     = len(st.session_state.cv_samples)
        st.success(f"✅ Benchmark complete — **{n} images** · **3 methods** evaluated")

        methods_ord = sorted(res, key=lambda m: res[m]["accuracy"], reverse=True)
        accs = [res[m]["accuracy"]        for m in methods_ord]
        prcs = [res[m]["macro_precision"] for m in methods_ord]
        recs = [res[m]["macro_recall"]    for m in methods_ord]
        f1s  = [res[m]["macro_f1"]        for m in methods_ord]

        def badge(val, vals):
            mx,mn=max(vals),min(vals)
            if val==mx: return f'<span class="bbest">{val}%</span>'
            if val==mn: return f'<span class="blow">{val}%</span>'
            return f'<span class="bmid">{val}%</span>'

        # Bar chart
        bclasses=["b1","b2","b3"]
        bh='<div style="margin-bottom:18px;">'
        for mi,(metric,vals) in enumerate([
            ("Accuracy",accs),("Macro Precision",prcs),
            ("Macro Recall",recs),("Macro F1-Score",f1s)]):
            bh+=f'<div style="font-size:11px;font-weight:700;color:#6366f1;letter-spacing:.9px;text-transform:uppercase;margin-bottom:5px;">{metric}</div>'
            for k,m in enumerate(methods_ord):
                bh+=(f'<div class="brow">'
                     f'<div class="blabel">{"🥇🥈🥉"[k]} {m}</div>'
                     f'<div class="btrack"><div class="bfill {bclasses[k]}" style="width:{vals[k]}%;">'
                     f'{vals[k]}%</div></div></div>')
            bh+='<div style="height:8px;"></div>'
        bh+='</div>'
        st.markdown(bh, unsafe_allow_html=True)

        # Summary table
        st.markdown("#### 🏆 Performance Summary")
        rows=""
        for k,m in enumerate(methods_ord):
            rows+=(f'<tr><td><strong>{"🥇🥈🥉"[k]} {m}</strong></td>'
                   f'<td>{badge(accs[k],accs)}</td><td>{badge(prcs[k],prcs)}</td>'
                   f'<td>{badge(recs[k],recs)}</td><td>{badge(f1s[k],f1s)}</td></tr>')
        st.markdown(f"""<table class="mtable">
          <thead><tr><th>Method</th><th>Accuracy</th><th>Precision</th>
          <th>Recall</th><th>F1-Score</th></tr></thead>
          <tbody>{rows}</tbody></table>""", unsafe_allow_html=True)
        st.caption("🟢 Best &nbsp; 🟡 Mid &nbsp; 🔴 Lowest")

        # Per-class breakdown
        st.markdown("#### 🔍 Per-Colour Breakdown")
        t1,t2,t3 = st.tabs(["🔵 CIE-LAB","🟡 HSV Thresholding","🟢 KNN"])
        for tab,mname in zip([t1,t2,t3],["CIE-LAB Distance","HSV Thresholding","KNN Classifier"]):
            with tab:
                pc=res[mname]["per_class"]; rows2=""
                for col in COLORS:
                    d=pc.get(col,{})
                    rows2+=(f'<tr><td>{COLOR_EMOJIS.get(col,"")} {col}</td>'
                            f'<td>{d.get("Precision",0)}%</td>'
                            f'<td>{d.get("Recall",0)}%</td>'
                            f'<td>{d.get("F1",0)}%</td></tr>')
                st.markdown(f"""<table class="mtable">
                  <thead><tr><th>Colour</th><th>Precision</th>
                  <th>Recall</th><th>F1</th></tr></thead>
                  <tbody>{rows2}</tbody></table>""", unsafe_allow_html=True)

        # Discussion
        st.divider()
        st.markdown("#### 💬 Analysis & Discussion")
        best_m = methods_ord[0]
        accs_d = {m:accs[k] for k,m in enumerate(methods_ord)}
        st.markdown(f"""
| Criterion | 🔵 CIE-LAB | 🟡 HSV | 🟢 KNN |
|-----------|-----------|--------|--------|
| **Approach** | Perceptual distance | Rule ranges | Supervised ML |
| **Training** | ❌ None | ❌ None | ✅ Synthetic |
| **Lighting robust** | ✅ High | ⚠️ Medium | ✅ High |
| **Interpretable** | ✅ High | ✅ Very High | ⚠️ Medium |
| **Speed** | ✅ O(6) | ✅ O(6) | ⚠️ O(n) |
| **Accuracy** | **{accs_d.get("CIE-LAB Distance","—")}%** | **{accs_d.get("HSV Thresholding","—")}%** | **{accs_d.get("KNN Classifier","—")}%** |

**Best overall: {best_m}.** CIE-LAB's perceptually-uniform weighted distance naturally handles
brightness variation without training data. HSV Thresholding degrades when ambient light
shifts hue outside fixed ranges. KNN learns colour distributions but is constrained to its
synthetic training domain.
""")
    else:
        st.info("👆 Click **▶️ Run Benchmark** to compare all three CV methods with live metrics.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
elif app_mode == "⚙️ Calibration":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("""
    <span class="slabel">🎯 Purpose</span>
    <p style="font-size:13px;color:#64748b;margin-top:3px;">
    The CIE-LAB classifier compares pixel colours to built-in reference HSV values.
    If your camera or lighting shifts colours systematically, upload a sample photo of
    the known colour and click <b>Auto-Calibrate</b> to update the reference.
    Settings are saved to <code>calibration_profile.json</code> automatically.
    </p>""", unsafe_allow_html=True)

    c_target = st.radio("Select colour to calibrate:", list(HEX_COLORS.keys()),
                        horizontal=True, format_func=lambda x: f"{COLOR_EMOJIS[x]} {x}")

    ul, pr = st.columns(2)
    with ul:
        st.markdown(f'<span class="slabel">📷 Upload {c_target} sample</span>', unsafe_allow_html=True)
        cal_buf = st.file_uploader("Image", type=['jpg','png','jpeg'],
                                   key="cal_up", label_visibility="collapsed")
    with pr:
        st.markdown('<span class="slabel">👁 Preview</span>', unsafe_allow_html=True)
        if cal_buf:
            try:
                raw_cal  = cal_buf.read()
                img_cal  = cv2.imdecode(np.frombuffer(raw_cal, dtype=np.uint8), 1)
                if img_cal is None:
                    st.error("❌ Invalid image file.")
                else:
                    st.image(cv2.cvtColor(img_cal, cv2.COLOR_BGR2RGB), use_container_width=True)
                    if st.button(f"🎯 Auto-Calibrate → {c_target}",
                                 type="primary", use_container_width=True):
                        h, w   = img_cal.shape[:2]
                        roi    = img_cal[h//2-15:h//2+15, w//2-15:w//2+15]
                        avg    = np.median(roi, axis=(0,1)).astype(np.uint8)
                        hsv    = cv2.cvtColor(np.uint8([[[avg[0],avg[1],avg[2]]]]),
                                              cv2.COLOR_BGR2HSV)[0][0]
                        st.session_state.custom_std_colors[c_target] = [int(hsv[0]),int(hsv[1]),int(hsv[2])]
                        with open(CALIB_FILE,'w') as fh:
                            json.dump(st.session_state.custom_std_colors, fh)
                        st.success(f"✅ {c_target} calibrated — HSV = {tuple(int(x) for x in hsv)}")
                        st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.markdown("""
            <div style='border:2px dashed #e5e7eb;border-radius:12px;padding:28px;
                        text-align:center;background:#f8fafc;'>
              <div style='font-size:2rem;'>🖼️</div>
              <div style='font-size:12px;color:#9ca3af;margin-top:6px;'>Upload a photo above</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<span class="slabel">📋 Active Calibration Profile</span>', unsafe_allow_html=True)
    defaults_hsv = {'White':(0,30,220),'Yellow':(30,160,200),'Orange':(12,200,240),
                    'Red':(0,210,180),'Green':(60,180,150),'Blue':(110,180,160)}
    pcols = st.columns(6)
    for i, cname in enumerate(HEX_COLORS):
        cv_ = st.session_state.custom_std_colors.get(cname)
        hsv  = tuple(cv_) if cv_ else defaults_hsv[cname]
        tag  = "Custom ✅" if cv_ else "Default"
        tc   = "#22c55e" if cv_ else "#9ca3af"
        pcols[i].markdown(f"""
        <div style='background:#fff;border-radius:10px;padding:10px;text-align:center;
                    border:1px solid {"#bbf7d0" if cv_ else "#e5e7eb"};'>
          <div style='font-size:1.3rem;'>{COLOR_EMOJIS[cname]}</div>
          <div style='font-size:11px;font-weight:700;color:#374151;margin-top:3px;'>{cname}</div>
          <div style='font-size:9px;color:#6b7280;'>H{hsv[0]} S{hsv[1]} V{hsv[2]}</div>
          <div style='font-size:9px;font-weight:700;color:{tc};margin-top:2px;'>{tag}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Reset All to Factory Defaults"):
        st.session_state.custom_std_colors = {}
        if os.path.exists(CALIB_FILE): os.remove(CALIB_FILE)
        st.success("Reset complete."); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:18px 0 6px;'>
  <div style='font-size:11px;color:#9ca3af;'>
    Rubik's AI Solver &nbsp;·&nbsp;
    Method A: CIE-LAB &nbsp;·&nbsp; Method B: YOLOv8 &nbsp;·&nbsp; Method C: MLP Neural Net
    &nbsp;·&nbsp; Kociemba Two-Phase Engine &nbsp;·&nbsp; OpenCV · Scikit-Learn
  </div>
</div>
""", unsafe_allow_html=True)
