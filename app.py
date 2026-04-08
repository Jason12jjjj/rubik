# ==============================================================================
# RUBIK'S CUBE SOLVER - PREMIUM INTERACTIVE VERSION
# ==============================================================================
import io
import os
import json
import numpy as np
import cv2
import streamlit as st
import streamlit.components.v1 as components
from streamlit_image_coordinates import streamlit_image_coordinates
from rubiks_core import validate_cube_state, solve_cube

# --- 1. System Configuration & State Management ---
st.set_page_config(page_title="AI Rubik's Solver", page_icon="🧊", layout="wide")

# Handle Interactive Map Clicks via Query Parameters
if "face_click" in st.query_params:
    clicked_face = st.query_params["face_click"]
    st.session_state.programmatic_face = clicked_face
    # Clear the parameter to avoid loop-reruns
    st.query_params.clear()

AVAILABLE_COLORS = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
FACES = ['Up', 'Left', 'Front', 'Right', 'Back', 'Down']
COLOR_EMOJIS = {'White': '⬜', 'Red': '🟥', 'Green': '🟩', 'Yellow': '🟨', 'Orange': '🟧', 'Blue': '🟦'}
HEX_COLORS = {'White': '#f8f9fa', 'Red': '#ff4b4b', 'Green': '#09ab3b', 'Yellow': '#ffeb3b', 'Orange': '#ffa500', 'Blue': '#1e88e5'}
CENTER_COLORS = {'Up': 'White', 'Left': 'Orange', 'Front': 'Green', 'Right': 'Red', 'Back': 'Blue', 'Down': 'Yellow'}
COLOR_TO_FACE_NAME = {v: k for k, v in CENTER_COLORS.items()}
CALIB_FILE = "calibration_profile.json"

ORIENTATION_GUIDE = {
    'Up':    "Point the camera at the **⬜ White** face. Rotate the cube so the adjacent **🟦 Blue** face is on the **TOP edge ⬆️** of the camera screen.",
    'Left':  "Point the camera at the **🟧 Orange** face. Rotate the cube so the adjacent **⬜ White** face is on the **TOP edge ⬆️** of the camera screen.",
    'Front': "Point the camera at the **🟩 Green** face. Rotate the cube so the adjacent **⬜ White** face is on the **TOP edge ⬆️** of the camera screen.",
    'Right': "Point the camera at the **🟥 Red** face. Rotate the cube so the adjacent **⬜ White** face is on the **TOP edge ⬆️** of the camera screen.",
    'Back':  "Point the camera at the **🟦 Blue** face. Rotate the cube so the adjacent **⬜ White** face is on the **TOP edge ⬆️** of the camera screen.",
    'Down':  "Point the camera at the **🟨 Yellow** face. Rotate the cube so the adjacent **🟩 Green** face is on the **TOP edge ⬆️** of the camera screen."
}

# --- Initialize Session State ---
if 'cube_state' not in st.session_state:
    st.session_state.cube_state = {f: (['White'] * 4 + [CENTER_COLORS[f]] + ['White'] * 4) for f in FACES}

if 'processed_photos' not in st.session_state:
    st.session_state.processed_photos = {}

if 'custom_std_colors' not in st.session_state:
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, 'r') as f:
                st.session_state.custom_std_colors = json.load(f)
        except Exception:
            st.session_state.custom_std_colors = {}
    else:
        st.session_state.custom_std_colors = {}

if 'cube_size' not in st.session_state:
    st.session_state.cube_size = 50

if 'shared_face_images' not in st.session_state:
    st.session_state.shared_face_images = {}

if 'auto_detect' not in st.session_state:
    st.session_state.auto_detect = True

if 'auto_advance' not in st.session_state:
    st.session_state.auto_advance = True

if 'last_solution' not in st.session_state:
    st.session_state.last_solution = None

if 'uploader_key_version' not in st.session_state:
    st.session_state.uploader_key_version = 0

if 'programmatic_face' not in st.session_state:
    st.session_state.programmatic_face = FACES[0]

# --- Global Styles ---
st.markdown("""
<style>
    [data-testid="stCameraInput"] { position: relative; }
    [data-testid="stCameraInput"]::after { display: none !important; }
    .stButton>button { border-radius: 8px; }
    
    /* Make the clickable map links look like buttons */
    .face-link {
        text-decoration: none !important;
        color: inherit !important;
        display: block;
        transition: transform 0.15s ease;
    }
    .face-link:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# ── ADVANCED COMPUTER VISION (WARP PERSPECTIVE VERSION) ──────────────────────
def auto_detect_cube_face(image_bytes, expected_center, show_diag=False):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None, None, "No Image Data", {}, None
    
    diag_imgs = {}
    pad = 40
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    h_p, w_p = img_padded.shape[:2]
    
    # Standardize work height for consistent metric thresholds
    work_h = 650
    work_w = int(w_p * (work_h / h_p))
    work_img = cv2.resize(img_padded, (work_w, work_h))
    tracking_img = work_img.copy()

    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 💎 Industrial Extraction: Find all sticker candidates
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 3)
    if show_diag: diag_imgs['Feature Map'] = thresh
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if (work_h*0.01)**2 < area < (work_h*0.3)**2:
            x, y, w, h = cv2.boundingRect(c)
            ratio = min(w, h) / max(w, h)
            if ratio > 0.4:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    candidates.append({'cnt': c, 'center': (cx, cy), 'area': area})

    best_cnt = None
    pass_info = f"Found {len(candidates)} features."
    
    # 💎 Grid Discovery: Find the best 3x3 cluster
    if len(candidates) >= 4:
        # Sort candidates to find clusters (naive spatial clustering)
        # We look for a group that forms a quadrilateral
        all_pts = np.array([c['center'] for c in candidates])
        hull = cv2.convexHull(all_pts)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
        
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if (work_h*work_w * 0.005) < area < (work_h*work_w * 0.85):
                best_cnt = approx.reshape(4, 2)
                pass_info = f"Grid Locked: {len(candidates)} features in area."
        else:
            # Fallback to largest bounding box of features
            (cx, cy), (w, h), angle = cv2.minAreaRect(all_pts)
            if (work_h*work_w * 0.005) < (w*h) < (work_h*work_w * 0.85):
                mpts = cv2.boxPoints(((cx, cy), (w, h), angle))
                best_cnt = np.int32(mpts)
                pass_info = f"Grid Fallback: {len(candidates)} features."

    # Final Fallback to Edge Detection if stickers are too faint
    if best_cnt is None:
        edges = cv2.dilate(cv2.Canny(blurred, 20, 100), np.ones((5,5), np.uint8))
        if show_diag: diag_imgs['Edge Guide'] = edges
        cnts_e, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_a = 0
        for ce in cnts_e:
            area = cv2.contourArea(ce)
            if (work_h*work_w * 0.005) < area < (work_h*work_w * 0.85):
                hull_e = cv2.convexHull(ce)
                approx_e = cv2.approxPolyDP(hull_e, 0.05 * cv2.arcLength(hull_e, True), True)
                if len(approx_e) == 4:
                    best_cnt = approx_e.reshape(4, 2)
                    pass_info = "Outline Locked (Pass 2)"
                    break

    if best_cnt is None: return None, None, pass_info, diag_imgs, None

    # --- Precise Homography Pass ---
    # Map from Work Image to Padded Image coords
    scale = h_p / work_h
    best_cnt_p = (best_cnt.astype("float32") * scale)
    
    # Sort corners (TL, TR, BR, BL)
    rect = np.zeros((4, 2), dtype="float32")
    s = best_cnt_p.sum(axis=1)
    rect[0], rect[2] = best_cnt_p[np.argmin(s)], best_cnt_p[np.argmax(s)]
    diff = np.diff(best_cnt_p, axis=1)
    rect[1], rect[3] = best_cnt_p[np.argmin(diff)], best_cnt_p[np.argmax(diff)]
    
    dst = np.array([[0,0], [299,0], [299,299], [0,299]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_padded, M, (300, 300))
    debug_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    # --- Deep Alignment Refinement (Saturation Centroids) ---
    std_colors = get_calibrated_colors()
    detected = ['White'] * 9
    refined_pts_warped = [] 
    hsv_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    sat_map = hsv_warped[:,:,1]
    
    grid_metrics = [] # Meta for transparency
    
    for r in range(3):
        for c in range(3):
            tx, ty = int((c+0.5)*100), int((r+0.5)*100)
            win = 35
            y1, y2, x1, x2 = max(0,ty-win), min(300,ty+win), max(0,tx-win), min(300,tx+win)
            roi_sat = sat_map[y1:y2, x1:x2]
            moms = cv2.moments(roi_sat)
            
            snap_dx, snap_dy = 0, 0
            if moms["m00"] > 50:
                sx, sy = x1 + int(moms["m10"]/moms["m00"]), y1 + int(moms["m01"]/moms["m00"])
                dist_snap = np.sqrt((sx-tx)**2 + (sy-ty)**2)
                if dist_snap < 30:
                    final_x, final_y = sx, sy
                    snap_dx, snap_dy = sx - tx, sy - ty
                else: 
                    final_x, final_y = tx, ty
            else:
                final_x, final_y = tx, ty
            
            refined_pts_warped.append((final_x, final_y))
            roi = warped[final_y-6:final_y+6, final_x-6:final_x+6]
            bgr_avg = np.median(roi, axis=(0,1)).astype(np.uint8) if roi.size > 0 else [0,0,0]
            lab = cv2.cvtColor(np.uint8([[[bgr_avg[0], bgr_avg[1], bgr_avg[2]]]]), cv2.COLOR_BGR2LAB)[0][0]
            
            # --- Color Scoring & Metadata ---
            dists = {}
            min_d, best_c = float('inf'), 'White'
            for name, (hs,ss,vs) in std_colors.items():
                sl = cv2.cvtColor(cv2.cvtColor(np.uint8([[[hs,ss,vs]]]), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2LAB)[0][0]
                dL, da, db = float(lab[0])-float(sl[0]), float(lab[1])-float(sl[1]), float(lab[2])-float(sl[2])
                d_val = np.round(np.sqrt(0.1*(dL**2) + 2.0*(da**2) + 2.0*(db**2)), 1)
                dists[name] = d_val
                if d_val < min_d: min_d, best_c = d_val, name
            
            detected[r*3+c] = best_c
            grid_metrics.append({
                'slot': r*3+c,
                'final_color': best_c,
                'distances': dists,
                'snap_offset': (snap_dx, snap_dy),
                'confidence': np.round(100 - min_d, 1)
            })
            
            short = f"{COLOR_EMOJIS[best_c][0]}"
            cv2.circle(debug_warped, (final_x, final_y), 6, (255,255,255), 1)
            cv2.putText(debug_warped, f"{best_c[0]}", (final_x-5, final_y+5), 0, 0.4, (255,255,255), 1, cv2.LINE_AA)

    # --- AR Overlay (Full Resolution Feature-Locked) ---
    best_cnt_scale = (best_cnt.astype("float32") * (h_p / work_h))
    pts_w = best_cnt_scale.reshape(4, 2)
    s = pts_w.sum(axis=1)
    diff = np.diff(pts_w, axis=1)
    rect_w = np.zeros((4, 2), dtype="float32")
    rect_w[0], rect_w[2] = pts_w[np.argmin(s)], pts_w[np.argmax(s)]
    rect_w[1], rect_w[3] = pts_w[np.argmin(diff)], pts_w[np.argmax(diff)]
    M_inv_full = cv2.getPerspectiveTransform(dst, rect_w)
    
    ar_img = img_padded.copy()
    draw_map = {'White': (255,255,255), 'Yellow': (0,255,255), 'Orange': (0,165,255), 'Red': (0,0,255), 'Green': (0,255,0), 'Blue': (255,0,0)}
    dot_radius = max(4, int(h_p / 150))
    font_scale = h_p / 1200.0

    for i, (fx, fy) in enumerate(refined_pts_warped):
        p_orig = cv2.perspectiveTransform(np.array([[[fx, fy]]], dtype="float32"), M_inv_full)
        px, py = int(p_orig[0][0][0]), int(p_orig[0][0][1])
        c_name = detected[i]
        bgr = draw_map.get(c_name, (0, 255, 0))
        cv2.circle(ar_img, (px, py), dot_radius, bgr, -1)
        cv2.circle(ar_img, (px, py), dot_radius + 2, (255,255,255), max(1, int(dot_radius/4)))
        label_pos = (px + dot_radius + 5, py + dot_radius + 5)
        cv2.putText(ar_img, c_name[0], label_pos, 0, font_scale, (0,0,0), int(font_scale*4), cv2.LINE_AA)
        cv2.putText(ar_img, c_name[0], label_pos, 0, font_scale, (255,255,255), int(font_scale*2), cv2.LINE_AA)

    final_ar = ar_img[pad:pad+h_p-2*pad, pad:pad+w_p-2*pad]
    detected[4] = expected_center
    return detected, debug_warped, pass_info, diag_imgs, cv2.cvtColor(final_ar, cv2.COLOR_BGR2RGB), grid_metrics

def run_manual_grid_extract(image_bytes, expected_center, scale_percent):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None, None, False
    
    h, w = img.shape[:2]
    # Use the scale requested by user or 50% default
    gs = int(min(h, w) * (scale_percent / 100.0))
    sx, sy = (w - gs) // 2, (h - gs) // 2
    
    # Crop and resize to 300x300 to unify UI with Auto-detect
    cropped = img[sy:sy+gs, sx:sx+gs]
    warped = cv2.resize(cropped, (300, 300))
    debug_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    std_colors = get_calibrated_colors()
    detected = ['White'] * 9
    cell_sz = 100
    for r in range(3):
        for c in range(3):
            cx, cy = int((c+0.5)*cell_sz), int((r+0.5)*cell_sz)
            roi = warped[cy-10:cy+10, cx-10:cx+10]
            if roi.size == 0: continue
            b_a, g_a, r_a = np.median(roi, axis=(0,1)).astype(np.uint8)
            lab = cv2.cvtColor(np.uint8([[[b_a,g_a,r_a]]]), cv2.COLOR_BGR2LAB)[0][0]
            
            min_d, best = float('inf'), 'White'
            for name, (hs,ss,vs) in std_colors.items():
                sl = cv2.cvtColor(cv2.cvtColor(np.uint8([[[hs,ss,vs]]]), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2LAB)[0][0]
                d = np.sqrt(0.5*(int(lab[0])-int(sl[0]))**2 + 1.25*(int(lab[1])-int(sl[1]))**2 + (int(lab[2])-int(sl[2]))**2)
                if d < min_d: min_d, best = d, name
            detected[r*3+c] = best
            cv2.circle(debug_warped, (cx, cy), 15, (255, 255, 255), 2)
            cv2.putText(debug_warped, best, (cx-30, cy+35), 0, 0.5, (255,255,255), 1)

    detected[4] = expected_center
    return detected, debug_warped, True

# ── LEGACY CV FUNCTIONS (Kept for manual override fallback) ──────────────────
def get_calibrated_colors():
    defaults = {'White':(0,30,220), 'Yellow':(30,160,200), 'Orange':(12,200,240), 'Red':(0,210,180), 'Green':(60,180,150), 'Blue':(110,180,160)}
    for k, v in st.session_state.custom_std_colors.items(): defaults[k] = tuple(v)
    return defaults

# ── UI COMPONENTS ────────────────────────────────────────────────────────────
def render_3d_solution(solution_str, speed):
    def inv(s):
        r = []
        for m in reversed(s.split()):
            if "'" in m: r.append(m.replace("'",""))
            elif "2" in m: r.append(m)
            else: r.append(m+"'")
        return " ".join(r)
    html = f"""<script src="https://cdn.cubing.net/v0/js/cubing/twisty" type="module"></script>
    <twisty-player experimental-setup-alg="{inv(solution_str)}" alg="{solution_str}" visualization="PG3D" control-panel="bottom-row" tempo-scale="{speed}" background="none" style="width:100%; height:380px;"></twisty-player>"""
    components.html(html, height=400)

def render_interactive_map(active_face):
    grid = {'Up':(0,1), 'Left':(1,0), 'Front':(1,1), 'Right':(1,2), 'Back':(1,3), 'Down':(2,1)}
    html = '<div style="display:grid;grid-template-columns:repeat(4,60px);gap:8px;justify-content:center;font-family:sans-serif;">'
    for r in range(3):
        for c in range(4):
            f_k = next((f for f, p in grid.items() if p == (r, c)), None)
            if f_k:
                colors = st.session_state.cube_state[f_k]
                is_sc = (f_k in st.session_state.processed_photos)
                active_style = "border:3px solid #00e5ff; box-shadow:0 0 10px #00e5ff;" if f_k == active_face else "border:1px solid rgba(255,255,255,0.2);"
                html += f"""<a href="/?face_click={f_k}" target="_self" class="face-link">
                    <div style="opacity:{1.0 if (is_sc or f_k==active_face) else 0.4}; text-align:center;">
                        <div style="font-size:10px; font-weight:bold; color:{'#00e5ff' if f_k==active_face else '#ddd'}; margin-bottom:2px;">{COLOR_EMOJIS[CENTER_COLORS[f_k]]} {f_k} {'✅' if is_sc else ''}</div>
                        <div style="display:grid;grid-template-columns:repeat(3,18px);gap:2px; {active_style} border-radius:4px; padding:3px; background:rgba(255,255,255,0.05);">"""
                for clr in colors:
                    html += f'<div style="width:18px;height:18px;background:{HEX_COLORS[clr]};border:1px solid rgba(0,0,0,0.3);border-radius:2px;"></div>'
                html += '</div></div></a>'
            else: html += '<div></div>'
    html += '</div>'
    return html

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    app_mode = st.radio("Mode:", ["📸 Scan & Solve", "⚙️ Tune Colors"], label_visibility="collapsed")
    st.divider()
    if app_mode == "📸 Scan & Solve":
        st.markdown("### ⚙️ Settings")
        st.session_state.auto_advance = st.checkbox("⏩ Auto-advance on success", value=st.session_state.auto_advance)
        st.divider()
        st.markdown("### 🗺️ Interactive Map")
        st.caption("Click a face below to jump to its scanner.")
        st.markdown(render_interactive_map(st.session_state.programmatic_face), unsafe_allow_html=True)
        st.divider()
        with st.expander("🛠️ Advanced Tools"):
            st.session_state.show_diag = st.toggle("🔍 Diagnostic Vision", value=st.session_state.get('show_diag', False), help="Show intermediate CV steps to help debug detection issues.")
            st.session_state.cube_size = st.slider("Manual Grid Size (%)", 10, 100, st.session_state.cube_size, help="Adjust the fixed overlay size in Manual Mode.")
            if st.button("🗑️ Reset All Scans", use_container_width=True):
                st.session_state.processed_photos = {}
                st.session_state.shared_face_images = {}
                st.rerun()
    else:
        st.markdown("Map is disabled in Calibration mode.")

# ── MAIN SCANNER ─────────────────────────────────────────────────────────────
if app_mode == "📸 Scan & Solve":
    curr = st.session_state.programmatic_face
    st.title("🧊 AI Rubik's Solver")

    # Header Nav
    hcol1, hcol2, hcol3 = st.columns([1, 2, 1])
    with hcol1: 
        if st.button("⬅️ Previous", use_container_width=True): 
            st.session_state.programmatic_face = FACES[(FACES.index(curr)-1)%6]; st.rerun()
    with hcol2: st.markdown(f"<h2 style='text-align:center; margin-top:-10px;'>{COLOR_EMOJIS[CENTER_COLORS[curr]]} {curr} Face</h2>", unsafe_allow_html=True)
    with hcol3: 
        if st.button("Next ➡️", use_container_width=True): 
            st.session_state.programmatic_face = FACES[(FACES.index(curr)+1)%6]; st.rerun()

    if st.session_state.get('last_face') != curr:
        st.session_state.last_face = curr
        st.session_state.uploader_key_version += 1

    st.info(f"🧭 **HOLD:** {ORIENTATION_GUIDE[curr]}")
    c_cam, c_edit = st.columns([1, 1])
    
    with c_cam:
        st.markdown("### 📷 Camera Input")
        sh = st.session_state.shared_face_images.get(curr)
        if sh: st.success("♻️ Image shared/demo loaded.")
        
        imth = st.radio("Method:", ["📹 Live", "📂 Upload"], horizontal=True, label_visibility="collapsed")
        st.session_state.auto_detect = st.toggle("🤖 Auto-Detect", value=st.session_state.auto_detect)
        
        v = st.session_state.uploader_key_version
        buf = st.camera_input("P", key=f"c_{v}") if imth == "📹 Live" else st.file_uploader("U", type=['jpg','png'], key=f"u_{v}")
        
        act = buf or (io.BytesIO(sh) if sh and imth == "📂 Upload" else None)
        if act:
            key = f"{curr}_{getattr(act, 'file_id', id(act))}"
            if st.session_state.processed_photos.get(curr) != key:
                if hasattr(act, 'seek'): act.seek(0)
                
                # BRANCH: Auto-Detect vs Manual Alignment
                msg_success = "✨ **I flattened the cube face!** Check the orientation and colors."
                msg_fail = "❌ Cube not found. Try to hold it flatter, or switch to Manual Mode."
                
                if st.session_state.auto_detect:
                    res = auto_detect_cube_face(act, CENTER_COLORS[curr], show_diag=st.session_state.get('show_diag'))
                    # Unpack 6 values: detected, debug_warped, info, diag_imgs, track_img, grid_metrics
                    d, db, info, diags, track, metrics = res if len(res) == 6 else (None, None, "API Error", {}, None, None)
                    success = (d is not None)
                    
                    if success:
                        st.session_state.cube_state[curr] = d
                        t_col, r_col = st.columns([1, 1])
                        with t_col: st.image(track, caption="🎯 Target Locked", use_container_width=True)
                        with r_col: st.image(db, caption="📸 Color Result", use_container_width=True)
                        st.success(msg_success)
                        
                        # 🔬 NEW: High-Transparency Diagnostic HUD
                        if st.session_state.get('show_diag') and metrics:
                            with st.expander("🔬 🔬 View Detailed Vision Metrics (HUD)", expanded=True):
                                st.markdown("### 🧬 3x3 Grid Analysis")
                                df_data = []
                                for m in metrics:
                                    row = {"Slot": m['slot']+1, "Color": m['final_color'], "Snap_XY": f"{m['snap_offset']}", "Conf%": m['confidence']}
                                    # Add all 6 color distances
                                    for color_name, dist in m['distances'].items():
                                        row[f"d({color_name[0]})"] = dist
                                    df_data.append(row)
                                st.table(df_data)
                                st.caption("💡 **Legend:** d(C) = Delta-E Distance (lower is closer). Snap_XY = offset from theoretical center.")
                        
                    else:
                        st.error(f"❌ Cube not found. (Diag: {info})")
                        if st.session_state.get('show_diag') and diags:
                            st.write("### 🖼️ Intermediate CV Steps")
                            dcols = st.columns(len(diags))
                            for i, (name, img) in enumerate(diags.items()):
                                with dcols[i]: st.image(img, caption=name, use_container_width=True)
                else:
                    d, db, success = run_manual_grid_extract(act, CENTER_COLORS[curr], st.session_state.cube_size)
                    info = "Manual"
                    msg_success = "📐 **Manual Alignment Used.**"
                    if success:
                        st.session_state.cube_state[curr] = d
                        st.image(db, caption="📸 Capture Preview", use_container_width=True)
                        st.success(msg_success)
                    msg_fail = "❌ Failed to read manual grid. Check your connection."

                if success:
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        if st.button("✅ Confirm & Save", type="primary", use_container_width=True, key=f"conf_{key}"):
                            st.session_state.processed_photos[curr] = key
                            st.session_state[f"cached_img_{curr}"] = db
                            if st.session_state.auto_advance:
                                un = [f for f in FACES if f not in st.session_state.processed_photos]
                                if un: st.session_state.programmatic_face = un[0]
                            st.rerun()
                    with col_b2:
                        if st.button("🔄 Retake", use_container_width=True, key=f"ign_{key}"):
                            st.session_state.uploader_key_version += 1; st.rerun()
                else:
                    st.error(msg_fail)
                    if hasattr(act, 'seek'): act.seek(0)
                    st.image(act, use_container_width=True)
                    if not st.session_state.auto_detect:
                        st.info("💡 Adjust the 'Manual Grid Size' slider in the sidebar to match the cube in the photo.")

        # Show the permanent saved image if it exists
        elif curr in st.session_state.processed_photos and f"cached_img_{curr}" in st.session_state:
            st.image(st.session_state[f"cached_img_{curr}"], caption="Saved Detection")

    with c_edit:
        st.markdown("### 🖱️ Edit Colors")
        st.caption("Click squares to cycle selection.")
        
        def cycle(i):
            seq = ['White', 'Yellow', 'Green', 'Blue', 'Red', 'Orange']
            cur_v = st.session_state.cube_state[curr][i]
            st.session_state.cube_state[curr][i] = seq[(seq.index(cur_v)+1)%len(seq)]
            st.session_state.last_solution = None

        for r in range(3):
            ecols = st.columns(3)
            for c in range(3):
                idx = r*3 + c
                with ecols[c]:
                    if idx == 4: st.button(f"{COLOR_EMOJIS[CENTER_COLORS[curr]]}\nCtr", disabled=True, use_container_width=True)
                    else:
                        v_c = st.session_state.cube_state[curr][idx]
                        st.button(f"{COLOR_EMOJIS[v_c]}\n{v_c}", key=f"e_{curr}_{idx}", use_container_width=True, on_click=cycle, args=(idx,))

        if st.button("🧹 Reset Face", use_container_width=True):
            st.session_state.cube_state[curr] = ['White']*9
            st.session_state.cube_state[curr][4] = CENTER_COLORS[curr]
            st.session_state.processed_photos.pop(curr, None)
            st.rerun()

    st.divider()
    if st.button("🚀 VALIDATE & SOLVE", type="primary", use_container_width=True):
        if len(st.session_state.processed_photos) < 6: st.warning("⚠️ Scan all 6 faces!")
        else:
            ok, msg = validate_cube_state(st.session_state.cube_state)
            if ok: 
                with st.spinner("Solving..."): st.session_state.last_solution = solve_cube(st.session_state.cube_state)
            else: st.error(f"❌ {msg}")

    sol = st.session_state.get('last_solution')
    if sol:
        if sol == "!IMPOSSIBLE_STATE!": st.error("❌ Physical Error: Impossible State. Check alignment.")
        else:
            st.success(f"✅ moves: {sol}")
            render_3d_solution(sol, st.select_slider("Speed", [0.5, 1.0, 2.0], 1.0))

elif app_mode == "⚙️ Tune Colors":
    st.title("⚙️ Color Calibration")
    c_cal = st.radio("Target:", AVAILABLE_COLORS, horizontal=True, format_func=lambda x: f"{COLOR_EMOJIS[x]} {x}")
    v = st.session_state.uploader_key_version
    cal_meth = st.radio("Method:", ["📹 Live", "📂 Upload"], horizontal=True, label_visibility="collapsed", key="cal_meth")
    if cal_meth == "📹 Live":
        cal_buf = st.camera_input("Sample", key=f"cal_{v}")
    else:
        cal_buf = st.file_uploader("Upload Image", type=['jpg','png'], key=f"ulc_{v}")
    if cal_buf:
        img_cal = cv2.imdecode(np.asarray(bytearray(cal_buf.read()), dtype=np.uint8), 1)
        if img_cal is not None:
            rgb_cal = cv2.cvtColor(img_cal, cv2.COLOR_BGR2RGB)
            val_xy = streamlit_image_coordinates(rgb_cal, key=f"click_{c_cal}_{v}")
            cx, cy = (val_xy['x'], val_xy['y']) if val_xy else (img_cal.shape[1]//2, img_cal.shape[0]//2)
            roi = img_cal[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
            if roi.size > 0:
                avg_bgr = np.median(roi, axis=(0,1)).astype(np.uint8)
                hsv_val = cv2.cvtColor(np.uint8([[[avg_bgr[0],avg_bgr[1],avg_bgr[2]]]]), cv2.COLOR_BGR2HSV)[0][0]
                cl1, cl2 = st.columns(2)
                with cl1:
                    cv2.circle(rgb_cal, (cx, cy), 15, (255,255,255), 3)
                    st.image(rgb_cal, caption="Click to sample surface color")
                with cl2:
                    st.markdown(f'<div style="width:100px;height:100px;background:rgb({avg_bgr[2]},{avg_bgr[1]},{avg_bgr[0]});border:2px solid #fff;"></div>', unsafe_allow_html=True)
                    if st.button(f"Save HSV {hsv_val} for {c_cal}", type="primary"):
                        st.session_state.custom_std_colors[c_cal] = [int(hsv_val[0]), int(hsv_val[1]), int(hsv_val[2])]
                        with open(CALIB_FILE, 'w') as f_json: json.dump(st.session_state.custom_std_colors, f_json)
                        st.success("Calibration Saved!")
