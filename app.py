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
def auto_detect_cube_face(image_bytes, expected_center):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None, None, False
    
    # 1. Padding: Add black border so edge-touching stickers are detected correctly
    pad = 40
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    h_p, w_p = img_padded.shape[:2]
    
    # Resize for processing speed
    work_h = 650
    work_w = int(w_p * (work_h / h_p))
    work_img = cv2.resize(img_padded, (work_w, work_h))

    # Pre-processing
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    best_cnt = None
    
    # Pass 1: Adaptive Threshold (Finding stickers)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cnts_s, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stks = []
    # Relaxed constraints for tilting/compression
    for c in cnts_s:
        a = cv2.contourArea(c)
        if (work_h*0.02)**2 < a < (work_h*0.4)**2:
            x,y,w,h = cv2.boundingRect(c)
            if 0.25 < w/max(1,h) < 4.0: 
                stks.append(c)

    total_area = work_h * work_w
    if len(stks) >= 4:
        # Use Rotated Bounding Box (minAreaRect)
        combined = np.vstack(stks)
        (cx, cy), (w, h), angle = cv2.minAreaRect(combined)
        
        # Area Qualification Check
        box_area = w * h
        if (total_area * 0.10 < box_area < total_area * 0.60):
            # Pass 1 Success: Manual calculation of box points
            theta = angle * np.pi / 180.0
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rect_pts = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
            best_cnt = np.array([
                [cx + p[0]*cos_t - p[1]*sin_t, cy + p[0]*sin_t + p[1]*cos_t] 
                for p in rect_pts
            ]).astype(np.int32)
        else:
            # Area invalid, let it fall through to Pass 2 (Outline Detection)
            best_cnt = None
    
    # Pass 2: Fallback to Canny + Dilation if Pass 1 found nothing or failed area check
    if best_cnt is None:
        edges = cv2.Canny(blurred, 20, 100)
        edges = cv2.dilate(edges, np.ones((3,3)))
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            # Pixel-Based Constraints: 10% < area < 60%
            if (total_area * 0.10) < area < (total_area * 0.60):
                peri = cv2.arcLength(cnt, True)
                appr = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(appr) == 4 and area > max_area:
                    max_area = area
                    best_cnt = appr.reshape(4, 2)

    if best_cnt is None: return None, None, False

    # --- Perspective Warping ---
    # Convert points from 'work' size back to 'padded' size
    pts_padded = (best_cnt.reshape(4, 2) * (h_p / work_h)).astype("float32")
    
    # Sort points for WarpPerspective (TL, TR, BR, BL)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts_padded.sum(axis=1)
    rect[0], rect[2] = pts_padded[np.argmin(s)], pts_padded[np.argmax(s)]
    diff = np.diff(pts_padded, axis=1)
    rect[1], rect[3] = pts_padded[np.argmin(diff)], pts_padded[np.argmax(diff)]

    dst = np.array([[0,0], [299,0], [299,299], [0,299]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    # Perform warp on the PADDED image
    warped = cv2.warpPerspective(img_padded, M, (300, 300))
    debug_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    std_colors = get_calibrated_colors()
    detected = ['White'] * 9
    cell_sz = 100
    for r in range(3):
        for c in range(3):
            cx, cy = int((c+0.5)*cell_sz), int((r+0.5)*cell_sz)
            roi = warped[cy-12:cy+12, cx-12:cx+12]
            if roi.size == 0: continue
            
            bgr_avg = np.median(roi, axis=(0,1)).astype(np.uint8)
            pixel_lab = cv2.cvtColor(np.uint8([[[bgr_avg[0], bgr_avg[1], bgr_avg[2]]]]), cv2.COLOR_BGR2LAB)[0][0]
            
            min_dist, best_c = float('inf'), 'White'
            for c_name, (hs, ss, vs) in std_colors.items():
                std_lab = cv2.cvtColor(cv2.cvtColor(np.uint8([[[hs, ss, vs]]]), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2LAB)[0][0]
                # Lab distance (weighted for perceptibility)
                d = np.sqrt(0.5*(int(pixel_lab[0])-int(std_lab[0]))**2 + 1.25*(int(pixel_lab[1])-int(std_lab[1]))**2 + (int(pixel_lab[2])-int(std_lab[2]))**2)
                if d < min_dist: min_dist, best_c = d, c_name
            
            detected[r*3+c] = best_c
            cv2.circle(debug_warped, (cx, cy), 15, (255, 255, 255), 2)
            # Smaller, thinner text to avoid 'full view overlap'
            cv2.putText(debug_warped, best_c, (cx-25, cy+40), 0, 0.4, (0,0,0), 2)
            cv2.putText(debug_warped, best_c, (cx-25, cy+40), 0, 0.4, (255,255,255), 1)

    detected[4] = expected_center
    return detected, debug_warped, True

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
                if st.session_state.auto_detect:
                    d, db, success = auto_detect_cube_face(act, CENTER_COLORS[curr])
                    msg_success = "✨ **I flattened the cube face!** Check the orientation and colors."
                    msg_fail = "❌ Cube outline not found. Try to hold it flatter, or switch to Manual Mode (uncheck Auto-Detect) to use the fixed grid."
                else:
                    d, db, success = run_manual_grid_extract(act, CENTER_COLORS[curr], st.session_state.cube_size)
                    msg_success = "📐 **Manual Alignment Used.** I captured the colors inside the center grid."
                    msg_fail = "❌ Failed to read manual grid. Check your connection."

                if success:
                    st.session_state.cube_state[curr] = d
                    st.image(db, caption="📸 Capture Preview", use_container_width=True)
                    st.success(msg_success)
                    
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
