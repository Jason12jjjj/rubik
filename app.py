# ==============================================================================
# RUBIK'S CUBE SOLVER - UX ENHANCED VERSION
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
    st.session_state.cube_state = {}
    for face in FACES:
        default_face = ['White'] * 9
        default_face[4] = CENTER_COLORS[face]
        st.session_state.cube_state[face] = default_face

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

# --- CSS: Global Styles ---
st.markdown("""
<style>
    /* Hide specific Streamlit camera input overlays */
    [data-testid="stCameraInput"] { position: relative; }
    [data-testid="stCameraInput"]::after { display: none !important; }
    
    /* Style for the interactive grid buttons */
    .stButton>button {
        border-radius: 8px;
    }
    
    /* Make the 3x3 grid buttons more distinct */
    div[data-testid="stVerticalBlock"] > div > div > div > button {
        height: 60px !important;
        font-size: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTER VISION
# ─────────────────────────────────────────────────────────────────────────────
def auto_detect_cube_region(img):
    h_img, w_img = img.shape[:2]
    min_area = (min(h_img, w_img) * 0.25) ** 2
    max_area = (min(h_img, w_img) * 0.95) ** 2

    def score_contours(cnts):
        best = None
        best_score = 0
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area: continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            x, y, w, bh = cv2.boundingRect(cnt)
            aspect = min(w, bh) / max(w, bh)
            if aspect < 0.82: continue
            cx, cy = x + w / 2.0, y + bh / 2.0
            dist_to_center = ((cx - w_img / 2.0) ** 2 + (cy - h_img / 2.0) ** 2) ** 0.5
            max_dist = ((w_img / 2.0) ** 2 + (h_img / 2.0) ** 2) ** 0.5
            center_weight = max(0.1, 1.0 - (dist_to_center / max_dist))
            if dist_to_center > min(w_img, h_img) * 0.35: continue
            shape_bonus = 3.0 if len(approx) == 4 else (1.5 if len(approx) in [5, 6] else 0.5)
            side = int(min(w, bh) * 0.92)
            if side <= 0: continue
            score = area * aspect * shape_bonus * (center_weight ** 2)
            if score > best_score:
                best_score = score
                bx, by = int(cx - side / 2.0), int(cy - side / 2.0)
                bx, by = max(0, min(bx, w_img - side)), max(0, min(by, h_img - side))
                best = (bx, by, side)
        return best

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    cnts_edge, _ = cv2.findContours(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = score_contours(cnts_edge)
    if best: return best

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_sat = cv2.inRange(hsv, (0, 45, 45), (180, 255, 255))
    cnts, _ = cv2.findContours(cv2.morphologyEx(mask_sat, cv2.MORPH_CLOSE, np.ones((7,7))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return score_contours(cnts)

def get_calibrated_colors():
    std_colors = {'White': (0,30,200), 'Yellow': (30,140,200), 'Orange': (13,170,200), 'Red': (0,180,150), 'Green': (65,140,150), 'Blue': (110,150,150)}
    for c_name, custom_hsv in st.session_state.custom_std_colors.items():
        std_colors[c_name] = tuple(custom_hsv)
    return std_colors

def classify_color(bgr_pixel, std_colors):
    pixel_lab = cv2.cvtColor(np.uint8([[[bgr_pixel[0], bgr_pixel[1], bgr_pixel[2]]]]), cv2.COLOR_BGR2LAB)[0][0]
    l, a, b = int(pixel_lab[0]), int(pixel_lab[1]), int(pixel_lab[2])
    min_dist, best_color = float('inf'), 'White'
    for c_name, (h_s, s_s, v_s) in std_colors.items():
        std_lab = cv2.cvtColor(cv2.cvtColor(np.uint8([[[h_s, s_s, v_s]]]), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2LAB)[0][0]
        l_s, a_s, b_s = int(std_lab[0]), int(std_lab[1]), int(std_lab[2])
        w_L = 0.5 if c_name == 'White' else 0.15
        dist = ((a - a_s) * 2.0)**2 + ((b - b_s) * 2.0)**2 + ((l - l_s) * w_L)**2
        if dist < min_dist: min_dist, best_color = dist, c_name
    if best_color == 'Red' and bgr_pixel[2] > 0 and (bgr_pixel[1]/bgr_pixel[2]) > 0.30: return 'Orange'
    return best_color

def extract_colors_from_image(image_bytes, expected_center):
    img = cv2.imdecode(np.asarray(bytearray(image_bytes.read()), dtype=np.uint8), 1)
    if img is None: return None, None, "error"
    h_img, w_img = img.shape[:2]
    std_colors = get_calibrated_colors()
    
    region = auto_detect_cube_region(img) if st.session_state.get('auto_detect', True) else None
    if region:
        start_x, start_y, grid_size = region
        method = "auto"
    else:
        if st.session_state.get('auto_detect', True): return None, None, "not_detected"
        grid_size = int(min(h_img, w_img) * (st.session_state.get('cube_size', 50)/100.0))
        start_x, start_y, method = (w_img-grid_size)//2, (h_img-grid_size)//2, "manual"

    cell_size = grid_size // 3
    debug_img = img.copy()
    color_box = (0, 255, 0) if method == "auto" else (255, 255, 255)
    cv2.rectangle(debug_img, (start_x, start_y), (start_x+grid_size, start_y+grid_size), color_box, 3)
    
    detected_colors = []
    # Dynamic local calibration using center tile
    cx_c, cy_c = start_x + 1.5*cell_size, start_y + 1.5*cell_size
    roi_c = img[int(cy_c-5):int(cy_c+5), int(cx_c-5):int(cx_c+5)]
    if roi_c.size > 0:
        cb, cg, cr = np.median(roi_c, axis=(0,1)).astype(np.uint8)
        if classify_color((cb,cg,cr), std_colors) == expected_center:
            hsv = cv2.cvtColor(np.uint8([[[cb,cg,cr]]]), cv2.COLOR_BGR2HSV)[0][0]
            std_colors[expected_center] = (int(hsv[0]), int(hsv[1]), int(hsv[2]))

    for row in range(3):
        for col in range(3):
            cx, cy = int(start_x + (col+0.5)*cell_size), int(start_y + (row+0.5)*cell_size)
            roi = img[max(0, cy-5):min(h_img, cy+5), max(0, cx-5):min(w_img, cx+5)]
            color = classify_color(np.median(roi, axis=(0,1)), std_colors) if roi.size > 0 else 'White'
            detected_colors.append(color)
            cv2.circle(debug_img, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(debug_img, color, (cx-20, cy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
    return detected_colors, cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), method

# ─────────────────────────────────────────────────────────────────────────────
# 3D SOLVER RENDERING
# ─────────────────────────────────────────────────────────────────────────────
def render_3d_solution(solution_str, speed):
    def invert(sol):
        inv = []
        for m in reversed(sol.split()):
            if "'" in m: inv.append(m.replace("'", ""))
            elif "2" in m: inv.append(m)
            else: inv.append(m + "'")
        return " ".join(inv)
    setup = invert(solution_str)
    html = f"""<script src="https://cdn.cubing.net/v0/js/cubing/twisty" type="module"></script>
    <twisty-player experimental-setup-alg="{setup}" alg="{solution_str}" visualization="PG3D" control-panel="bottom-row" tempo-scale="{speed}" background="none" style="width:100%; height:360px;"></twisty-player>"""
    components.html(html, height=420)

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_orientation_guide(face):
    c = CENTER_COLORS[face]
    t = {'Up':'Blue', 'Left':'White', 'Front':'White', 'Right':'White', 'Back':'White', 'Down':'Green'}[face]
    return f"""<div style="display: flex; flex-direction: column; align-items: center; padding: 10px; border: 2px dashed #444; border-radius: 8px; width: fit-content; margin-bottom: 15px;">
        <span style="font-size: 11px; font-weight: bold; color: #ffeb3b; margin-bottom: 4px;">⬆️ TOP EDGE ⬆️</span>
        <div style="width: 60px; height: 15px; background-color: {HEX_COLORS[t]}; border: 1px solid #000;"></div>
        <div style="width: 60px; height: 60px; background-color: {HEX_COLORS[c]}; border: 1px solid #000; display: flex; align-items: center; justify-content: center; font-size: 24px;">{COLOR_EMOJIS[c]}</div>
    </div>"""

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧊 AI Rubik's Solver")
    app_mode = st.radio("Mode:", ["📸 Scan & Solve", "⚙️ Tune Colors"], label_visibility="collapsed")
    st.divider()

    if app_mode == "📸 Scan & Solve":
        st.markdown("### ⚙️ Settings")
        st.session_state.auto_advance = st.checkbox("⏩ Auto-advance on success", value=st.session_state.auto_advance)
        st.divider()

        st.markdown("### 🗺️ Interactive Map")
        st.caption("Click a face to jump to it.")
        grid_pos = {'Up':(0,1), 'Left':(1,0), 'Front':(1,1), 'Right':(1,2), 'Back':(1,3), 'Down':(2,1)}
        active = st.session_state.programmatic_face
        for r in range(3):
            cols = st.columns(4)
            for c in range(4):
                f_key = next((f for f, p in grid_pos.items() if p == (r, c)), None)
                with cols[c]:
                    if f_key:
                        is_sc = (f_key in st.session_state.processed_photos)
                        lbl = f"{COLOR_EMOJIS[CENTER_COLORS[f_key]]}{'✅' if is_sc else ''}"
                        if st.button(lbl, key=f"nav_{f_key}", use_container_width=True, type="primary" if f_key == active else "secondary"):
                            st.session_state.programmatic_face = f_key
                            st.rerun()
        st.divider()
        with st.expander("🧪 Debug Tools"):
            if st.button("🚀 Load Demo Images", use_container_width=True):
                for f, c_n in CENTER_COLORS.items():
                    fname = f"{c_n.lower()}1.jpeg"
                    if os.path.exists(fname):
                        with open(fname, "rb") as f_in:
                            st.session_state.shared_face_images[f] = f_in.read()
                            st.session_state.processed_photos.pop(f, None)
                st.session_state.uploader_key_version += 1
                st.rerun()
            if st.button("🗑️ Reset All Scans", use_container_width=True):
                st.session_state.processed_photos = {}
                st.session_state.shared_face_images = {}
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
if app_mode == "📸 Scan & Solve":
    curr = st.session_state.programmatic_face
    
    # Navigation Header
    h_cols = st.columns([1, 2, 1])
    with h_cols[0]:
        if st.button("⬅️ Previous", use_container_width=True):
            st.session_state.programmatic_face = FACES[(FACES.index(curr)-1)%6]; st.rerun()
    with h_cols[1]:
        st.markdown(f"<h2 style='text-align:center; margin-top:-10px;'>{COLOR_EMOJIS[CENTER_COLORS[curr]]} {curr} Face</h2>", unsafe_allow_html=True)
    with h_cols[2]:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.programmatic_face = FACES[(FACES.index(curr)+1)%6]; st.rerun()

    if st.session_state.get('last_face') != curr:
        st.session_state.last_face = curr
        st.session_state.uploader_key_version += 1

    st.info(f"🧭 **HOLD:** {ORIENTATION_GUIDE[curr]}")
    st.markdown(render_orientation_guide(curr), unsafe_allow_html=True)

    c_cam, c_edit = st.columns([1, 1])
    
    with c_cam:
        st.markdown("### 📷 Camera Input")
        shared = st.session_state.shared_face_images.get(curr)
        if shared: st.info("♻️ Demo/Shared photo ready.")
        
        in_method = st.radio("Method:", ["📹 Live", "📂 Upload"], horizontal=True, label_visibility="collapsed")
        auto_on = st.toggle("🤖 Auto-Detect Cube", value=st.session_state.auto_detect)
        st.session_state.auto_detect = auto_on
        
        ver = st.session_state.uploader_key_version
        buf = st.camera_input("Scan", key=f"c_{ver}") if in_method == "📹 Live" else st.file_uploader("Upload", type=['jpg','jpeg','png'], key=f"u_{ver}")
        
        active_buf = buf or (io.BytesIO(shared) if shared and not buf and in_method == "📂 Upload" else None)
        
        if active_buf:
            c_key = f"{curr}_{id(active_buf)}"
            if st.session_state.processed_photos.get(curr) != c_key:
                if hasattr(active_buf, 'seek'): active_buf.seek(0)
                det, dbg, meth = extract_colors_from_image(active_buf, CENTER_COLORS[curr])
                if det:
                    det[4] = CENTER_COLORS[curr] # Safety Force
                    st.session_state.cube_state[curr] = det
                    st.session_state[f"dbg_{curr}"] = dbg
                    st.session_state.processed_photos[curr] = c_key
                    if st.session_state.auto_advance:
                        unsc = [f for f in FACES if f not in st.session_state.processed_photos]
                        if unsc: st.session_state.programmatic_face = unsc[0]
                    st.rerun()
                else:
                    st.error("❌ Cube not found. Ensure it is centered.")

        if f"dbg_{curr}" in st.session_state:
            st.image(st.session_state[f"dbg_{curr}"], caption="Detected Grid")

    with c_edit:
        st.markdown("### 🖱️ Edit Colors")
        st.caption("Click squares to cycle: White → Yellow → Green → Blue → Red → Orange")
        
        def cycle(idx):
            cols_seq = ['White', 'Yellow', 'Green', 'Blue', 'Red', 'Orange']
            cur_c = st.session_state.cube_state[curr][idx]
            st.session_state.cube_state[curr][idx] = cols_seq[(cols_seq.index(cur_c)+1)%6]
            st.session_state.last_solution = None

        for r in range(3):
            ecols = st.columns(3)
            for c in range(3):
                idx = r*3 + c
                with ecols[c]:
                    if idx == 4:
                        st.button(f"{COLOR_EMOJIS[CENTER_COLORS[curr]]}\n(Ctr)", disabled=True, use_container_width=True)
                    else:
                        c_val = st.session_state.cube_state[curr][idx]
                        if st.button(f"{COLOR_EMOJIS[c_val]}\n{c_val}", key=f"e_{curr}_{idx}", use_container_width=True, on_click=cycle, args=(idx,)):
                            pass
        if st.button("🧹 Reset Face", use_container_width=True):
            st.session_state.cube_state[curr] = ['White']*9
            st.session_state.cube_state[curr][4] = CENTER_COLORS[curr]
            st.session_state.processed_photos.pop(curr, None)
            st.rerun()

    st.divider()
    if st.button("🚀 VALIDATE & SOLVE", type="primary", use_container_width=True):
        if len(st.session_state.processed_photos) < 6:
            st.warning("⚠️ Scan all 6 faces first!")
        else:
            ok, msg = validate_cube_state(st.session_state.cube_state)
            if ok:
                st.session_state.last_solution = solve_cube(st.session_state.cube_state)
            else:
                st.error(f"❌ {msg}")

    sol = st.session_state.get('last_solution')
    if sol:
        if sol == "!IMPOSSIBLE_STATE!":
            st.error("❌ Impossible arrangement! Check the Map for flipped edges/corners.")
        else:
            st.success(f"✅ Solution: {sol}")
            render_3d_solution(sol, st.select_slider("Speed", [0.5, 1.0, 2.0], 1.0))

elif app_mode == "⚙️ Tune Colors":
    st.markdown("## ⚙️ Color Calibration")
    c_cal = st.radio("Target:", AVAILABLE_COLORS, horizontal=True, format_func=lambda x: f"{COLOR_EMOJIS[x]} {x}")
    ver = st.session_state.uploader_key_version
    cal_buf = st.camera_input("Sample", key=f"cal_{ver}") or st.file_uploader("Upload", type=['jpg','png'], key=f"ulc_{ver}")
    
    if cal_buf:
        img = cv2.imdecode(np.asarray(bytearray(cal_buf.read()), dtype=np.uint8), 1)
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            val = streamlit_image_coordinates(rgb, key=f"click_{c_cal}_{ver}")
            cx, cy = (val['x'], val['y']) if val else (img.shape[1]//2, img.shape[0]//2)
            roi = img[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
            if roi.size > 0:
                avg = np.median(roi, axis=(0,1)).astype(np.uint8)
                hsv = cv2.cvtColor(np.uint8([[[avg[0],avg[1],avg[2]]]]), cv2.COLOR_BGR2HSV)[0][0]
                
                col1, col2 = st.columns(2)
                with col1:
                    cv2.circle(rgb, (cx, cy), 15, (255,255,255), 3)
                    st.image(rgb, caption="Click to sample")
                with col2:
                    st.markdown(f'<div style="width:100px;height:100px;background-color:rgb({avg[2]},{avg[1]},{avg[0]}); border:2px solid #fff;"></div>', unsafe_allow_html=True)
                    if st.button(f"Save HSV {hsv} for {c_cal}", type="primary"):
                        st.session_state.custom_std_colors[c_cal] = [int(hsv[0]), int(hsv[1]), int(hsv[2])]
                        with open(CALIB_FILE, 'w') as f: json.dump(st.session_state.custom_std_colors, f)
                        st.success("Tuned!")
