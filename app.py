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

# ── COMPUTER VISION FUNCTIONS ────────────────────────────────────────────────
def auto_detect_cube_region(img):
    h_img, w_img = img.shape[:2]
    # Restrict to realistic sizes
    min_area = (min(h_img, w_img) * 0.25) ** 2
    max_area = (min(h_img, w_img) * 0.95) ** 2

    def score_contours(cnts):
        best = None
        best_score = 0
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            x, y, w, bh = cv2.boundingRect(cnt)

            # A cube face must be roughly a perfect square.
            aspect = min(w, bh) / max(w, bh)
            if aspect < 0.70: # Slightly relaxed for some perspectives
                continue

            # Prefer things closer to the center, but allow searching anywhere
            cx = x + w / 2.0
            cy = y + bh / 2.0
            dist_to_center = ((cx - w_img / 2.0) ** 2 + (cy - h_img / 2.0) ** 2) ** 0.5
            max_dist = ((w_img / 2.0) ** 2 + (h_img / 2.0) ** 2) ** 0.5
            center_weight = max(0.2, 1.0 - (dist_to_center / max_dist))
            
            # Shape bonus: 4 sides is better
            shape_bonus = 3.0 if len(approx) == 4 else (1.0 if len(approx) in [5, 6] else 0.5)

            score = area * aspect * shape_bonus * center_weight

            if score > best_score:
                best_score = score
                side = int(min(w, bh) * 0.94)
                bx = int(cx - side / 2.0)
                by = int(cy - side / 2.0)
                bx = max(0, min(bx, w_img - side))
                by = max(0, min(by, h_img - side))
                best = (bx, by, side)
        return best

    # Pass 1: Canny (Grid lines)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    cnts_edge, _ = cv2.findContours(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = score_contours(cnts_edge)
    if best: return best

    # Pass 2: Saturation (for colorful backgrounds)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 45, 45), (180, 255, 255))
    cnts, _ = cv2.findContours(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return score_contours(cnts)

def classify_color(bgr_pixel, std_colors):
    pixel_lab = cv2.cvtColor(np.uint8([[[bgr_pixel[0], bgr_pixel[1], bgr_pixel[2]]]]), cv2.COLOR_BGR2LAB)[0][0]
    l, a, b = int(pixel_lab[0]), int(pixel_lab[1]), int(pixel_lab[2])
    min_dist, best_c = float('inf'), 'White'
    for name, (hs, ss, vs) in std_colors.items():
        sl = cv2.cvtColor(cv2.cvtColor(np.uint8([[[hs, ss, vs]]]), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2LAB)[0][0]
        dist = ((a-int(sl[1]))*2)**2 + ((b-int(sl[2]))*2)**2 + ((l-int(sl[0]))*(0.5 if name=='White' else 0.15))**2
        if dist < min_dist: min_dist, best_c = dist, name
    if best_c == 'Red' and bgr_pixel[2]>0 and (bgr_pixel[1]/bgr_pixel[2])>0.3: return 'Orange'
    return best_c

def get_calibrated_colors():
    defaults = {'White':(0,30,200), 'Yellow':(30,140,200), 'Orange':(13,170,200), 'Red':(0,180,150), 'Green':(65,140,150), 'Blue':(110,150,150)}
    for k, v in st.session_state.custom_std_colors.items(): defaults[k] = tuple(v)
    return defaults

def extract_colors_from_image(image_bytes, expected_center):
    img = cv2.imdecode(np.asarray(bytearray(image_bytes.read()), dtype=np.uint8), 1)
    if img is None: return None, None, "error"
    h_i, w_i = img.shape[:2]
    std = get_calibrated_colors()
    region = auto_detect_cube_region(img) if st.session_state.auto_detect else None
    if region:
        sx, sy, gs, meth = region[0], region[1], region[2], "auto"
    else:
        if st.session_state.auto_detect: return None, None, "not_found"
        gs = int(min(h_i, w_i) * (st.session_state.cube_size/100.0))
        sx, sy, meth = (w_i-gs)//2, (h_i-gs)//2, "manual"
    
    cs = gs // 3
    # Local calc
    c_roi = img[int(sy+1.5*cs-5):int(sy+1.5*cs+5), int(sx+1.5*cs-5):int(sx+1.5*cs+5)]
    if c_roi.size > 0:
        b, g, r = np.median(c_roi, axis=(0,1)).astype(np.uint8)
        if classify_color((b,g,r), std) == expected_center:
            hsv = cv2.cvtColor(np.uint8([[[b,g,r]]]), cv2.COLOR_BGR2HSV)[0][0]
            std[expected_center] = (int(hsv[0]), int(hsv[1]), int(hsv[2]))

    detected, debug = [], img.copy()
    col_rect = (0, 255, 0) if meth=="auto" else (255, 255, 255)
    cv2.rectangle(debug, (sx, sy), (sx+gs, sy+gs), col_rect, 3)
    for r in range(3):
        for c in range(3):
            cx, cy = int(sx+(c+0.5)*cs), int(sy+(r+0.5)*cs)
            roi = img[max(0,cy-5):min(h_i,cy+5), max(0,cx-5):min(w_i,cx+5)]
            clr = classify_color(np.median(roi, axis=(0,1)), std) if roi.size>0 else 'White'
            detected.append(clr)
            cv2.circle(debug, (cx, cy), 6, (0,255,0), -1)
            cv2.putText(debug, clr, (cx-15, cy+20), 0, 0.4, (255,255,255), 1)
    return detected, cv2.cvtColor(debug, cv2.COLOR_BGR2RGB), meth

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
            
            # If we don't have a confirmed scan yet, process this one
            if st.session_state.processed_photos.get(curr) != key:
                if hasattr(act, 'seek'): act.seek(0)
                d, db, m = extract_colors_from_image(act, CENTER_COLORS[curr])
                
                if d:
                    # Show preview but DON'T jump yet
                    st.image(db, caption=f"AI Detection ({m.upper()}): Please verify colors on the right.", use_container_width=True)
                    
                    st.success("✅ **Scan processed!** Verify the grid on the right, then confirm below.")
                    
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        if st.button("👍 Confirm & Save", type="primary", use_container_width=True, key=f"conf_{key}"):
                            d[4] = CENTER_COLORS[curr]
                            st.session_state.cube_state[curr] = d
                            st.session_state.processed_photos[curr] = key
                            st.session_state[f"cached_img_{curr}"] = db
                            
                            if st.session_state.auto_advance:
                                un = [f for f in FACES if f not in st.session_state.processed_photos]
                                if un: st.session_state.programmatic_face = un[0]
                            st.rerun()
                    with col_b2:
                        if st.button("🔄 Retake / Ignore", use_container_width=True, key=f"ign_{key}"):
                            st.session_state.uploader_key_version += 1
                            st.rerun()
                else:
                    st.error("❌ Detection failed. Ensure the cube is clearly visible and within the center area.")

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
    cal_buf = st.camera_input("Sample", key=f"cal_{v}") or st.file_uploader("Upload", type=['jpg','png'], key=f"ulc_{v}")
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
