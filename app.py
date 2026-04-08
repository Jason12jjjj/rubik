# ==============================================================================
# RUBIK'S CUBE SOLVER (COMPLETE & STABLE VERSION)
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

if 'last_solution' not in st.session_state:
    st.session_state.last_solution = None

if 'uploader_key_version' not in st.session_state:
    st.session_state.uploader_key_version = 0

# --- CSS: Overlay grid on camera ---
c_size = st.session_state.get('cube_size', 50)
st.markdown("""
<style>
    [data-testid="stCameraInput"] { position: relative; }
    [data-testid="stCameraInput"]::after { display: none !important; }
</style>
""", unsafe_allow_html=True)

if not st.session_state.get('auto_detect', True):
    st.markdown(f"""
<style>
    [data-testid="stCameraInput"]::after {{
        content: "";
        display: block !important;
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        width: {c_size}%;
        height: auto;
        aspect-ratio: 1 / 1;
        z-index: 999;
        border: 4px solid rgba(0, 255, 0, 0.9);
        pointer-events: none;
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.3);
        background-image:
            linear-gradient(to right, transparent 33.33%, rgba(0,255,0,0.7) 33.33%, rgba(0,255,0,0.7) calc(33.33% + 2px), transparent calc(33.33% + 2px)),
            linear-gradient(to right, transparent 66.66%, rgba(0,255,0,0.7) 66.66%, rgba(0,255,0,0.7) calc(66.66% + 2px), transparent calc(66.66% + 2px)),
            linear-gradient(to bottom, transparent 33.33%, rgba(0,255,0,0.7) 33.33%, rgba(0,255,0,0.7) calc(33.33% + 2px), transparent calc(33.33% + 2px)),
            linear-gradient(to bottom, transparent 66.66%, rgba(0,255,0,0.7) 66.66%, rgba(0,255,0,0.7) calc(66.66% + 2px), transparent calc(66.66% + 2px));
    }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTER VISION 
# ─────────────────────────────────────────────────────────────────────────────
def auto_detect_cube_region(img):
    h_img, w_img = img.shape[:2]
    min_area = (min(h_img, w_img) * 0.20) ** 2
    max_area = (min(h_img, w_img) * 0.95) ** 2

    k_size = max(5, int(min(w_img, h_img) * 0.015))
    k_size = k_size if k_size % 2 != 0 else k_size + 1

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

            aspect = min(w, bh) / max(w, bh)
            if aspect < 0.70:
                continue

            cx = x + w / 2.0
            cy = y + bh / 2.0
            dist_to_center = ((cx - w_img / 2.0) ** 2 + (cy - h_img / 2.0) ** 2) ** 0.5
            max_dist = ((w_img / 2.0) ** 2 + (h_img / 2.0) ** 2) ** 0.5
            center_weight = max(0.1, 1.0 - (dist_to_center / max_dist))
            
            if dist_to_center > min(w_img, h_img) * 0.45:
                continue

            shape_bonus = 3.0 if len(approx) == 4 else (1.5 if len(approx) in [5, 6] else 0.5)
            side = int(min(w, bh) * 0.92) 
            if side <= 0:
                continue

            score = area * aspect * shape_bonus * (center_weight ** 2)

            if score > best_score:
                best_score = score
                bx = int(cx - side / 2.0)
                by = int(cy - side / 2.0)
                bx = max(0, min(bx, w_img - side))
                by = max(0, min(by, h_img - side))
                best = (bx, by, side)
        return best

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edge, iterations=2)
    cnts_edge, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_region = score_contours(cnts_edge)
    if best_region is not None:
        return best_region

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_sat = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
    clean = cv2.morphologyEx(mask_sat, cv2.MORPH_CLOSE, kernel_edge, iterations=3)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel_edge, iterations=2)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return score_contours(cnts)


def get_calibrated_colors():
    std_colors = {
        'White':  (0, 30, 200),
        'Yellow': (30, 140, 200),
        'Orange': (13, 170, 200),
        'Red':    (0,  180, 150),
        'Green':  (65, 140, 150),
        'Blue':   (110, 150, 150)
    }
    for c_name, custom_hsv in st.session_state.custom_std_colors.items():
        std_colors[c_name] = tuple(custom_hsv) 
    return std_colors


def classify_color(bgr_pixel, std_colors):
    pixel_bgr = np.uint8([[[bgr_pixel[0], bgr_pixel[1], bgr_pixel[2]]]])
    pixel_lab = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2LAB)[0][0]
    l, a, b = int(pixel_lab[0]), int(pixel_lab[1]), int(pixel_lab[2])

    min_dist   = float('inf')
    best_color = 'White'
    
    for c_name, (h_std, s_std, v_std) in std_colors.items():
        std_hsv = np.uint8([[[h_std, s_std, v_std]]])
        std_bgr = cv2.cvtColor(std_hsv, cv2.COLOR_HSV2BGR)
        std_lab = cv2.cvtColor(std_bgr, cv2.COLOR_BGR2LAB)[0][0]
        l_std, a_std, b_std = int(std_lab[0]), int(std_lab[1]), int(std_lab[2])
        
        weight_L = 0.5 if c_name == 'White' else 0.15
        dist = ((a - a_std) * 2.0) ** 2 + ((b - b_std) * 2.0) ** 2 + ((l - l_std) * weight_L) ** 2

        if dist < min_dist:
            min_dist   = dist
            best_color = c_name
            
    if best_color == 'Red' and int(bgr_pixel[2]) > 0:
        g_r_ratio = float(bgr_pixel[1]) / float(bgr_pixel[2])
        if g_r_ratio > 0.30:
            best_color = 'Orange'
            
    return best_color


def extract_colors_from_image(image_bytes, expected_center):
    raw = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(raw, 1)

    if img is None:
        st.error("❌ Could not decode the image. Please upload a valid JPG or PNG file.")
        st.stop()

    h_img, w_img = img.shape[:2]
    std_colors   = get_calibrated_colors()
    debug_img    = img.copy()

    region = None
    if st.session_state.get('auto_detect', True):
        region = auto_detect_cube_region(img)

    if region is None:
        if st.session_state.get('auto_detect', True):
            return None, None, "not_detected"
        else:
            grid_scale = st.session_state.get('cube_size', 50)
            grid_size  = int(min(h_img, w_img) * (grid_scale / 100.0))
            start_x    = (w_img - grid_size) // 2
            start_y    = (h_img - grid_size) // 2
            detection_method = "manual"
    else:
        start_x, start_y, grid_size = region
        detection_method = "auto"

    cell_size = max(1, grid_size // 3)
    color_box = (0, 255, 0) if detection_method == "auto" else (255, 255, 255)

    cv2.rectangle(debug_img, (start_x, start_y), (start_x + grid_size, start_y + grid_size), color_box, 3)
    for i in range(1, 3):
        cv2.line(debug_img, (start_x + i * cell_size, start_y), (start_x + i * cell_size, start_y + grid_size), color_box, 2)
        cv2.line(debug_img, (start_x, start_y + i * cell_size), (start_x + grid_size, start_y + i * cell_size), color_box, 2)

    label = "AUTO DETECTED" if detection_method == "auto" else "MANUAL GRID"
    cv2.putText(debug_img, label, (start_x, max(start_y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_box, 2)

    detected_colors = ['White'] * 9

    cx_ctr = start_x + 1 * cell_size + cell_size // 2
    cy_ctr = start_y + 1 * cell_size + cell_size // 2
    roi_ctr = img[max(0, cy_ctr - 5):min(h_img, cy_ctr + 5), max(0, cx_ctr - 5):min(w_img, cx_ctr + 5)]
    
    if roi_ctr.size > 0:
        c_b, c_g, c_r = np.median(roi_ctr, axis=(0, 1)).astype(np.uint8)
        if classify_color((c_b, c_g, c_r), std_colors) == expected_center:
            hsv = cv2.cvtColor(np.uint8([[[c_b, c_g, c_r]]]), cv2.COLOR_BGR2HSV)[0][0]
            std_colors[expected_center] = (int(hsv[0]), int(hsv[1]), int(hsv[2]))
            
    for row in range(3):
        for col in range(3):
            cx = start_x + col * cell_size + cell_size // 2
            cy = start_y + row * cell_size + cell_size // 2
            y1, y2 = max(0, cy - 5), min(h_img, cy + 5)
            x1, x2 = max(0, cx - 5), min(w_img, cx + 5)
            roi = img[y1:y2, x1:x2]

            idx = row * 3 + col
            if roi.size > 0:
                avg_b, avg_g, avg_r = np.median(roi, axis=(0, 1)).astype(np.uint8)
                detected_colors[idx] = classify_color((avg_b, avg_g, avg_r), std_colors)

            cv2.circle(debug_img, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(debug_img, detected_colors[idx], (cx - 20, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(debug_img, detected_colors[idx], (cx - 20, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return detected_colors, cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), detection_method


# ─────────────────────────────────────────────────────────────────────────────
# 3D TwistyPlayer & Map
# ─────────────────────────────────────────────────────────────────────────────
def render_3d_solution(solution_str, speed):
    def invert_solution(sol):
        inv = []
        for m in reversed(sol.split()):
            if "'" in m: inv.append(m.replace("'", ""))
            elif "2" in m: inv.append(m)
            else: inv.append(m + "'")
        return " ".join(inv)

    setup_alg = invert_solution(solution_str)

    html = f"""
    <script src="https://cdn.cubing.net/v0/js/cubing/twisty" type="module"></script>
    <style>twisty-player {{ width: 100%; height: 360px; }}</style>
    <twisty-player
      experimental-setup-alg="{setup_alg}"
      alg="{solution_str}"
      visualization="PG3D"
      control-panel="bottom-row"
      tempo-scale="{speed}"
      background="none"
      hint-facelets="none">
    </twisty-player>
    """
    components.html(html, height=420)

def render_live_map(active_face=None):
    grid_positions = {'Up':(1,2),'Left':(2,1),'Front':(2,2),'Right':(2,3),'Back':(2,4),'Down':(3,2)}
    html = '<div style="display:grid;grid-template-columns:repeat(4,68px);gap:6px;justify-content:center;text-align:center;font-family:sans-serif;margin-top:8px;">'
    for row in range(1, 4):
        for col in range(1, 5):
            found = next((f for f, p in grid_positions.items() if p == (row, col)), None)
            if found:
                colors      = st.session_state.cube_state[found]
                is_active   = (found == active_face)
                is_scanned  = (found in st.session_state.processed_photos)
                center_emoji = COLOR_EMOJIS[CENTER_COLORS[found]]

                border_style = ("3px solid #00e5ff; box-shadow: 0 0 10px #00e5ff;" if is_active else "1px solid rgba(255,255,255,0.2);")
                opacity = "1.0" if (is_scanned or is_active) else "0.35"
                status_dot = "✅" if is_scanned else ""

                html += (f'<div style="opacity:{opacity}; display:flex; flex-direction:column; align-items:center;">'
                         f'<div style="font-size:11px;font-weight:bold;color:{"#00e5ff" if is_active else "#ddd"};'
                         f'margin-bottom:2px;">{center_emoji} {found} {status_dot}</div>'
                         f'<div style="font-size:10px; color:#ffeb3b; padding-bottom:1px; line-height:1;">⬆️ Top</div>'
                         f'<div style="display:grid;grid-template-columns:repeat(3,20px);gap:2px;justify-content:center;'
                         f'border:{border_style};border-radius:4px;padding:3px;background:rgba(0,0,0,0.25);">')
                for color in colors:
                    html += f'<div style="width:20px;height:20px;background-color:{HEX_COLORS[color]};border:1px solid rgba(0,0,0,0.4);border-radius:2px;"></div>'
                html += '</div></div>'
            else:
                html += '<div></div>'
    html += '</div>'
    return html


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    app_mode = st.radio("Choose Mode:", ["📸 Scan & Solve", "⚙️ Tune Colors"])
    st.divider()

    if app_mode == "📸 Scan & Solve":
        active = st.session_state.get('face_selector', FACES[0])
        st.markdown("## 🗺️ Live Cube Map")
        st.caption("Dims = not yet scanned. Glowing = current face.")
        st.markdown(render_live_map(active_face=active), unsafe_allow_html=True)
    else:
        st.markdown("## 🗺️ Cube State")
        scanned = len(st.session_state.processed_photos)
        st.caption(f"{scanned}/6 faces scanned. Switch to Scan & Solve to see the map.")

    if st.button("🗑️ Clear All Shared Photos", use_container_width=True):
        st.session_state.shared_face_images = {}
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧊 AI Rubik's Solver")

# ══════════════════════════════════════════════════════════════════════════════
# MODE A: SCAN & SOLVE
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "📸 Scan & Solve":

    with st.expander("🎓 Beginner's Guide", expanded=False):
        st.markdown("""
        **Why 'Impossible Cube State'?** The algorithm is very strict about piece combinations.
        The most common cause is holding the cube at the wrong angle when taking a photo.

        **Golden Rule:** Follow the 🧭 **HOW TO HOLD** instructions exactly. The colour adjacent
        to the face you're scanning must be visible at the **TOP edge of the camera screen**.
        """)

    if st.session_state.custom_std_colors:
        st.success(f"🧠 Custom lighting active for: {', '.join(st.session_state.custom_std_colors.keys())}")

    if st.session_state.get('scan_success_msg'):
        st.success(st.session_state.scan_success_msg)
        st.session_state.scan_success_msg = None

    def render_orientation_guide(face):
        center_col = CENTER_COLORS[face]
        top_col = {'Up':'Blue', 'Left':'White', 'Front':'White', 'Right':'White', 'Back':'White', 'Down':'Green'}[face]
        return f"""
        <div style="display: flex; flex-direction: column; align-items: center; padding: 10px; border: 2px dashed #444; border-radius: 8px; width: fit-content; margin-bottom: 15px;">
            <span style="font-size: 11px; font-weight: bold; color: #ffeb3b; margin-bottom: 4px; text-transform: uppercase;">⬆️ Top edge of Camera ⬆️</span>
            <div style="width: 60px; height: 20px; background-color: {HEX_COLORS[top_col]}; border: 2px solid #222; border-radius: 4px 4px 0 0; margin-bottom: 2px; box-shadow: inset 0 0 5px rgba(0,0,0,0.3);"></div>
            <div style="width: 60px; height: 60px; background-color: {HEX_COLORS[center_col]}; border: 2px solid #222; border-radius: 0 0 4px 4px; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: inset 0 0 10px rgba(0,0,0,0.3);">
                {COLOR_EMOJIS[center_col]}
            </div>
            <span style="font-size: 12px; font-weight: bold; color: #fff; margin-top: 6px;">Hold like this</span>
        </div>
        """

    # 🌟 FIX #4: Read from the proxy variable BEFORE rendering the radio widget
    if 'force_next_face' in st.session_state:
        st.session_state.face_selector = st.session_state.force_next_face
        del st.session_state.force_next_face

    current_face = st.radio(
        "🧭 **Select which face you are scanning:**", FACES,
        format_func=lambda x: f"{COLOR_EMOJIS[CENTER_COLORS[x]]} {x} Face",
        horizontal=True, key="face_selector"
    )

    if st.session_state.get('last_scanned_face') != current_face:
        st.session_state.last_scanned_face = current_face
        st.session_state.uploader_key_version += 1

    st.info(f"🧭 **HOW TO HOLD:** {ORIENTATION_GUIDE[current_face]}")
    st.markdown(render_orientation_guide(current_face), unsafe_allow_html=True)

    col_camera, col_manual = st.columns([1, 1])

    with col_camera:
        st.write("### 📷 Image Input")

        shared_bytes = st.session_state.shared_face_images.get(current_face)
        if shared_bytes:
            st.info(f"♻️ **Shared photo from Tune Colors is ready for {current_face} face.**")

        input_method = st.radio("Input Method:", ["📹 Live Camera", "📂 Upload Photo"],
                                horizontal=True, label_visibility="collapsed", key="scan_method")

        current_version = st.session_state.uploader_key_version
        
        if input_method == "📹 Live Camera":
            img_buffer = st.camera_input("Take a picture", key=f"global_cam_{current_version}")
        else:
            img_buffer = st.file_uploader("Upload face image", type=['png', 'jpg', 'jpeg'], key=f"global_up_{current_version}")

        active_buffer = img_buffer
        is_shared     = False
        if active_buffer is None and shared_bytes and input_method == "📂 Upload Photo":
            active_buffer = io.BytesIO(shared_bytes)
            is_shared     = True

        if active_buffer is not None:
            cache_key = (f"{current_face}_shared" if is_shared
                         else f"{current_face}_{getattr(active_buffer, 'file_id', id(active_buffer))}")

            if st.session_state.processed_photos.get(current_face) != cache_key:
                if hasattr(active_buffer, 'seek'):
                    active_buffer.seek(0)
                detected, debug_img, method = extract_colors_from_image(active_buffer, CENTER_COLORS[current_face])
                
                if detected is None:
                    st.error("❌ **Could not detect the Rubik's cube.** Please ensure it is centered and clearly visible.")
                    if current_face in st.session_state.processed_photos:
                        del st.session_state.processed_photos[current_face]
                    st.stop()

                if detected[4] != CENTER_COLORS[current_face]:
                    st.error(f"❌ **Wrong Face Scanned!** Your camera detected a **{detected[4]}** center, but this is the **{current_face} ({CENTER_COLORS[current_face]})** slot.")
                    if current_face in st.session_state.processed_photos:
                        del st.session_state.processed_photos[current_face]
                    st.stop()

                detected[4] = CENTER_COLORS[current_face]

                st.session_state.cube_state[current_face] = detected
                for i in range(9):
                    btn_key = f"sel_{current_face}_{i}"
                    if btn_key in st.session_state:
                        del st.session_state[btn_key]

                st.session_state[f"debug_{current_face}"] = debug_img
                st.session_state[f"method_{current_face}"] = method
                st.session_state.processed_photos[current_face] = cache_key
                st.session_state.last_solution = None

                st.session_state.scan_success_msg = f"🎉 **{current_face} Face** scanned successfully."
                unscanned = [f for f in FACES if f not in st.session_state.processed_photos]
                if unscanned:
                    # Use proxy variable to schedule the UI change for the next run
                    st.session_state.force_next_face = unscanned[0]
                    st.session_state.uploader_key_version += 1
                    st.session_state.scan_success_msg += f" Auto-advanced to **{unscanned[0]} Face**."
                else:
                    st.session_state.scan_success_msg += " All 6 faces are ready to solve!"
                    st.session_state.uploader_key_version += 1

                st.rerun()

        if f"debug_{current_face}" in st.session_state:
            method_tag = st.session_state.get(f"method_{current_face}", "manual")
            caption = ("✅ Auto-detected cube region!" if method_tag == "auto" else "Manual grid used.")
            # 🌟 FIX #1: Width limit for debug image
            st.image(st.session_state[f"debug_{current_face}"], caption=caption, width=400)

    with col_manual:
        st.write("### 🖱️ Edit / Verify Colors")
        st.write("Read Left-to-Right, Top-to-Bottom.")
        grid_cols = st.columns(3)
        current_face_colors = st.session_state.cube_state[current_face]

        def update_tile_color(face, idx):
            st.session_state.cube_state[face][idx] = st.session_state[f"sel_{face}_{idx}"]
            st.session_state.last_solution = None

        for i in range(9):
            col_idx = i % 3
            with grid_cols[col_idx]:
                if i == 4:
                    st.button(f"{COLOR_EMOJIS[CENTER_COLORS[current_face]]} Ctr",
                              key=f"lock_{current_face}", disabled=True, use_container_width=True)
                else:
                    cur_c = current_face_colors[i]
                    st.selectbox(
                        "Color",
                        options=AVAILABLE_COLORS,
                        index=AVAILABLE_COLORS.index(cur_c) if cur_c in AVAILABLE_COLORS else 0,
                        format_func=lambda x: f"{COLOR_EMOJIS[x]} {x}",
                        key=f"sel_{current_face}_{i}",
                        label_visibility="collapsed",
                        on_change=update_tile_color,
                        args=(current_face, i)
                    )

        st.divider()
        next_idx = (FACES.index(current_face) + 1) % 6
        if next_idx == 0:
            st.success("🎉 All 6 faces scanned! You can click Validate & Solve below.")

    st.divider()

    if st.button("Validate & Solve", type="primary", use_container_width=True):
        scanned_count = len(st.session_state.processed_photos)
        if scanned_count < 6:
            st.warning(f"⚠️ You have only scanned {scanned_count}/6 faces. Please capture all 6 faces before solving.")
            st.session_state.last_solution = None
        else:
            is_valid, validation_msg = validate_cube_state(st.session_state.cube_state)
            if is_valid:
                with st.spinner("Running Kociemba algorithm..."):
                    result = solve_cube(st.session_state.cube_state)
                st.session_state.last_solution = result
            else:
                st.session_state.last_solution = None
                st.error(f"❌ Validation Failed: {validation_msg}")

    solution = st.session_state.get('last_solution')
    if solution is not None:
        if solution == "!IMPOSSIBLE_STATE!" or solution.startswith("Error") or solution.startswith("System Error"):
            st.error("❌ **Algorithm Error: Impossible Cube State Detected!**")
            st.markdown("""
            The color counts are correct (9 of each), but their arrangement is **physically impossible**.
            **How to fix:** Check the 🗺️ **Live Cube Map** — find any square that doesn't match your physical cube.
            """)
        else:
            st.balloons()
            st.success("🎉 Solution Found!")
            st.markdown(f"**Algorithm moves:** `{solution}`")
            st.divider()
            st.markdown("### 🎬 3D Step-by-Step Playback")
            speed = st.select_slider("⏩ Playback Speed", options=[0.5, 1.0, 2.0, 3.0], value=1.0, format_func=lambda x: f"{x}x")
            render_3d_solution(solution, speed)
            
            st.divider()
            if st.button("Scan Another Cube", type="primary", use_container_width=True):
                st.session_state.processed_photos = {}
                for face in FACES:
                    default_face = ['White'] * 9
                    default_face[4] = CENTER_COLORS[face]
                    st.session_state.cube_state[face] = default_face
                st.session_state.last_solution = None
                st.session_state.force_next_face = FACES[0]
                st.session_state.uploader_key_version += 1
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MODE B: TUNE COLORS
# ══════════════════════════════════════════════════════════════════════════════
elif app_mode == "⚙️ Tune Colors":
    st.markdown("### ⚙️ Color Calibration")
    st.write("Does the AI misread colors due to your lighting? Teach it correct HSV values by sampling from a photo.")

    calib_color = st.radio(
        "Select the color to calibrate:",
        AVAILABLE_COLORS,
        format_func=lambda x: f"{COLOR_EMOJIS[x]} {x}",
        horizontal=True
    )

    if st.session_state.get('last_calib_color') != calib_color:
        st.session_state.last_calib_color = calib_color
        st.session_state.uploader_key_version += 1

    target_face = COLOR_TO_FACE_NAME.get(calib_color)

    st.info(f"**Instructions:** Place the **{calib_color}** block in the center of the frame, or upload a photo and steer the crosshair.")

    calib_method = st.radio("Method:", ["📹 Live Camera", "📂 Upload Photo"],
                             horizontal=True, label_visibility="collapsed", key="calib_method")

    cv_ver = st.session_state.uploader_key_version
    if calib_method == "📹 Live Camera":
        calib_img_buffer = st.camera_input("Take a photo of the target color", key=f"calib_cam_{cv_ver}")
    else:
        calib_img_buffer = st.file_uploader("Upload photo", type=['png', 'jpg', 'jpeg'], key=f"calib_up_{cv_ver}")

    if calib_img_buffer is not None:
        calib_img_buffer.seek(0)
        raw_bytes = calib_img_buffer.read()

        if target_face:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"### 🔀 Quick Share to {target_face} Face")
                use_for_scan = st.checkbox(
                    f"📌 **Yes, also use this photo as the {COLOR_EMOJIS[calib_color]} {target_face} Face scan**",
                    value=(target_face in st.session_state.shared_face_images),
                    key=f"bridge_{calib_color}"
                )

            if use_for_scan:
                if st.session_state.shared_face_images.get(target_face) != raw_bytes:
                    st.session_state.shared_face_images[target_face] = raw_bytes
                    st.session_state.processed_photos.pop(target_face, None)
                st.success(f"✅ Photo shared! Switch to **Scan & Solve → {target_face} Face** and it will be auto-loaded.")
            else:
                st.session_state.shared_face_images.pop(target_face, None)

            st.markdown("---")

        st.write("#### 🎯 Point and Click Calibration")
        st.write("Click directly on the exact color sticker in the image below to sample its physical color.")
        
        img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)

        if img is None:
            st.error("❌ Could not decode the image. Please upload a valid JPG or PNG file.")
            st.stop()

        h_img, w_img = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 🌟 FIX #2: Width limit for the clickable image
        value = streamlit_image_coordinates(
            img_rgb,
            key=f"calib_click_{calib_color}_{cv_ver}",
            width=450
        )
        
        if value is None:
            cx, cy = w_img // 2, h_img // 2
            st.info("👆 Click anywhere on the image above to sample the color at that point.")
        else:
            cx, cy = value["x"], value["y"]

        cx = max(10, min(w_img - 10, cx))
        cy = max(10, min(h_img - 10, cy))

        roi = img[max(0, cy-5):min(h_img, cy+5), max(0, cx-5):min(w_img, cx+5)]
        avg_b, avg_g, avg_r = np.median(roi, axis=(0, 1)).astype(np.uint8)

        pixel_bgr = np.uint8([[[avg_b, avg_g, avg_r]]])
        hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        debug_img = img.copy()
        cv2.circle(debug_img, (cx, cy), 15, (255, 255, 255), 4)
        cv2.circle(debug_img, (cx, cy), 15, (0, 0, 0), 2)
        debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)

        st.write("#### 📊 Captured Sample")
        c1, c2 = st.columns([1, 1])
        with c1:
            # 🌟 FIX #3: Width limit for the debug output image
            st.image(debug_img_rgb, caption="Selected Position", width=250)
        with c2:
            sample_hex = '#{:02x}{:02x}{:02x}'.format(int(avg_r), int(avg_g), int(avg_b))
            st.markdown(f'<div style="width:100px;height:100px;background-color:{sample_hex};border:2px solid white;border-radius:10px;margin-bottom:8px;"></div>', unsafe_allow_html=True)
            st.caption(f"HSV: [{h}, {s}, {v}]")
            st.caption(f"RGB: [{int(avg_r)}, {int(avg_g)}, {int(avg_b)}]")

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("### 💾 Save Calibration Profile")
            st.write("Happy with the target? Save this lighting profile permanently.")
            if st.button(f"✅ Save HSV [{h}, {s}, {v}] as standard for {calib_color}", type="primary", use_container_width=True):
                st.session_state.custom_std_colors[calib_color] = [h, s, v]
                try:
                    with open(CALIB_FILE, 'w') as f:
                        json.dump(st.session_state.custom_std_colors, f)
                    st.success(f"🎉 Success! AI has learned your new {calib_color} profile.")
                except Exception as e:
                    st.error(f"Failed to save profile: {e}")

            elif (calib_color in st.session_state.custom_std_colors
                    and tuple(st.session_state.custom_std_colors[calib_color]) == (h, s, v)):
                st.info(f"📌 The crosshair is currently exactly on your saved standard HSV profile for {calib_color}.")
