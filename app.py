# ==============================================================================
# RUBIK'S CUBE SOLVER
# ==============================================================================
import io
import os
import json
import numpy as np
import cv2
import streamlit as st
import streamlit.components.v1 as components
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
    st.session_state.processed_photos = {}      # key: face name → last cache key

# FIX #5: Load custom colors from JSON file so it persists across server reboots
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
    st.session_state.shared_face_images = {}    # key: face name → raw bytes

if 'auto_detect' not in st.session_state:
    st.session_state.auto_detect = True

if 'last_solution' not in st.session_state:
    st.session_state.last_solution = None

# FIX #2: Streamlit Native Component Key Versioning instead of JS hacks
if 'uploader_key_version' not in st.session_state:
    st.session_state.uploader_key_version = 0

# --- CSS: Overlay 3x3 grid on camera — only rendered in manual mode ---
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


def auto_advance(nf):
    st.session_state.face_selector = nf


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTER VISION — Auto-Detect Rubik's Face
# ─────────────────────────────────────────────────────────────────────────────
def auto_detect_cube_region(img):
    h_img, w_img = img.shape[:2]
    min_area   = (min(h_img, w_img) * 0.18) ** 2
    max_area   = (min(h_img, w_img) * 0.99) ** 2 # Relaxed to 0.99 for tightly cropped photos

    def score_contours(cnts):
        best = None
        best_score = 0
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            x, y, w, bh = cv2.boundingRect(cnt)

            aspect = min(w, bh) / max(w, bh)
            if aspect < 0.65:
                continue

            side = int(min(w, bh) * 0.95)
            if side <= 0:
                continue

            shape_bonus = 1.3 if 4 <= len(approx) <= 6 else 1.0
            score       = area * aspect * shape_bonus

            if score > best_score:
                best_score = score
                cx  = x + w  // 2
                cy  = y + bh // 2
                sx  = max(0, cx - side // 2)
                sy  = max(0, cy - side // 2)
                sx  = min(sx, w_img - side)
                sy  = min(sy, h_img - side)
                best = (sx, sy, side)
        return best

    # ── Pass 1: Saturation Masking (Best for coloured faces) ──
    hsv      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_sat = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    clean  = cv2.morphologyEx(mask_sat, cv2.MORPH_CLOSE, kernel, iterations=3)
    clean  = cv2.morphologyEx(clean,   cv2.MORPH_OPEN,  kernel, iterations=2)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_region = score_contours(cnts)
    if best_region is not None:
        return best_region

    # ── Pass 2: Edge Detection Fallback (Best for White face or low saturation) ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edge, iterations=2)
    cnts_edge, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return score_contours(cnts_edge)


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
        std_colors[c_name] = tuple(custom_hsv) # Re-tuple incase loaded from JSON as list
    return std_colors


def classify_color(bgr_pixel, std_colors):
    pixel_bgr = np.uint8([[[bgr_pixel[0], bgr_pixel[1], bgr_pixel[2]]]])
    hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    min_dist   = float('inf')
    best_color = 'White'
    for c_name, (h_std, s_std, v_std) in std_colors.items():
        if c_name == 'White':
            dist = ((s - s_std) * 1.5) ** 2 + ((v - v_std) * 0.5) ** 2
        else:
            dh   = min(abs(h - h_std), 180 - abs(h - h_std))
            dist = (dh * 3.0) ** 2 + ((s - s_std) * 1.0) ** 2 + ((v - v_std) * 0.2) ** 2
        if dist < min_dist:
            min_dist   = dist
            best_color = c_name
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
            # If Auto-Detect was requested but algorithms couldn't find a crisp outline
            # (e.g. image is perfectly cropped, blurry, or very close up),
            # fallback to assuming the cube heavily dominates the frame (90%).
            grid_size  = int(min(h_img, w_img) * 0.90)
            start_x    = (w_img - grid_size) // 2
            start_y    = (h_img - grid_size) // 2
            detection_method = "auto"
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
    # When auto is active, always use a Green box.
    color_box = (0, 255, 0) if detection_method == "auto" else (255, 255, 255)

    cv2.rectangle(debug_img,
                  (start_x, start_y),
                  (start_x + grid_size, start_y + grid_size),
                  color_box, 3)
    for i in range(1, 3):
        cv2.line(debug_img,
                 (start_x + i * cell_size, start_y),
                 (start_x + i * cell_size, start_y + grid_size),
                 color_box, 2)
        cv2.line(debug_img,
                 (start_x, start_y + i * cell_size),
                 (start_x + grid_size, start_y + i * cell_size),
                 color_box, 2)

    label = "AUTO DETECTED" if detection_method == "auto" else "MANUAL GRID"
    cv2.putText(debug_img, label,
                (start_x, max(start_y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_box, 2)

    detected_colors = ['White'] * 9
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
            cv2.putText(debug_img, detected_colors[idx], (cx - 20, cy + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(debug_img, detected_colors[idx], (cx - 20, cy + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    detected_colors[4] = expected_center
    return detected_colors, cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), detection_method


# ─────────────────────────────────────────────────────────────────────────────
# 3D TwistyPlayer
# ─────────────────────────────────────────────────────────────────────────────
def render_3d_solution(solution_str, speed):
    html = f"""
    <script src="https://cdn.cubing.net/v0/js/cubing/twisty" type="module"></script>
    <style>
      twisty-player {{ width: 100%; height: 360px; }}
    </style>
    <twisty-player
      alg="{solution_str}"
      visualization="PG3D"
      control-panel="bottom-row"
      tempo-scale="{speed}"
      background="none"
      hint-facelets="none">
    </twisty-player>
    <p style="color:#aaa; font-size:13px; margin-top:6px;">
      ▶ Press Play to watch the cube solve itself. Drag to rotate view.
    </p>
    """
    components.html(html, height=420)


# ─────────────────────────────────────────────────────────────────────────────
# Live Mini-Map
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FACE_COLORS = {face: (['White'] * 4 + [CENTER_COLORS[face]] + ['White'] * 4) for face in FACES}

def render_live_map(active_face=None):
    grid_positions = {'Up':(1,2),'Left':(2,1),'Front':(2,2),'Right':(2,3),'Back':(2,4),'Down':(3,2)}
    html = '<div style="display:grid;grid-template-columns:repeat(4,62px);gap:4px;justify-content:center;text-align:center;font-family:sans-serif;">'
    for row in range(1, 4):
        for col in range(1, 5):
            found = next((f for f, p in grid_positions.items() if p == (row, col)), None)
            if found:
                colors      = st.session_state.cube_state[found]
                is_active   = (found == active_face)
                is_scanned  = (found in st.session_state.processed_photos)
                center_emoji = COLOR_EMOJIS[CENTER_COLORS[found]]

                border_style = ("3px solid #00e5ff; box-shadow: 0 0 8px #00e5ff;"
                                if is_active else "1px solid rgba(255,255,255,0.15);")
                opacity = "1.0" if (is_scanned or is_active) else "0.4"

                html += (f'<div style="opacity:{opacity};">'
                         f'<div style="font-size:10px;font-weight:bold;color:{"#00e5ff" if is_active else "#888"};'
                         f'margin-bottom:2px;">{center_emoji} {found}</div>'
                         f'<div style="display:grid;grid-template-columns:repeat(3,18px);gap:2px;justify-content:center;'
                         f'border:{border_style};border-radius:3px;padding:2px;">')
                for color in colors:
                    html += (f'<div style="width:18px;height:18px;background-color:{HEX_COLORS[color]};'
                             f'border:1px solid rgba(0,0,0,0.3);border-radius:2px;"></div>')
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

    st.markdown("## 🤖 Detection Mode")
    auto_detect_on = st.toggle("✨ Auto-Detect Cube", value=st.session_state.auto_detect,
                               help="When ON, AI will automatically locate the Rubik's cube face in your photo—no centering needed!")
    st.session_state.auto_detect = auto_detect_on

    if not auto_detect_on:
        st.markdown("## 📐 Manual Grid Size")
        st.slider("📏 Grid Size", min_value=30, max_value=80, key="cube_size",
                  help="Resize the targeting box to fit your cube in the centre of the camera.")

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

    st.divider()
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

        **Quick Fix:** After scanning, check the 🗺️ **Live Cube Map** on the left.
        If any square looks wrong, click it in the `✏️ Edit / Verify Colors` panel to fix it!
        """)

    if st.session_state.custom_std_colors:
        st.success(f"🧠 Custom lighting active for: {', '.join(st.session_state.custom_std_colors.keys())}")

    current_face = st.radio(
        "🧭 **Select which face you are scanning:**", FACES,
        format_func=lambda x: f"{COLOR_EMOJIS[CENTER_COLORS[x]]} {x} Face",
        horizontal=True, key="face_selector"
    )

    # FIX #2: Increment keys instead of invoking Javascript 
    if st.session_state.get('last_scanned_face') != current_face:
        st.session_state.last_scanned_face = current_face
        st.session_state.uploader_key_version += 1

    st.info(f"🧭 **HOW TO HOLD:** {ORIENTATION_GUIDE[current_face]}")

    col_camera, col_manual = st.columns([1, 1])

    with col_camera:
        st.write("### 📷 Image Input")

        shared_bytes = st.session_state.shared_face_images.get(current_face)
        if shared_bytes:
            st.info(f"♻️ **Shared photo from Tune Colors is ready for {current_face} face.** Upload a new photo below to override.")

        input_method = st.radio("Input Method:", ["📹 Live Camera", "📂 Upload Photo"],
                                horizontal=True, label_visibility="collapsed", key="scan_method")

        # Dynamically named keys ensure the widget dies and rebuilds clean when changing faces
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
                st.session_state.cube_state[current_face] = detected
                st.session_state[f"debug_{current_face}"] = debug_img
                st.session_state[f"method_{current_face}"] = method
                st.session_state.processed_photos[current_face] = cache_key
                st.session_state.last_solution = None
                st.rerun()

        if f"debug_{current_face}" in st.session_state:
            method_tag = st.session_state.get(f"method_{current_face}", "manual")
            caption = ("✅ Auto-detected cube region! Fix any errors on the right ➡️"
                       if method_tag == "auto"
                       else "Manual grid used. Fix errors on the right ➡️")
            st.image(st.session_state[f"debug_{current_face}"], caption=caption)

    with col_manual:
        st.write("### 🖱️ Edit / Verify Colors")
        st.write("Read Left-to-Right, Top-to-Bottom.")
        grid_cols = st.columns(3)
        current_face_colors = st.session_state.cube_state[current_face]
        for i in range(9):
            col_idx = i % 3
            with grid_cols[col_idx]:
                if i == 4:
                    st.button(f"{COLOR_EMOJIS[CENTER_COLORS[current_face]]} Ctr",
                              key=f"lock_{current_face}", disabled=True, use_container_width=True)
                else:
                    cur_c = current_face_colors[i]
                    if st.button(f"{COLOR_EMOJIS[cur_c]} {cur_c}",
                                 key=f"btn_{current_face}_{i}", use_container_width=True):
                        ni = (AVAILABLE_COLORS.index(cur_c) + 1) % len(AVAILABLE_COLORS)
                        current_face_colors[i] = AVAILABLE_COLORS[ni]
                        st.session_state.cube_state[current_face] = current_face_colors
                        st.session_state.last_solution = None
                        st.rerun()

        st.divider()

        next_idx = (FACES.index(current_face) + 1) % 6
        if next_idx == 0:
            st.success("🎉 All 6 faces scanned! Click Validate & Solve below.")
        else:
            next_f = FACES[next_idx]
            st.button(f"🚀 Looks Good! Proceed to **{next_f} Face** ➡️",
                      on_click=auto_advance, args=(next_f,), use_container_width=True)

    st.divider()

    if st.button("Validate & Solve", type="primary", use_container_width=True):
        # FIX #3: Warn gracefully instead of cryptic errors if they haven't finished scanning
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

            **How to fix:**
            1. Check the 🗺️ **Live Cube Map** — find any square that doesn't match your physical cube.
            2. Use `✏️ Edit / Verify Colors` to click the wrong squares and correct them.
            3. Most common cause: cube held sideways when photographed.
            """)
        else:
            st.balloons()
            st.success("🎉 Solution Found!")
            st.markdown(f"**Algorithm moves:** `{solution}`")
            st.divider()
            st.markdown("### 🎬 3D Step-by-Step Playback")
            st.write("Watch the cube solve itself. Drag the 3D view to rotate. Use the timeline to step move-by-move.")
            speed = st.select_slider(
                "⏩ Playback Speed",
                options=[0.5, 1.0, 2.0, 3.0],
                value=1.0,
                format_func=lambda x: f"{x}x"
            )
            render_3d_solution(solution, speed)


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

    cv = st.session_state.uploader_key_version
    if calib_method == "📹 Live Camera":
        calib_img_buffer = st.camera_input("Take a photo of the target color", key=f"calib_cam_{cv}")
    else:
        calib_img_buffer = st.file_uploader("Upload photo", type=['png', 'jpg', 'jpeg'], key=f"calib_up_{cv}")


    if calib_img_buffer is not None:
        calib_img_buffer.seek(0)
        raw_bytes = calib_img_buffer.read()

        # ── Photo Memory Bridge ──────────────────────────────────────────────
        if target_face:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"### 🔀 Quick Share to {target_face} Face")
                st.write(f"Save time! Send this exact photo directly to the scanner so you don't have to upload it twice.")
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

        # ── Fine-tune crosshair ──────────────────────────────────────────────
        st.write("#### 🎯 Fine-tune Targeting")
        st.write("Steer the crosshair to the exact color you want to sample.")
        off_col1, off_col2 = st.columns(2)
        with off_col1:
            offset_x = st.slider("➡️ Horizontal (%)", -50, 50, 0, key=f"calib_x_{calib_color}")
        with off_col2:
            offset_y = st.slider("⬇️ Vertical (%)", -50, 50, 0, key=f"calib_y_{calib_color}")

        img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)

        if img is None:
            st.error("❌ Could not decode the image. Please upload a valid JPG or PNG file.")
            st.stop()

        h_img, w_img = img.shape[:2]
        cx = int(w_img / 2 + w_img * (offset_x / 100.0))
        cy = int(h_img / 2 + h_img * (offset_y / 100.0))
        cx = max(10, min(w_img - 10, cx))
        cy = max(10, min(h_img - 10, cy))

        roi = img[max(0, cy-5):min(h_img, cy+5), max(0, cx-5):min(w_img, cx+5)]
        avg_b, avg_g, avg_r = np.median(roi, axis=(0, 1)).astype(np.uint8)

        pixel_bgr = np.uint8([[[avg_b, avg_g, avg_r]]])
        hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        debug_img = img.copy()
        cv2.rectangle(debug_img, (cx - 22, cy - 22), (cx + 22, cy + 22), (255, 255, 255), 3)
        cv2.line(debug_img, (cx - 32, cy), (cx + 32, cy), (0, 255, 0), 2)
        cv2.line(debug_img, (cx, cy - 32), (cx, cy + 32), (0, 255, 0), 2)
        debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)

        st.write("#### 📊 Captured Sample")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(debug_img_rgb, caption="Crosshair position")
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
                # FIX #5: Save to session state AND physically to JSON file
                st.session_state.custom_std_colors[calib_color] = [h, s, v]
                try:
                    with open(CALIB_FILE, 'w') as f:
                        json.dump(st.session_state.custom_std_colors, f)
                    st.success(f"🎉 Success! AI has learned your new {calib_color} profile. Parameters saved permanently.")
                except Exception as e:
                    st.error(f"Failed to save profile: {e}")

            elif (calib_color in st.session_state.custom_std_colors
                    and tuple(st.session_state.custom_std_colors[calib_color]) == (h, s, v)):
                st.info(f"📌 The crosshair is currently exactly on your saved standard HSV profile for {calib_color}.")
