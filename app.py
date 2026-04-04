# ==============================================================================
# RUBIK'S CUBE SOLVER
# ==============================================================================
import io
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
    st.session_state.custom_std_colors = {}

if 'cube_size' not in st.session_state:
    st.session_state.cube_size = 50

if 'shared_face_images' not in st.session_state:
    st.session_state.shared_face_images = {}

if 'auto_detect' not in st.session_state:
    st.session_state.auto_detect = True

# --- CSS: Overlay 3x3 grid on camera — only rendered in manual mode ---
c_size = st.session_state.get('cube_size', 50)

# Base: grid overlay is always HIDDEN by default
st.markdown("""
<style>
    [data-testid="stCameraInput"] { position: relative; }
    [data-testid="stCameraInput"]::after { display: none !important; }
</style>
""", unsafe_allow_html=True)

# Only show the green grid when auto-detect is explicitly turned OFF
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
# HELPER: JS front-end clear
# ─────────────────────────────────────────────────────────────────────────────
def trigger_frontend_clear():
    components.html("""
        <script>
            setTimeout(() => {
                window.parent.document.querySelectorAll('button').forEach(b => {
                    const t = b.innerText.toLowerCase();
                    const a = (b.getAttribute('aria-label') || '').toLowerCase();
                    const title = (b.title || '').toLowerCase();
                    if (t.includes('clear photo') || a.includes('remove') || title.includes('remove')) {
                        b.click();
                    }
                });
            }, 120);
        </script>
    """, height=0)

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTER VISION — Auto-Detect Rubik's Face
# ─────────────────────────────────────────────────────────────────────────────
def auto_detect_cube_region(img):
    """
    Try to automatically find the largest square face of a Rubik's cube
    using edge detection + contour analysis.
    Returns (start_x, start_y, grid_size) or None on failure.
    """
    h_img, w_img = img.shape[:2]

    # --- Step 1: Convert to HSV and build a mask for "cube-like" saturated colors ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Reject very dark pixels and very desaturated pixels (background, hands, etc.)
    mask_sat  = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))

    # --- Step 2: Morphological cleanup ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    clean  = cv2.morphologyEx(mask_sat, cv2.MORPH_CLOSE, kernel, iterations=3)
    clean  = cv2.morphologyEx(clean,   cv2.MORPH_OPEN,  kernel, iterations=2)

    # --- Step 3: Find contours in the cleaned mask ---
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best = None
    best_score = 0
    min_area = (min(h_img, w_img) * 0.18) ** 2   # cube must be at least 18% of smaller dim
    max_area = (min(h_img, w_img) * 0.96) ** 2

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Approximate the contour to a polygon
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        x, y, w, bh = cv2.boundingRect(cnt)

        # Score: prefer large, squarish regions
        aspect = min(w, bh) / max(w, bh)
        if aspect < 0.65:   # skip very elongated regions
            continue

        # Prefer exactly 4-corner polygons (a face), but allow blobs too
        shape_bonus = 1.3 if 4 <= len(approx) <= 6 else 1.0
        score = area * aspect * shape_bonus

        if score > best_score:
            best_score = score
            # Build a square bounding box centred on the detected region
            side = int(min(w, bh) * 0.95)          # slight inset to avoid border
            cx   = x + w  // 2
            cy   = y + bh // 2
            sx   = max(0, cx - side // 2)
            sy   = max(0, cy - side // 2)
            # Clamp so we never go out of image bounds
            sx   = min(sx, w_img - side)
            sy   = min(sy, h_img - side)
            best = (sx, sy, side)

    return best


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
        std_colors[c_name] = custom_hsv
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
    """
    Main CV pipeline.  When auto-detect is ON it tries to locate the cube
    automatically; falls back to the manual grid-size setting if it fails.
    Returns (9-color list, annotated RGB image, detection_method_str)
    """
    raw = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(raw, 1)

    h_img, w_img = img.shape[:2]
    std_colors = get_calibrated_colors()
    debug_img  = img.copy()
    detection_method = "manual"

    # ── Try auto-detection ──────────────────────────────────────────────────
    region = None
    if st.session_state.get('auto_detect', True):
        region = auto_detect_cube_region(img)

    # ── Fall back to manual centre-crop ─────────────────────────────────────
    if region is None:
        grid_scale = st.session_state.get('cube_size', 50)
        grid_size  = int(min(h_img, w_img) * (grid_scale / 100.0))
        start_x    = (w_img - grid_size) // 2
        start_y    = (h_img - grid_size) // 2
        detection_method = "manual"
    else:
        start_x, start_y, grid_size = region
        detection_method = "auto"

    cell_size = max(1, grid_size // 3)

    # ── Draw bounding box & grid ─────────────────────────────────────────────
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

    # ── Label ───────────────────────────────────────────────────────────────
    label = "AUTO DETECTED" if detection_method == "auto" else "MANUAL GRID"
    cv2.putText(debug_img, label,
                (start_x, max(start_y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_box, 2)

    # ── Sample each of the 9 cells ───────────────────────────────────────────
    detected_colors = []
    for row in range(3):
        for col in range(3):
            cx = start_x + col * cell_size + cell_size // 2
            cy = start_y + row * cell_size + cell_size // 2

            # Clamp sample region
            y1, y2 = max(0, cy - 5), min(h_img, cy + 5)
            x1, x2 = max(0, cx - 5), min(w_img, cx + 5)
            roi = img[y1:y2, x1:x2]

            if roi.size == 0:
                detected_colors.append('White')
                continue

            avg_b, avg_g, avg_r = np.median(roi, axis=(0, 1)).astype(np.uint8)
            best_color = classify_color((avg_b, avg_g, avg_r), std_colors)
            detected_colors.append(best_color)

            cv2.circle(debug_img, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(debug_img, best_color, (cx - 20, cy + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(debug_img, best_color, (cx - 20, cy + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    detected_colors[4] = expected_center
    return detected_colors, cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), detection_method


# ─────────────────────────────────────────────────────────────────────────────
# 3D TwistyPlayer
# ─────────────────────────────────────────────────────────────────────────────
def render_3d_solution(solution_str, speed):
    html = f"""
    <script src="https://cdn.cubing.net/esm/cubing/twisty" type="module"></script>
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
def render_live_map():
    html = '<div style="display:grid;grid-template-columns:repeat(4,55px);gap:6px;justify-content:center;text-align:center;font-family:sans-serif;">'
    grid_positions = {'Up':(1,2),'Left':(2,1),'Front':(2,2),'Right':(2,3),'Back':(2,4),'Down':(3,2)}
    for row in range(1, 4):
        for col in range(1, 5):
            found = next((f for f, p in grid_positions.items() if p == (row, col)), None)
            if found:
                html += f'<div><div style="font-size:11px;font-weight:bold;color:#888;margin-bottom:3px;">{found}</div>'
                html += '<div style="display:grid;grid-template-columns:repeat(3,16px);gap:2px;justify-content:center;">'
                for color in st.session_state.cube_state[found]:
                    html += f'<div style="width:16px;height:16px;background-color:{HEX_COLORS[color]};border:1px solid rgba(255,255,255,0.2);border-radius:2px;box-shadow:inset 0 0 2px rgba(0,0,0,0.5);"></div>'
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
                  help="Resize the green scanning box to fit your cube.")

    st.divider()
    st.markdown("## 🗺️ Live Cube Map")
    st.markdown(render_live_map(), unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Reset Lighting Calibration", use_container_width=True):
        st.session_state.custom_std_colors = {}
        st.rerun()
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

    if st.session_state.get('last_scanned_face') != current_face:
        st.session_state.last_scanned_face = current_face
        st.session_state.trigger_js_clear = True

    st.info(f"🧭 **HOW TO HOLD:** {ORIENTATION_GUIDE[current_face]}")

    col_camera, col_manual = st.columns([1, 1])

    with col_camera:
        st.write("### 📷 Image Input")

        shared_bytes = st.session_state.shared_face_images.get(current_face)
        if shared_bytes:
            st.info(f"♻️ **Shared photo from Tune Colors is ready for {current_face} face.** Upload a new photo below to override.")

        input_method = st.radio("Input Method:", ["📹 Live Camera", "📂 Upload Photo"],
                                horizontal=True, label_visibility="collapsed", key="scan_method")

        if input_method == "📹 Live Camera":
            img_buffer = st.camera_input("Take a picture", key="global_camera")
        else:
            img_buffer = st.file_uploader("Upload face image", type=['png', 'jpg', 'jpeg'], key="global_upload")

        if st.session_state.get('trigger_js_clear'):
            trigger_frontend_clear()
            st.session_state.trigger_js_clear = False

        # Resolve image: uploaded > shared
        active_buffer = img_buffer
        is_shared     = False
        if active_buffer is None and shared_bytes:
            active_buffer = io.BytesIO(shared_bytes)
            is_shared     = True

        if active_buffer is not None:
            cache_key = (f"{current_face}_shared" if is_shared
                         else f"{current_face}_{getattr(active_buffer, 'file_id', id(active_buffer))}")

            if st.session_state.processed_photos.get('last_processed') != cache_key:
                if hasattr(active_buffer, 'seek'):
                    active_buffer.seek(0)
                detected, debug_img, method = extract_colors_from_image(active_buffer, CENTER_COLORS[current_face])
                st.session_state.cube_state[current_face] = detected
                st.session_state[f"debug_{current_face}"] = debug_img
                st.session_state[f"method_{current_face}"] = method
                st.session_state.processed_photos['last_processed'] = cache_key
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
                        st.rerun()

        st.divider()

        def auto_advance(nf):
            st.session_state.face_selector = nf

        next_idx = (FACES.index(current_face) + 1) % 6
        if next_idx == 0:
            st.success("🎉 All 6 faces scanned! Click Validate & Solve below.")
        else:
            next_f = FACES[next_idx]
            st.button(f"🚀 Looks Good! Proceed to **{next_f} Face** ➡️",
                      on_click=auto_advance, args=(next_f,), use_container_width=True)

    st.divider()

    if st.button("Validate & Solve", type="primary", use_container_width=True):
        is_valid, validation_msg = validate_cube_state(st.session_state.cube_state)
        if is_valid:
            with st.spinner("Running Kociemba algorithm..."):
                solution = solve_cube(st.session_state.cube_state)

            if "IMPOSSIBLE_STATE" in solution or "Error" in solution:
                st.error("❌ **Algorithm Error: Impossible Cube State Detected!**")
                st.markdown("""
                The color counts are correct (9 of each), but their arrangement is **physically impossible** — like
                reassembling a cube with pieces glued in wrong positions.

                **How to fix:**
                1. Check the 🗺️ **Live Cube Map** sidebar — find any square that doesn't match your physical cube.
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
        else:
            st.error(f"❌ Validation Failed: {validation_msg}")


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
        st.session_state.trigger_calib_js_clear = True

    target_face = COLOR_TO_FACE_NAME.get(calib_color)  # e.g. 'White' -> 'Up'

    st.info(f"**Instructions:** Place the **{calib_color}** block in the center of the frame, or upload a photo and steer the crosshair.")

    calib_method = st.radio("Method:", ["📹 Live Camera", "📂 Upload Photo"],
                             horizontal=True, label_visibility="collapsed", key="calib_method")

    if calib_method == "📹 Live Camera":
        calib_img_buffer = st.camera_input("Take a photo of the target color", key="calib_camera")
    else:
        calib_img_buffer = st.file_uploader("Upload photo", type=['png', 'jpg', 'jpeg'], key="calib_upload")

    if st.session_state.get('trigger_calib_js_clear'):
        trigger_frontend_clear()
        st.session_state.trigger_calib_js_clear = False

    if calib_img_buffer is not None:

        # ── Photo Memory Bridge ──────────────────────────────────────────────
        if target_face:
            st.markdown("---")
            use_for_scan = st.checkbox(
                f"📌 **Also use this photo as the {COLOR_EMOJIS[calib_color]} {target_face} Face scan in Scan & Solve**",
                value=(target_face in st.session_state.shared_face_images),
                key=f"bridge_{calib_color}",
                help=(f"The {calib_color} face is the {target_face} face. "
                      f"Tick this to pre-load this photo into Scan & Solve for that face — no double upload needed!")
            )
            calib_img_buffer.seek(0)
            raw_bytes = calib_img_buffer.read()
            calib_img_buffer.seek(0)

            if use_for_scan:
                st.session_state.shared_face_images[target_face] = raw_bytes
                # Invalidate cache so Scan & Solve re-processes it
                st.session_state.processed_photos.pop('last_processed', None)
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

        calib_img_buffer.seek(0)
        raw = np.asarray(bytearray(calib_img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(raw, 1)

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

        if st.button(f"✅ Save as standard for {calib_color}", type="primary"):
            st.session_state.custom_std_colors[calib_color] = (h, s, v)

        if (calib_color in st.session_state.custom_std_colors
                and st.session_state.custom_std_colors[calib_color] == (h, s, v)):
            st.success(f"🎉 AI has learned your {calib_color} profile! HSV [{h}, {s}, {v}]")
