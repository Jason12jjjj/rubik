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
# Reverse lookup: which face holds a given center color
COLOR_TO_FACE_NAME = {v: k for k, v in CENTER_COLORS.items()}

ORIENTATION_GUIDE = {
    'Up':    "Point the camera dead-center at the **⬜ White** face. Rotate the cube so the adjacent **🟦 Blue** face is on the **TOP edge ⬆️** of the camera screen.",
    'Left':  "Point the camera dead-center at the **🟧 Orange** face. Rotate the cube so the adjacent **⬜ White** face is on the **TOP edge ⬆️** of the camera screen.",
    'Front': "Point the camera dead-center at the **🟩 Green** face. Rotate the cube so the adjacent **⬜ White** face is on the **TOP edge ⬆️** of the camera screen.",
    'Right': "Point the camera dead-center at the **🟥 Red** face. Rotate the cube so the adjacent **⬜ White** face is on the **TOP edge ⬆️** of the camera screen.",
    'Back':  "Point the camera dead-center at the **🟦 Blue** face. Rotate the cube so the adjacent **⬜ White** face is on the **TOP edge ⬆️** of the camera screen.",
    'Down':  "Point the camera dead-center at the **🟨 Yellow** face. Rotate the cube so the adjacent **🟩 Green** face is on the **TOP edge ⬆️** of the camera screen."
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

# Photo Memory Bridge: stores raw bytes per face from Tune Colors tab
# Key: face name (e.g. 'Up'), Value: bytes of the uploaded image
if 'shared_face_images' not in st.session_state:
    st.session_state.shared_face_images = {}

# --- CSS: Overlay 3x3 grid on camera ---
c_size = st.session_state.get('cube_size', 50)
st.markdown(f"""
<style>
    [data-testid="stCameraInput"] {{
        position: relative;
    }}
    [data-testid="stCameraInput"]::after {{
        content: "";
        display: block;
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
            linear-gradient(to right, transparent 33.33%, rgba(0, 255, 0, 0.7) 33.33%, rgba(0, 255, 0, 0.7) calc(33.33% + 2px), transparent calc(33.33% + 2px)),
            linear-gradient(to right, transparent 66.66%, rgba(0, 255, 0, 0.7) 66.66%, rgba(0, 255, 0, 0.7) calc(66.66% + 2px), transparent calc(66.66% + 2px)),
            linear-gradient(to bottom, transparent 33.33%, rgba(0, 255, 0, 0.7) 33.33%, rgba(0, 255, 0, 0.7) calc(33.33% + 2px), transparent calc(33.33% + 2px)),
            linear-gradient(to bottom, transparent 66.66%, rgba(0, 255, 0, 0.7) 66.66%, rgba(0, 255, 0, 0.7) calc(66.66% + 2px), transparent calc(66.66% + 2px));
    }}
</style>
""", unsafe_allow_html=True)

# --- Helper: JS Clear ---
def trigger_frontend_clear():
    components.html("""
        <script>
            setTimeout(() => {
                const btns = window.parent.document.querySelectorAll('button');
                btns.forEach(b => {
                    const t = b.innerText.toLowerCase();
                    const a = b.getAttribute('aria-label') || "";
                    const title = b.title || "";
                    if (t.includes('clear photo') || a.includes('Remove') || title.includes('Remove')) {
                        b.click();
                    }
                });
            }, 100);
        </script>
    """, height=0)

# --- 2. Computer Vision Logic ---
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

def extract_colors_from_image(image_bytes, expected_center):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    grid_scale = st.session_state.get('cube_size', 50)
    height, width, _ = img.shape
    grid_size = int(min(height, width) * (grid_scale / 100.0))
    cell_size = grid_size // 3

    start_x = (width - grid_size) // 2
    start_y = (height - grid_size) // 2

    debug_img = img.copy()
    detected_colors = []
    std_colors = get_calibrated_colors()

    cv2.rectangle(debug_img, (start_x, start_y), (start_x + grid_size, start_y + grid_size), (255, 255, 255), 3)
    for i in range(1, 3):
        cv2.line(debug_img, (start_x + i * cell_size, start_y), (start_x + i * cell_size, start_y + grid_size), (255, 255, 255), 2)
        cv2.line(debug_img, (start_x, start_y + i * cell_size), (start_x + grid_size, start_y + i * cell_size), (255, 255, 255), 2)

    for row in range(3):
        for col in range(3):
            cx = start_x + (col * cell_size) + (cell_size // 2)
            cy = start_y + (row * cell_size) + (cell_size // 2)

            roi_bgr = img[cy-5:cy+5, cx-5:cx+5]
            avg_b, avg_g, avg_r = np.median(roi_bgr, axis=(0, 1)).astype(np.uint8)

            pixel_bgr = np.uint8([[[avg_b, avg_g, avg_r]]])
            hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

            min_dist = float('inf')
            best_color = 'White'

            for c_name, (h_std, s_std, v_std) in std_colors.items():
                if c_name == 'White':
                    dist = ((s - s_std) * 1.5)**2 + ((v - v_std) * 0.5)**2
                else:
                    dh = min(abs(h - h_std), 180 - abs(h - h_std))
                    dist = (dh * 3.0)**2 + ((s - s_std) * 1.0)**2 + ((v - v_std) * 0.2)**2

                if dist < min_dist:
                    min_dist = dist
                    best_color = c_name

            detected_colors.append(best_color)
            cv2.circle(debug_img, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(debug_img, best_color, (cx-20, cy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(debug_img, best_color, (cx-20, cy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    detected_colors[4] = expected_center
    debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
    return detected_colors, debug_img_rgb

# --- 3. 3D Solution Renderer ---
def render_3d_solution(solution_str, speed):
    """Embed a cubing.js TwistyPlayer to animate the solution step-by-step."""
    # Speed mapping: user speed (0.5, 1, 2, 3) -> ms per move
    speed_ms = int(1000 / speed)
    html = f"""
    <script src="https://cdn.cubing.net/esm/cubing/twisty" type="module"></script>
    <style>
      twisty-player {{
        width: 100%;
        height: 350px;
      }}
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
      ▶ Press Play to watch the cube solve itself step-by-step. Drag to rotate the view.
    </p>
    """
    components.html(html, height=400)

# --- 4. Live Mini-Map ---
def render_live_map():
    html = '<div style="display: grid; grid-template-columns: repeat(4, 55px); gap: 6px; justify-content: center; text-align: center; font-family: sans-serif;">'
    grid_positions = {'Up': (1, 2), 'Left': (2, 1), 'Front': (2, 2), 'Right': (2, 3), 'Back': (2, 4), 'Down': (3, 2)}
    for row in range(1, 4):
        for col in range(1, 5):
            found_face = next((f for f, p in grid_positions.items() if p == (row, col)), None)
            if found_face:
                html += f'<div><div style="font-size: 11px; font-weight: bold; color: #888; margin-bottom: 3px;">{found_face}</div>'
                html += '<div style="display: grid; grid-template-columns: repeat(3, 16px); gap: 2px; justify-content: center;">'
                for color in st.session_state.cube_state[found_face]:
                    hex_c = HEX_COLORS[color]
                    html += f'<div style="width: 16px; height: 16px; background-color: {hex_c}; border: 1px solid rgba(255,255,255,0.2); border-radius: 2px; box-shadow: inset 0 0 2px rgba(0,0,0,0.5);"></div>'
                html += '</div></div>'
            else:
                html += '<div></div>'
    html += '</div>'
    return html

# --- 5. Sidebar ---
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    app_mode = st.radio("Choose Mode:", ["📸 Scan & Solve", "⚙️ Tune Colors"])
    st.divider()

    st.markdown("## 📐 Camera Tool")
    st.slider("📏 Viewport Grid Size", min_value=30, max_value=80, key="cube_size",
              help="Resize the green scanning box to perfectly fit your Rubik's Cube.")

    st.divider()
    st.markdown("## 🗺️ Live Cube Map")
    st.markdown(render_live_map(), unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Reset Lighting Calibration", use_container_width=True):
        st.session_state.custom_std_colors = {}
        st.rerun()

# --- 6. Main UI ---
st.title("🧊 AI Rubik's Solver")

# ══════════════════════════════════════════════════════════
# MODE A: SCAN & SOLVE
# ══════════════════════════════════════════════════════════
if app_mode == "📸 Scan & Solve":
    with st.expander("🎓 Beginner's Guide: Reading the Map & Avoiding Errors", expanded=False):
        st.markdown("""
        **Why does the AI sometimes say 'Impossible Cube State'?**
        A Rubik's cube algorithm is intensely strict. The most common error is holding the cube rotated sideways when taking a photo.

        **The Golden Rule:** Always strictly obey the 🧭 **HOW TO HOLD** instructions!

        **How to read the Live Map:** The **🗺️ Live Cube Map** on the left simulates an unfolded cardboard box of your Rubik's cube. Check it after scanning. If any single dot doesn't match your physical cube, simply click the dot in the `✏️ Manual Editing` section to fix it before pressing Solve!
        """)

    if st.session_state.custom_std_colors:
        calibrated_list = ", ".join(st.session_state.custom_std_colors.keys())
        st.success(f"🧠 **AI is using your custom lighting profiles for:** {calibrated_list}")

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

        # Check if Tune Colors has shared an image for this face
        shared_bytes = st.session_state.shared_face_images.get(current_face)
        if shared_bytes:
            st.success(f"♻️ **Using shared photo from Tune Colors for {current_face} face!**")
            st.write("_You can still upload a new photo below to override it._")

        input_method_options = ["📹 Live Camera", "📂 Upload Photo"]
        input_method = st.radio("Input Method:", input_method_options, horizontal=True,
                                label_visibility="collapsed", key="scan_method")

        if input_method == "📹 Live Camera":
            img_buffer = st.camera_input("Take a picture", key="global_camera")
        else:
            img_buffer = st.file_uploader("Upload face image", type=['png', 'jpg', 'jpeg'], key="global_upload")

        if st.session_state.get('trigger_js_clear'):
            trigger_frontend_clear()
            st.session_state.trigger_js_clear = False

        # Determine which image to process: uploaded takes priority, then shared
        active_buffer = img_buffer
        is_shared = False
        if active_buffer is None and shared_bytes:
            active_buffer = io.BytesIO(shared_bytes)
            is_shared = True

        if active_buffer is not None:
            if is_shared:
                cache_key = f"{current_face}_shared"
            else:
                cache_key = f"{current_face}_{getattr(active_buffer, 'file_id', id(active_buffer))}"

            if st.session_state.processed_photos.get('last_processed') != cache_key:
                if hasattr(active_buffer, 'seek'):
                    active_buffer.seek(0)
                detected, debug_img = extract_colors_from_image(active_buffer, CENTER_COLORS[current_face])
                st.session_state.cube_state[current_face] = detected
                st.session_state[f"debug_{current_face}"] = debug_img
                st.session_state.processed_photos['last_processed'] = cache_key
                st.rerun()

        if f"debug_{current_face}" in st.session_state:
            st.image(st.session_state[f"debug_{current_face}"], caption="AI Thought Process. Fix errors on the right ➡️")

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
                    current_color = current_face_colors[i]
                    if st.button(f"{COLOR_EMOJIS[current_color]} {current_color}",
                                 key=f"btn_{current_face}_{i}", use_container_width=True):
                        next_index = (AVAILABLE_COLORS.index(current_color) + 1) % len(AVAILABLE_COLORS)
                        current_face_colors[i] = AVAILABLE_COLORS[next_index]
                        st.session_state.cube_state[current_face] = current_face_colors
                        st.rerun()

        st.divider()

        def auto_advance(nf):
            st.session_state.face_selector = nf

        next_idx = (FACES.index(current_face) + 1) % 6
        if next_idx == 0:
            st.success("🎉 You have looped through all 6 faces! Try validating!")
        else:
            next_f = FACES[next_idx]
            st.button(f"🚀 Looks Good! Proceed to **{next_f} Face** ➡️",
                      on_click=auto_advance, args=(next_f,), use_container_width=True)

    st.divider()

    if st.button("Validate & Solve", type="primary", use_container_width=True):
        is_valid, validation_msg = validate_cube_state(st.session_state.cube_state)
        if is_valid:
            with st.spinner("Executing Algorithm..."):
                solution = solve_cube(st.session_state.cube_state)

            if "IMPOSSIBLE_STATE" in solution or "Error" in solution:
                st.error("❌ **Algorithm Error: Impossible Cube State Detected!**")
                st.markdown("""
                The AI counted exactly 9 squares of each color, but the way they are logically arranged is **Physically Impossible**.

                **How to fix:**
                1. Look at the 🗺️ **Live Cube Map** on the left sidebar.
                2. Find any dot that looks wrong compared to your real cube.
                3. Use the `✏️ Manual Editing` buttons to **click the bad colors** to swap them!
                """)
            else:
                st.balloons()
                st.success("🎉 Solution Found!")
                st.markdown(f"**Algorithm Moves:** `{solution}`")

                st.divider()
                st.markdown("### 🎬 3D Step-by-Step Solution Playback")
                st.write("Watch the cube solve itself! Use the timeline to step through each move.")

                speed = st.select_slider(
                    "⏩ Playback Speed",
                    options=[0.5, 1.0, 2.0, 3.0],
                    value=1.0,
                    format_func=lambda x: f"{x}x"
                )
                render_3d_solution(solution, speed)
        else:
            st.error(f"❌ Validation Failed: {validation_msg}")

# ══════════════════════════════════════════════════════════
# MODE B: TUNE COLORS
# ══════════════════════════════════════════════════════════
elif app_mode == "⚙️ Tune Colors":
    st.markdown("### ⚙️ Explicit Color Calibration")
    st.write("Does the AI struggle with specific colors due to your room's lighting? Teach it what those colors *actually* look like right now!")

    calib_color = st.radio(
        "Select the color you want to calibrate:",
        AVAILABLE_COLORS,
        format_func=lambda x: f"{COLOR_EMOJIS[x]} {x}",
        horizontal=True
    )

    if st.session_state.get('last_calib_color') != calib_color:
        st.session_state.last_calib_color = calib_color
        st.session_state.trigger_calib_js_clear = True

    # Determine which face this color corresponds to (for the Photo Bridge)
    target_face = COLOR_TO_FACE_NAME.get(calib_color)  # e.g. 'White' -> 'Up'

    st.info(f"**Instructions:** Place a **{calib_color}** block directly in the absolute **DEAD CENTER** of the frame.")

    calib_method = st.radio("Calibration Method:", ["📹 Live Camera", "📂 Upload Photo"],
                             horizontal=True, label_visibility="collapsed", key="calib_method")

    if calib_method == "📹 Live Camera":
        calib_img_buffer = st.camera_input("Take a photo of the target color", key="calib_camera")
    else:
        calib_img_buffer = st.file_uploader("Upload photo of the target color",
                                              type=['png', 'jpg', 'jpeg'], key="calib_upload")

    if st.session_state.get('trigger_calib_js_clear'):
        trigger_frontend_clear()
        st.session_state.trigger_calib_js_clear = False

    if calib_img_buffer is not None:
        # --- Photo Memory Bridge Checkbox ---
        if target_face:
            use_for_scan = st.checkbox(
                f"📌 Also use this photo as the **{target_face} Face** scan in Scan & Solve",
                value=False,
                key=f"bridge_{calib_color}",
                help=f"The {calib_color} center belongs to the {target_face} face. "
                     f"Checking this will pre-load this photo into the Scan & Solve scanner for that face."
            )
            if use_for_scan:
                calib_img_buffer.seek(0)
                raw_bytes = calib_img_buffer.read()
                calib_img_buffer.seek(0)
                if st.session_state.shared_face_images.get(target_face) != raw_bytes:
                    st.session_state.shared_face_images[target_face] = raw_bytes
                    # Clear cache so it re-processes on next visit
                    st.session_state.processed_photos.pop('last_processed', None)
                st.success(f"✅ This photo will be automatically used for the **{target_face} Face** in Scan & Solve!")
            else:
                # If unchecked, remove the bridge if it was previously set
                if target_face in st.session_state.shared_face_images:
                    del st.session_state.shared_face_images[target_face]

        st.write("#### 🎯 Fine-tune Targeting")
        st.write("Move the sliders to steer the targeting crosshair over your desired color.")
        off_col1, off_col2 = st.columns(2)
        with off_col1:
            offset_x = st.slider("➡️ Horizontal Offset (%)", -50, 50, 0, key=f"calib_x_{calib_color}")
        with off_col2:
            offset_y = st.slider("⬇️ Vertical Offset (%)", -50, 50, 0, key=f"calib_y_{calib_color}")

        calib_img_buffer.seek(0)
        file_bytes = np.asarray(bytearray(calib_img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        height, width, _ = img.shape
        cx = int((width / 2) + (width * (offset_x / 100.0)))
        cy = int((height / 2) + (height * (offset_y / 100.0)))
        cx = max(10, min(width - 10, cx))
        cy = max(10, min(height - 10, cy))

        roi_bgr = img[cy-5:cy+5, cx-5:cx+5]
        avg_b, avg_g, avg_r = np.median(roi_bgr, axis=(0, 1)).astype(np.uint8)

        pixel_bgr = np.uint8([[[avg_b, avg_g, avg_r]]])
        hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        debug_img = img.copy()
        cv2.rectangle(debug_img, (cx-20, cy-20), (cx+20, cy+20), (255, 255, 255), 3)
        cv2.circle(debug_img, (cx, cy), 5, (0, 255, 0), -1)
        debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)

        st.write("#### Captured Sample:")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(debug_img_rgb, caption="Targeted Region (crosshair)")
        with col2:
            sample_hex = '#{:02x}{:02x}{:02x}'.format(avg_r, avg_g, avg_b)
            st.markdown(f'<div style="width: 100px; height: 100px; background-color: {sample_hex}; border: 2px solid white; border-radius: 10px;"></div>', unsafe_allow_html=True)
            st.caption(f"Raw HSV: [{h}, {s}, {v}]")

        if st.button(f"✅ Save as new standard for {calib_color}", type="primary"):
            st.session_state.custom_std_colors[calib_color] = (h, s, v)

        if calib_color in st.session_state.custom_std_colors and st.session_state.custom_std_colors[calib_color] == (h, s, v):
            st.success(f"🎉 Success! The AI has successfully learned your new {calib_color} standard.")
