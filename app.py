# ==============================================================================
# RUBIK'S CUBE SOLVER - DEMO SAFE VERSION (Stable Camera + CSS Targeting + Fixed Map)
# ==============================================================================
import numpy as np
import cv2
import streamlit as st
from rubiks_core import validate_cube_state, solve_cube

# --- 1. System Configuration & State Management ---
st.set_page_config(page_title="AI Rubik's Solver", page_icon="🧊", layout="wide")

AVAILABLE_COLORS = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
FACES = ['Up', 'Left', 'Front', 'Right', 'Back', 'Down']
COLOR_EMOJIS = {'White': '⬜', 'Red': '🟥', 'Green': '🟩', 'Yellow': '🟨', 'Orange': '🟧', 'Blue': '🟦'}
HEX_COLORS = {'White': '#f8f9fa', 'Red': '#ff4b4b', 'Green': '#09ab3b', 'Yellow': '#ffeb3b', 'Orange': '#ffa500', 'Blue': '#1e88e5'}
CENTER_COLORS = {'Up': 'White', 'Left': 'Orange', 'Front': 'Green', 'Right': 'Red', 'Back': 'Blue', 'Down': 'Yellow'}

ORIENTATION_GUIDE = {
    'Up':    "Look at the **⬜ White Face**. Ensure the **🟦 Blue Center** is pointing UP.",
    'Left':  "Look at the **🟧 Orange Face**. Ensure the **⬜ White Center** is pointing UP.",
    'Front': "Look at the **🟩 Green Face**. Ensure the **⬜ White Center** is pointing UP.",
    'Right': "Look at the **🟥 Red Face**. Ensure the **⬜ White Center** is pointing UP.",
    'Back':  "Look at the **🟦 Blue Face**. Ensure the **⬜ White Center** is pointing UP.",
    'Down':  "Look at the **🟨 Yellow Face**. Ensure the **🟩 Green Center** is pointing UP."
}

# Initialize states
if 'cube_state' not in st.session_state:
    st.session_state.cube_state = {}
    for face in FACES:
        default_face = ['White'] * 9 
        default_face[4] = CENTER_COLORS[face] 
        st.session_state.cube_state[face] = default_face

# Memory to track which photos have been processed to prevent overwrite bugs
if 'processed_photos' not in st.session_state:
    st.session_state.processed_photos = {}

# Memory for Custom Calibration
if 'custom_std_colors' not in st.session_state:
    st.session_state.custom_std_colors = {}

if 'cube_size' not in st.session_state:
    st.session_state.cube_size = 50

# --- 🌟 CSS HACK: Overlay a Targeting Box on the Camera ---
c_size = st.session_state.get('cube_size', 50)
st.markdown(f"""
<style>
    [data-testid="stCameraInput"] {{
        position: relative;
    }}
    /* Bind the grid directly to the camera container and elevate z-index */
    [data-testid="stCameraInput"]::after {{
        content: "";
        display: block;
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        
        /* Use percentage bounds for perfect alignment */
        width: {c_size}%;
        height: auto;
        aspect-ratio: 1 / 1;
        z-index: 999; /* Ensure it stays above the video feed */
        
        border: 4px solid rgba(0, 255, 0, 0.9);
        pointer-events: none;
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.3); /* Transparent dark surroundings */
        
        /* 3x3 Grid */
        background-image: 
            linear-gradient(to right, transparent 33.33%, rgba(0, 255, 0, 0.7) 33.33%, rgba(0, 255, 0, 0.7) calc(33.33% + 2px), transparent calc(33.33% + 2px)),
            linear-gradient(to right, transparent 66.66%, rgba(0, 255, 0, 0.7) 66.66%, rgba(0, 255, 0, 0.7) calc(66.66% + 2px), transparent calc(66.66% + 2px)),
            linear-gradient(to bottom, transparent 33.33%, rgba(0, 255, 0, 0.7) 33.33%, rgba(0, 255, 0, 0.7) calc(33.33% + 2px), transparent calc(33.33% + 2px)),
            linear-gradient(to bottom, transparent 66.66%, rgba(0, 255, 0, 0.7) 66.66%, rgba(0, 255, 0, 0.7) calc(66.66% + 2px), transparent calc(66.66% + 2px));
    }}
</style>
""", unsafe_allow_html=True)

def get_calibrated_colors():
    # Base profiles for typical muted webcams
    std_colors = {
        'White':  (0, 30, 200),
        'Yellow': (30, 140, 200),
        'Orange': (13, 170, 200),
        'Red':    (0,  180, 150),
        'Green':  (65, 140, 150),
        'Blue':   (110, 150, 150)
    }
    
    # Override with explicitly user-calibrated colors if available
    for c_name, custom_hsv in st.session_state.custom_std_colors.items():
        std_colors[c_name] = custom_hsv
            
    return std_colors

# --- 2. Computer Vision Logic ---
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

    # HSV robust classification replaces LAB distance 

    for row in range(3):
        for col in range(3):
            cx = start_x + (col * cell_size) + (cell_size // 2)
            cy = start_y + (row * cell_size) + (cell_size // 2)
            
            roi_bgr = img[cy-5:cy+5, cx-5:cx+5]
            avg_b, avg_g, avg_r = np.median(roi_bgr, axis=(0, 1)).astype(np.uint8)
            
            # Ultimate Fix: Weighted HSV Nearest-Neighbor
            # Using rigid if-else fails because True Orange and Warm White overlap heavily on cheap webcams.
            # Instead, we measure the 'distance' to realistic webcam profiles.
            pixel_bgr = np.uint8([[[avg_b, avg_g, avg_r]]])
            hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
            
            min_dist = float('inf')
            best_color = 'White'
            
            for c_name, (h_std, s_std, v_std) in std_colors.items():
                if c_name == 'White':
                    # White ignores Hue! Distance is governed strictly by how close Sat is to 0 (30).
                    # We penalize Saturation heavily (1.5) to keep it from swallowing real colors.
                    dist = ((s - s_std) * 1.5)**2 + ((v - v_std) * 0.5)**2
                else:
                    # Hue wraps around 180 in OpenCV
                    dh = min(abs(h - h_std), 180 - abs(h - h_std))
                    # Hue is weighted 3.0x because it's the strongest identifier for matching chromaticity
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

# --- 3. Live Mini-Map ---
def render_live_map():
    html = '<div style="display: grid; grid-template-columns: repeat(4, 50px); gap: 5px; justify-content: center;">'
    grid_positions = {'Up': (1, 2), 'Left': (2, 1), 'Front': (2, 2), 'Right': (2, 3), 'Back': (2, 4), 'Down': (3, 2)}
    for row in range(1, 4):
        for col in range(1, 5):
            found_face = next((f for f, p in grid_positions.items() if p == (row, col)), None)
            if found_face:
                html += '<div style="display: grid; grid-template-columns: repeat(3, 15px); gap: 1px;">'
                for color in st.session_state.cube_state[found_face]:
                    hex_c = HEX_COLORS[color]
                    html += f'<div style="width: 15px; height: 15px; background-color: {hex_c}; border: 1px solid #444; border-radius: 2px;"></div>'
                html += '</div>'
            else: html += '<div></div>'
    html += '</div>'
    return html

with st.sidebar:
    st.markdown("## 🧭 Navigation")
    app_mode = st.radio("Choose Mode:", ["📸 Scan & Solve", "⚙️ Tune Colors"])
    st.divider()
    
    st.markdown("## 📐 Camera Tool")
    st.slider("📏 Viewport Grid Size", min_value=30, max_value=80, key="cube_size", help="Resize the green scanning box to perfectly fit your Rubik's Cube.")
    
    st.divider()
    st.markdown("## 🗺️ Live Cube Map")
    st.markdown(render_live_map(), unsafe_allow_html=True) 
    
    st.divider()
    if st.button("🔄 Reset Lighting Calibration", use_container_width=True):
        st.session_state.custom_std_colors = {}
        st.rerun()
    
# --- 4. Main User Interface ---
st.title("🧊 AI Rubik's Solver")

if app_mode == "📸 Scan & Solve":
    if st.session_state.custom_std_colors:
        calibrated_list = ", ".join(st.session_state.custom_std_colors.keys())
        st.success(f"🧠 **AI is using your custom lighting profiles for:** {calibrated_list}")

    # Use a single global camera to prevent browser freezing with multiple video streams
    current_face = st.radio("🧭 **Select which face you are scanning:**", FACES, format_func=lambda x: f"{COLOR_EMOJIS[CENTER_COLORS[x]]} {x} Face", horizontal=True)

    st.info(f"🧭 **HOW TO HOLD:** {ORIENTATION_GUIDE[current_face]}")

    col_camera, col_manual = st.columns([1, 1])

    with col_camera:
        st.write(f"### 📷 Camera Scanner")
        st.write("Align the cube inside the dark central box.")
        
        img_buffer = st.camera_input("Take a picture", key="global_camera")
        
        # We also need to track WHICH face this photo was applied to
        if img_buffer is not None:
            photo_id = img_buffer.file_id
            cache_key = f"{current_face}_{photo_id}"
            
            if st.session_state.processed_photos.get('last_processed') != cache_key:
                detected, debug_img = extract_colors_from_image(img_buffer, CENTER_COLORS[current_face])
                st.session_state.cube_state[current_face] = detected
                st.session_state[f"debug_{current_face}"] = debug_img
                st.session_state.processed_photos['last_processed'] = cache_key
                st.rerun() # Refresh map

        if f"debug_{current_face}" in st.session_state:
            st.image(st.session_state[f"debug_{current_face}"], caption="AI Thought Process. Fix errors on the right ➡️")

    with col_manual:
        st.write(f"### 🖱️ Edit / Verify Colors")
        st.write("Read Left-to-Right, Top-to-Bottom.")
        grid_cols = st.columns(3)
        current_face_colors = st.session_state.cube_state[current_face]
        for i in range(9):
            col_idx = i % 3
            with grid_cols[col_idx]:
                if i == 4:
                    st.button(f"{COLOR_EMOJIS[CENTER_COLORS[current_face]]} Ctr", key=f"lock_{current_face}", disabled=True, use_container_width=True)
                else:
                    current_color = current_face_colors[i]
                    if st.button(f"{COLOR_EMOJIS[current_color]} {current_color}", key=f"btn_{current_face}_{i}", use_container_width=True):
                        next_index = (AVAILABLE_COLORS.index(current_color) + 1) % len(AVAILABLE_COLORS)
                        current_face_colors[i] = AVAILABLE_COLORS[next_index]
                        st.session_state.cube_state[current_face] = current_face_colors
                        st.rerun()

    st.divider()

    if st.button("Validate & Solve", type="primary", use_container_width=True):
        is_valid, validation_msg = validate_cube_state(st.session_state.cube_state)
        if is_valid:
            with st.spinner("Executing Algorithm..."):
                solution = solve_cube(st.session_state.cube_state)
                if "Error" in solution: st.error(solution)
                else:
                    st.balloons()
                    st.success("🎉 Solution Found!")
                    st.markdown(f"### ➡️ Steps: `{solution}`")
        else:
            st.error(f"❌ Validation Failed: {validation_msg}")

elif app_mode == "⚙️ Tune Colors":
    st.markdown("### ⚙️ Explicit Color Calibration")
    st.write("Does the AI struggle with specific colors due to your room's lighting? Teach it what those colors *actually* look like right now!")
    
    calib_color = st.radio("Select the color you want to calibrate:", AVAILABLE_COLORS, format_func=lambda x: f"{COLOR_EMOJIS[x]} {x}", horizontal=True)
    
    st.info(f"**Instructions:** Hold a **{calib_color}** block directly in the absolute **DEAD CENTER** of the aiming box and take a picture.")
    
    calib_img_buffer = st.camera_input("Take a photo of the target color", key="calib_camera")
    
    if calib_img_buffer is not None:
        file_bytes = np.asarray(bytearray(calib_img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1) 
        
        height, width, _ = img.shape
        cy = height // 2
        cx = width // 2
        
        # Sample the center 10x10 pixels
        roi_bgr = img[cy-5:cy+5, cx-5:cx+5]
        avg_b, avg_g, avg_r = np.median(roi_bgr, axis=(0, 1)).astype(np.uint8)
        
        pixel_bgr = np.uint8([[[avg_b, avg_g, avg_r]]])
        hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        
        # Draw a visual targeting box so the user knows what was sampled
        debug_img = img.copy()
        cv2.rectangle(debug_img, (cx-20, cy-20), (cx+20, cy+20), (255, 255, 255), 3)
        cv2.circle(debug_img, (cx, cy), 5, (0, 255, 0), -1)
        debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        
        # Render what the camera saw in the exact center
        st.write("#### Captured Sample:")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(debug_img_rgb, caption="Targeted Region (Center)")
        with col2:
            sample_hex = '#{:02x}{:02x}{:02x}'.format(avg_r, avg_g, avg_b)
            st.markdown(f'<div style="width: 100px; height: 100px; background-color: {sample_hex}; border: 2px solid white; border-radius: 10px;"></div>', unsafe_allow_html=True)
            st.caption(f"Raw HSV: [{h}, {s}, {v}]")
        
        if st.button(f"✅ Save as new standard for {calib_color}", type="primary"):
            st.session_state.custom_std_colors[calib_color] = (h, s, v)
            
        if calib_color in st.session_state.custom_std_colors and st.session_state.custom_std_colors[calib_color] == (h, s, v):
            st.success(f"🎉 Success! The AI has successfully learned your new {calib_color} standard.")
