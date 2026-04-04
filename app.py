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

# --- 🌟 CSS HACK: Overlay a Targeting Box on the Camera ---
st.markdown("""
<style>
    /* This creates a subtle aiming box over the Streamlit camera */
    [data-testid="stCameraInput"] > div:first-child::after {
        content: "";
        position: absolute;
        top: 20%; bottom: 20%; left: 30%; right: 30%;
        border: 3px dashed rgba(255, 255, 255, 0.7);
        pointer-events: none; /* Let clicks pass through */
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.3); /* Darken surroundings */
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Computer Vision Logic ---
def extract_colors_from_image(image_bytes, expected_center):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1) 
    
    height, width, _ = img.shape
    grid_size = min(height, width) // 2 
    cell_size = grid_size // 3
    
    start_x = (width - grid_size) // 2
    start_y = (height - grid_size) // 2
    
    debug_img = img.copy()
    detected_colors = []
    
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
            
            # These are actual typical webcam HSV centers, NOT ideal RGB #FFFFFF (which fails)
            std_colors = {
                'White':  (0, 30, 200),
                'Yellow': (30, 140, 200), # Yellow inherently loses saturation easily on cameras
                'Orange': (13, 170, 200),
                'Red':    (0,  180, 150),
                'Green':  (65, 140, 150),
                'Blue':   (110, 150, 150)
            }
            
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
    st.markdown("## 🗺️ Live Cube Map")
    st.markdown(render_live_map(), unsafe_allow_html=True) 
    st.info("💡 **Demo Strategy:** If the AI misidentifies a color due to room lighting, just click the buttons on the right to manually fix it!")
    
# --- 4. Main User Interface ---
st.title("🧊 AI Rubik's Solver")

tabs = st.tabs([f"{COLOR_EMOJIS[CENTER_COLORS[f]]} {f} Face" for f in FACES])

for idx, tab in enumerate(tabs):
    current_face = FACES[idx]
    
    with tab:
        st.info(f"🧭 **HOW TO HOLD:** {ORIENTATION_GUIDE[current_face]}")
        col_camera, col_manual = st.columns([1, 1])
        
        with col_camera:
            st.write(f"### 📷 Camera Scanner")
            st.write("Align the cube inside the dark central box.")
            
            img_buffer = st.camera_input("Take a picture", key=f"cam_{current_face}")
            
            # Prevent state-overwrite bug by tracking photo IDs
            if img_buffer is not None:
                photo_id = img_buffer.file_id
                if st.session_state.processed_photos.get(current_face) != photo_id:
                    detected, debug_img = extract_colors_from_image(img_buffer, CENTER_COLORS[current_face])
                    st.session_state.cube_state[current_face] = detected
                    st.session_state[f"debug_{current_face}"] = debug_img
                    st.session_state.processed_photos[current_face] = photo_id
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
