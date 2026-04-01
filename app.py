# ==============================================================================
# RUBIK'S CUBE SOLVER - STREAMLIT INTERFACE (V7 - Targeting Grid & Bug Fixes)
# ==============================================================================
import numpy as np
import cv2
import streamlit as st
from rubiks_core import validate_cube_state, solve_cube

# --- 1. System Configuration & State Management ---
st.set_page_config(page_title="AI Rubik's Solver", page_icon="🧊", layout="wide")

AVAILABLE_COLORS = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
FACES = ['Up', 'Left', 'Front', 'Right', 'Back', 'Down']

COLOR_EMOJIS = {
    'White': '⬜', 'Red': '🟥', 'Green': '🟩', 
    'Yellow': '🟨', 'Orange': '🟧', 'Blue': '🟦'
}

HEX_COLORS = {
    'White': '#f8f9fa', 'Red': '#ff4b4b', 'Green': '#09ab3b', 
    'Yellow': '#ffeb3b', 'Orange': '#ffa500', 'Blue': '#1e88e5'
}

CENTER_COLORS = {
    'Up': 'White', 'Left': 'Orange', 'Front': 'Green', 
    'Right': 'Red', 'Back': 'Blue', 'Down': 'Yellow'
}

ORIENTATION_GUIDE = {
    'Up':    "Look at the **⬜ White Face**. Ensure the **🟦 Blue Center** is pointing UP.",
    'Left':  "Look at the **🟧 Orange Face**. Ensure the **⬜ White Center** is pointing UP.",
    'Front': "Look at the **🟩 Green Face**. Ensure the **⬜ White Center** is pointing UP.",
    'Right': "Look at the **🟥 Red Face**. Ensure the **⬜ White Center** is pointing UP.",
    'Back':  "Look at the **🟦 Blue Face**. Ensure the **⬜ White Center** is pointing UP.",
    'Down':  "Look at the **🟨 Yellow Face**. Ensure the **🟩 Green Center** is pointing UP."
}

if 'cube_state' not in st.session_state:
    st.session_state.cube_state = {}
    for face in FACES:
        default_face = ['White'] * 9 
        default_face[4] = CENTER_COLORS[face] 
        st.session_state.cube_state[face] = default_face

def extract_colors_from_image(image_bytes, expected_center):
    """Real CV with Targeting Grid and Improved Color Tolerance."""
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1) 
    
    height, width, _ = img.shape
    grid_size = min(height, width) // 2 
    cell_size = grid_size // 3
    
    start_x = (width - grid_size) // 2
    start_y = (height - grid_size) // 2
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    debug_img = img.copy()
    detected_colors = []
    
    # 🌟 UX UPGRADE 1: Draw a 3x3 Targeting Grid for the user
    cv2.rectangle(debug_img, (start_x, start_y), (start_x + grid_size, start_y + grid_size), (255, 255, 255), 3)
    for i in range(1, 3):
        cv2.line(debug_img, (start_x + i * cell_size, start_y), (start_x + i * cell_size, start_y + grid_size), (255, 255, 255), 2)
        cv2.line(debug_img, (start_x, start_y + i * cell_size), (start_x + grid_size, start_y + i * cell_size), (255, 255, 255), 2)

    def get_color_name(h, s, v):
        # 🌟 CV UPGRADE: Lowered Saturation threshold to fix Yellow/Red becoming White
        if s < 40: return 'White'  # Used to be 60
        
        # Adjusted Hue ranges
        if 20 <= h <= 35: return 'Yellow'
        elif 35 < h <= 85: return 'Green'
        elif 85 < h <= 130: return 'Blue'
        elif 5 <= h < 20: return 'Orange'
        elif (0 <= h < 5 or 150 <= h <= 179): return 'Red'
        
        return 'White' 

    for row in range(3):
        for col in range(3):
            cx = start_x + (col * cell_size) + (cell_size // 2)
            cy = start_y + (row * cell_size) + (cell_size // 2)
            
            roi = hsv_img[cy-5:cy+5, cx-5:cx+5]
            avg_h, avg_s, avg_v = np.median(roi, axis=(0, 1)).astype(int)
            color_name = get_color_name(avg_h, avg_s, avg_v)
            detected_colors.append(color_name)
            
            # Draw the AI sampling points
            cv2.circle(debug_img, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(debug_img, color_name, (cx-25, cy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(debug_img, color_name, (cx-25, cy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    detected_colors[4] = expected_center
    debug_img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
    
    return detected_colors, debug_img_rgb  

# --- 2. Live Mini-Map Generator ---
def render_live_map():
    html = '<div style="display: grid; grid-template-columns: repeat(4, 50px); gap: 5px; justify-content: center;">'
    grid_positions = {'Up': (1, 2), 'Left': (2, 1), 'Front': (2, 2), 'Right': (2, 3), 'Back': (2, 4), 'Down': (3, 2)}
    
    for row in range(1, 4):
        for col in range(1, 5):
            found_face = next((face for face, pos in grid_positions.items() if pos == (row, col)), None)
            
            if found_face:
                html += '<div style="display: grid; grid-template-columns: repeat(3, 15px); gap: 1px;">'
                for color in st.session_state.cube_state[found_face]:
                    hex_c = HEX_COLORS[color]
                    html += f'<div style="width: 15px; height: 15px; background-color: {hex_c}; border: 1px solid #444; border-radius: 2px;"></div>'
                html += '</div>'
            else:
                html += '<div></div>'
    html += '</div>'
    return html

# --- 3. Sidebar UI ---
with st.sidebar:
    st.markdown("## 🗺️ Live Cube Map")
    st.components.v1.html(render_live_map(), height=250)
    st.divider()
    st.info("💡 **Tip:** AI saw it wrong? Just click the colored buttons on the right to fix it instantly!")

# --- 4. Main User Interface ---
st.title("🧊 Rubik's Cube Solver")

tabs = st.tabs([f"{COLOR_EMOJIS[CENTER_COLORS[f]]} {f} Face" for f in FACES])

for idx, tab in enumerate(tabs):
    current_face = FACES[idx]
    
    with tab:
        st.info(f"🧭 **HOW TO HOLD:** {ORIENTATION_GUIDE[current_face]}")
        col_camera, col_manual = st.columns([1, 1])
        
        with col_camera:
            st.write(f"### 📷 Auto-Scan")
            img_buffer = st.camera_input("Take a picture", key=f"cam_{current_face}")
            
            # 🌟 LOGIC UPGRADE: Only process if it's a NEW photo (Fixes the state overwrite bug)
            if img_buffer is not None:
                if st.session_state.get(f"processed_{current_face}") != img_buffer.file_id:
                    detected, debug_img = extract_colors_from_image(img_buffer, CENTER_COLORS[current_face])
                    st.session_state.cube_state[current_face] = detected
                    st.session_state[f"debug_img_{current_face}"] = debug_img
                    st.session_state[f"processed_{current_face}"] = img_buffer.file_id
                    st.rerun() # Refresh to update the map
                
                # Show the saved image for this specific tab
                if f"debug_img_{current_face}" in st.session_state:
                    st.image(st.session_state[f"debug_img_{current_face}"], caption="Targeting Grid - Align your cube inside the white box!", use_column_width=True)
                
        with col_manual:
            st.write(f"### 🖱️ Manual Override")
            for row in range(3):
                cols = st.columns(3)
                for col_idx in range(3):
                    tile_idx = row * 3 + col_idx
                    with cols[col_idx]:
                        if tile_idx == 4:
                            st.button(f"{COLOR_EMOJIS[CENTER_COLORS[current_face]]} Ctr", key=f"lock_{current_face}", disabled=True, use_container_width=True)
                        else:
                            current_color = st.session_state.cube_state[current_face][tile_idx]
                            button_label = f"{COLOR_EMOJIS[current_color]} {current_color}"
                            
                            if st.button(button_label, key=f"btn_{current_face}_{tile_idx}", use_container_width=True):
                                current_index = AVAILABLE_COLORS.index(current_color)
                                next_index = (current_index + 1) % len(AVAILABLE_COLORS)
                                st.session_state.cube_state[current_face][tile_idx] = AVAILABLE_COLORS[next_index]
                                st.rerun()

st.divider()

# --- 5. System Validation ---
st.subheader("🚀 Final Step: Validation & Solution")
if st.button("Validate All 6 Sides & Solve", type="primary", use_container_width=True):
    is_valid, validation_msg = validate_cube_state(st.session_state.cube_state)
    if is_valid:
        with st.spinner("Executing Algorithm..."):
            solution = solve_cube(st.session_state.cube_state)
            if "Error" in solution:
                st.error(solution)
            else:
                st.balloons()
                st.success("🎉 Solution Found!")
                st.markdown(f"### ➡️ Steps: `{solution}`")
    else:
        st.error(f"❌ Validation Failed: {validation_msg}")
