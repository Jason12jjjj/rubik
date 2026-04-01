# ==============================================================================
# RUBIK'S CUBE SOLVER - STREAMLIT INTERFACE (V4 - Ultimate User Friendly)
# ==============================================================================
import streamlit as st
from rubiks_core import validate_cube_state, solve_cube

# --- 1. System Configuration & State Management ---
st.set_page_config(page_title="AI Rubik's Solver", page_icon="🧊", layout="wide")

AVAILABLE_COLORS = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
FACES = ['Up', 'Left', 'Front', 'Right', 'Back', 'Down']

# UX UPGRADE 1: Color Emojis for faster visual recognition
COLOR_EMOJIS = {
    'White': '⬜', 'Red': '🟥', 'Green': '🟩', 
    'Yellow': '🟨', 'Orange': '🟧', 'Blue': '🟦'
}

# Standard center colors
CENTER_COLORS = {
    'Up': 'White', 'Left': 'Orange', 'Front': 'Green', 
    'Right': 'Red', 'Back': 'Blue', 'Down': 'Yellow'
}

# UX UPGRADE 2: Strict Orientation Guides to prevent Parity Errors
# This tells the user EXACTLY how to hold the cube for each face.
ORIENTATION_GUIDE = {
    'Up':    "Look at the **⬜ White Face**. Ensure the **🟦 Blue Center** is pointing UP (towards the ceiling).",
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
    """Simulates CV processing."""
    dummy = ['White'] * 9
    dummy[4] = expected_center
    return dummy

# --- 3. User Interface Design ---
st.title("🧊 Rubik's Cube Solver (Pro Version)")
st.markdown("""
Welcome! To solve your cube, please input the colors for all 6 sides. 
**⚠️ CRITICAL:** You must hold the cube exactly as instructed in the blue orientation box for each face!
""")

tabs = st.tabs([f"{COLOR_EMOJIS[CENTER_COLORS[f]]} {f} Face" for f in FACES])

for idx, tab in enumerate(tabs):
    current_face = FACES[idx]
    
    with tab:
        # Display the explicit holding instruction
        st.info(f"🧭 **HOW TO HOLD:** {ORIENTATION_GUIDE[current_face]}")
        
        col_camera, col_manual = st.columns([1, 1])
        
        with col_camera:
            st.write(f"### 📷 Auto-Scan")
            img_buffer = st.camera_input("Take a picture", key=f"cam_{current_face}")
            if img_buffer is not None:
                detected = extract_colors_from_image(img_buffer, CENTER_COLORS[current_face])
                st.session_state.cube_state[current_face] = detected
                st.success(f"Scanned {current_face} face successfully!")
                st.rerun()
                
        with col_manual:
            st.write(f"### 🖱️ Manual Edit")
            st.write("Read the cube from **Left-to-Right, Top-to-Bottom**.")
            
            grid_cols = st.columns(3)
            for i in range(9):
                col_idx = i % 3
                with grid_cols[col_idx]:
                    if i == 4:
                        st.success(f"Center\n\n{COLOR_EMOJIS[CENTER_COLORS[current_face]]} **{CENTER_COLORS[current_face]}**")
                    else:
                        current_color = st.session_state.cube_state[current_face][i]
                        new_color = st.selectbox(
                            f"Tile {i+1}", 
                            AVAILABLE_COLORS, 
                            index=AVAILABLE_COLORS.index(current_color), 
                            format_func=lambda x: f"{COLOR_EMOJIS[x]} {x}", # Adds emojis to the dropdown!
                            key=f"manual_{current_face}_{i}"
                        )
                        st.session_state.cube_state[current_face][i] = new_color

st.divider()

# --- 4. System Validation ---
st.subheader("🚀 Final Step: Validation & Solution")

if st.button("Validate All 6 Sides & Solve", type="primary", use_container_width=True):
    is_valid, validation_msg = validate_cube_state(st.session_state.cube_state)
    
    if is_valid:
        with st.spinner("Executing Kociemba Algorithm..."):
            solution = solve_cube(st.session_state.cube_state)
            if "Error" in solution:
                st.error(solution)
                st.warning("Hint: Did you hold the cube exactly as the blue instructions said? A rotated face causes this error!")
            else:
                st.balloons()
                st.success("🎉 Solution Found!")
                st.markdown(f"### ➡️ Steps: `{solution}`")
    else:
        st.error(f"❌ Validation Failed: {validation_msg}")
        st.warning("A real Rubik's cube must have exactly 9 tiles of each color. Check the tabs to find the mistake.")
