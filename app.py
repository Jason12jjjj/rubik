# ==============================================================================
# RUBIK'S CUBE SOLVER - STREAMLIT INTERFACE (V2 - Camera & Manual Mode)
# ==============================================================================
import streamlit as st
from rubiks_core import validate_cube_state, solve_cube

# --- 1. System Configuration & State Management ---
st.set_page_config(page_title="AI Rubik's Solver", page_icon="🧊", layout="wide")

AVAILABLE_COLORS = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
FACES = ['Up', 'Front', 'Right', 'Back', 'Left', 'Down']

# Standard center colors for reference (Centers are physically fixed on a cube)
CENTER_COLORS = {
    'Up': 'White', 'Front': 'Green', 'Right': 'Red', 
    'Back': 'Blue', 'Left': 'Orange', 'Down': 'Yellow'
}

# Initialize the cube state with default colors 
if 'cube_state' not in st.session_state:
    st.session_state.cube_state = {}
    for face in FACES:
        default_face = ['White'] * 9 # Default all tiles to white initially
        default_face[4] = CENTER_COLORS[face] # Lock the center tile to its correct color
        st.session_state.cube_state[face] = default_face

# --- 2. Dummy Computer Vision Function ---
def extract_colors_from_image(image_bytes, expected_center):
    """
    Simulates CV processing. Returns a dummy grid with the correct center.
    [Group Member will replace this with actual OpenCV logic later]
    """
    dummy = ['White'] * 9
    dummy[4] = expected_center
    return dummy

# --- 3. User Interface Design ---
st.title("🧊 Rubik's Cube Solver")
st.markdown("Configure your cube state below. You can use the **Camera** to auto-detect, or **Manually Enter** the colors yourself.")

# Dropdown to select which face the user is currently editing
current_face_label = st.selectbox("🎯 Select Face to Edit:", 
                                  [f"{f} Face (Center: {CENTER_COLORS[f]})" for f in FACES])
current_face = current_face_label.split(' ')[0] # Extract just the face name (e.g., 'Up')

col_camera, col_manual = st.columns([1, 1])

# OPTION A: Camera Input
with col_camera:
    st.subheader("📷 Option A: Camera Auto-Scan")
    st.write(f"Align the **{current_face}** face to the camera.")
    
    # We use a dynamic key so the camera refreshes when the user changes faces
    img_buffer = st.camera_input("Take a picture", key=f"cam_{current_face}")
    
    if img_buffer is not None:
        with st.spinner("Analyzing colors..."):
            detected = extract_colors_from_image(img_buffer, CENTER_COLORS[current_face])
            st.session_state.cube_state[current_face] = detected
            st.success(f"Camera scan applied to {current_face} face! Check the manual grid to verify.")

# OPTION B: Manual Input (Always visible and interactive)
with col_manual:
    st.subheader("🖱️ Option B: Manual Color Entry")
    st.write("Click the dropdowns to set the colors exactly as they appear on your cube.")
    
    # Create a 3x3 interactive grid for the currently selected face
    grid_cols = st.columns(3)
    for i in range(9):
        col_idx = i % 3
        with grid_cols[col_idx]:
            if i == 4:
                # Tile index 4 is the center piece (cannot be changed)
                st.info(f"Center\n**{CENTER_COLORS[current_face]}**")
            else:
                # Interactive dropdown for the 8 edge and corner pieces
                current_color = st.session_state.cube_state[current_face][i]
                new_color = st.selectbox(
                    f"Tile {i+1}", 
                    AVAILABLE_COLORS, 
                    index=AVAILABLE_COLORS.index(current_color), 
                    key=f"manual_{current_face}_{i}"
                )
                # Instantly update the session state when the user changes a dropdown
                st.session_state.cube_state[current_face][i] = new_color

st.divider()

# --- 4. Validation and Algorithm Execution ---
st.subheader("🚀 System Validation & Solution")
st.write("Once you have configured all 6 faces, click the button below to solve.")

if st.button("Validate & Generate Solution", type="primary", use_container_width=True):
    # Pass the entire cube state to the core logic for validation
    is_valid, validation_msg = validate_cube_state(st.session_state.cube_state)
    
    if is_valid:
        st.success("✅ Validation Passed: The cube state is mathematically valid.")
        with st.spinner("Executing Kociemba Algorithm..."):
            solution = solve_cube(st.session_state.cube_state)
            
            if "Error" in solution:
                st.error(solution)
            else:
                st.balloons()
                st.success("🎉 Solution Found!")
                st.markdown(f"### ➡️ Steps: `{solution}`")
                st.markdown("""
                **Notation Guide:**
                * **U, R, F, D, L, B:** Turn that face 90° clockwise.
                * **' (Prime):** Turn that face 90° counter-clockwise (e.g., U').
                * **2:** Turn that face 180° (e.g., R2).
                """)
    else:
        st.error(f"❌ Validation Failed: {validation_msg}")
        st.warning("Please check your colors. A real Rubik's cube must have exactly 9 squares of each color.")
