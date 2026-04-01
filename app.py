# ==============================================================================
# RUBIK'S CUBE SOLVER - STREAMLIT INTERFACE (V3 - Six Sides Tabs Layout)
# ==============================================================================
import streamlit as st
from rubiks_core import validate_cube_state, solve_cube

# --- 1. System Configuration & State Management ---
st.set_page_config(page_title="AI Rubik's Solver", page_icon="🧊", layout="wide")

AVAILABLE_COLORS = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
FACES = ['Up', 'Left', 'Front', 'Right', 'Back', 'Down']

# Standard center colors for reference (Centers are physically fixed on a cube)
CENTER_COLORS = {
    'Up': 'White', 'Left': 'Orange', 'Front': 'Green', 
    'Right': 'Red', 'Back': 'Blue', 'Down': 'Yellow'
}

# Initialize the 6 sides of the cube state
if 'cube_state' not in st.session_state:
    st.session_state.cube_state = {}
    for face in FACES:
        default_face = ['White'] * 9 
        default_face[4] = CENTER_COLORS[face] # Lock the center tile
        st.session_state.cube_state[face] = default_face

# --- 2. Dummy Computer Vision Function ---
def extract_colors_from_image(image_bytes, expected_center):
    """Simulates CV processing."""
    dummy = ['White'] * 9
    dummy[4] = expected_center
    return dummy

# --- 3. User Interface Design (The 6 Sides Tabs) ---
st.title("🧊 Rubik's Cube Solver (6 Sides)")
st.markdown("A Rubik's cube strictly requires **6 faces** to be scanned. Navigate through the tabs below to configure each side.")

# Create 6 distinct tabs for the 6 faces!
tabs = st.tabs([f"Cube Face: {f} ({CENTER_COLORS[f]} Center)" for f in FACES])

# Loop through each tab to generate the UI for all 6 sides
for idx, tab in enumerate(tabs):
    current_face = FACES[idx]
    
    with tab:
        col_camera, col_manual = st.columns([1, 1])
        
        # Option A: Camera for this specific face
        with col_camera:
            st.write(f"### 📷 Auto-Scan: {current_face} Face")
            img_buffer = st.camera_input("Take a picture", key=f"cam_{current_face}")
            
            if img_buffer is not None:
                detected = extract_colors_from_image(img_buffer, CENTER_COLORS[current_face])
                st.session_state.cube_state[current_face] = detected
                st.success(f"Scanned {current_face} face successfully!")
                
        # Option B: Manual Input for this specific face
        with col_manual:
            st.write(f"### 🖱️ Manual Edit: {current_face} Face")
            
            # Create a 3x3 interactive grid
            grid_cols = st.columns(3)
            for i in range(9):
                col_idx = i % 3
                with grid_cols[col_idx]:
                    if i == 4:
                        # Center piece is locked
                        st.info(f"Center\n**{CENTER_COLORS[current_face]}**")
                    else:
                        # Dropdown for the other 8 pieces
                        current_color = st.session_state.cube_state[current_face][i]
                        new_color = st.selectbox(
                            f"Tile {i+1}", 
                            AVAILABLE_COLORS, 
                            index=AVAILABLE_COLORS.index(current_color), 
                            key=f"manual_{current_face}_{i}"
                        )
                        st.session_state.cube_state[current_face][i] = new_color

st.divider()

# --- 4. System Validation (Checks all 6 sides) ---
st.subheader("🚀 Final Step: Validation & Solution")
st.write("Ensure all 6 tabs above are configured correctly before generating the solution.")

if st.button("Validate All 6 Sides & Solve", type="primary", use_container_width=True):
    # The validate_cube_state function strictly checks if 54 tiles exist (9 of each color)
    is_valid, validation_msg = validate_cube_state(st.session_state.cube_state)
    
    if is_valid:
        st.success("✅ Validation Passed: All 6 sides are mathematically valid.")
        with st.spinner("Executing Kociemba Algorithm..."):
            solution = solve_cube(st.session_state.cube_state)
            
            if "Error" in solution:
                st.error(solution)
            else:
                st.balloons()
                st.success("🎉 Solution Found!")
                st.markdown(f"### ➡️ Steps: `{solution}`")
    else:
        st.error(f"❌ Validation Failed: {validation_msg}")
        st.warning("A real Rubik's cube must have exactly 9 tiles of each color across its 6 sides. Please check all 6 tabs above.")
