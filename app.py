# ==============================================================================
# RUBIK'S CUBE SOLVER - STREAMLIT INTERFACE
# ==============================================================================
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from rubiks_core import validate_cube_state, solve_cube

# --- 1. System Configuration & State Management ---
st.set_page_config(page_title="AI Rubik's Solver", page_icon="🧊", layout="wide")

# Initialize session state to store the scanned faces
if 'scanned_faces' not in st.session_state:
    st.session_state.scanned_faces = {}

AVAILABLE_COLORS = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
FACES_TO_SCAN = ['Up (White Center)', 'Front (Green Center)', 'Right (Red Center)', 
                 'Back (Blue Center)', 'Left (Orange Center)', 'Down (Yellow Center)']

# --- 2. Dummy Computer Vision Function (Replace with your actual CV logic) ---
def extract_colors_from_image(image_bytes):
    """
    Simulates Computer Vision processing. 
    In a full implementation, this uses OpenCV (cv2) to convert the image 
    to HSV, find contours, map the 3x3 grid, and classify colors via ML/KNN.
    """
    # For prototype demonstration, we return a default grid.
    # [Group Member A & B will replace this with actual OpenCV grid extraction]
    return ['White', 'White', 'White', 
            'White', 'White', 'White', 
            'White', 'White', 'White']

# --- 3. User Interface Design ---
st.title("🧊 Computer Vision Rubik's Cube Solver")
st.markdown("Use your camera to scan the 6 faces. **Verify and correct** the AI's color detection to account for lighting anomalies.")

col_camera, col_data = st.columns([1, 1])

with col_camera:
    st.subheader("📷 1. Scan Cube Face")
    # Dropdown to select which face is being scanned
    current_face = st.selectbox("Select face to scan:", FACES_TO_SCAN)
    face_key = current_face.split(' ')[0] # Extract 'Up', 'Front', etc.
    
    # Streamlit Camera Input
    img_buffer = st.camera_input("Align face to center and take a picture")
    
    if img_buffer is not None:
        with st.spinner("AI analyzing colors using Computer Vision..."):
            # Call the CV function
            detected_colors = extract_colors_from_image(img_buffer)
            # Save to session state
            st.session_state.scanned_faces[face_key] = detected_colors
            st.success(f"Successfully scanned {face_key} face!")

with col_data:
    st.subheader("📝 2. Verification & Human-in-the-Loop")
    st.markdown("Review the AI detection. Modify any incorrect colors manually.")
    
    # Display the grid for the currently selected face if it exists
    face_key = current_face.split(' ')[0]
    if face_key in st.session_state.scanned_faces:
        colors = st.session_state.scanned_faces[face_key]
        
        # Create a 3x3 interactive grid using Streamlit columns
        st.write(f"**Current state for {face_key} face:**")
        grid_cols = st.columns(3)
        
        for i in range(9):
            col_idx = i % 3
            with grid_cols[col_idx]:
                # Allows user to override the AI's prediction (Exception Handling)
                new_color = st.selectbox(f"Tile {i+1}", AVAILABLE_COLORS, 
                                         index=AVAILABLE_COLORS.index(colors[i]), 
                                         key=f"{face_key}_{i}")
                st.session_state.scanned_faces[face_key][i] = new_color

st.divider()

# --- 4. Validation and Algorithm Execution ---
st.subheader("🚀 3. System Validation & Solution")

# Check if all 6 faces have been scanned
if len(st.session_state.scanned_faces) == 6:
    st.info("All 6 faces scanned. Running validation protocols...")
    
    # Business Rule Validation
    is_valid, validation_msg = validate_cube_state(st.session_state.scanned_faces)
    
    if is_valid:
        st.success("✅ Validation Passed: The cube state is mathematically valid.")
        
        if st.button("Generate Optimal Solution", type="primary", use_container_width=True):
            with st.spinner("Executing Kociemba Algorithm..."):
                solution = solve_cube(st.session_state.scanned_faces)
                
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
        st.warning("Please check the 'Verification' tab and fix any color misclassifications caused by lighting.")
else:
    st.warning(f"Scan progress: {len(st.session_state.scanned_faces)}/6 faces completed. Please scan the remaining faces.")