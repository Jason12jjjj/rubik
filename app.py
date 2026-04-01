# ==============================================================================
# RUBIK'S CUBE SOLVER - STREAMLIT INTERFACE (V8 - Live AR Grid & Better Color)
# ==============================================================================
import streamlit as st
import numpy as np
import cv2
import threading # NEW: For safe multithreading data transfer
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
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

# Essential for cloud deployment to traverse firewalls
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Thread safety mechanism to transfer data from video thread to streamlit thread
state_lock = threading.Lock()
if 'cube_state' not in st.session_state:
    st.session_state.cube_state = {}
    for face in FACES:
        default_face = ['White'] * 9 
        default_face[4] = CENTER_COLORS[face] 
        st.session_state.cube_state[face] = default_face
    # Shared object to hold temporary scan results across threads
    st.session_state.temp_scan = {'current_colors': None}

# --- 2. Live Video Processing Class (WebRTC) ---
class CubeFaceProcessor(VideoProcessorBase):
    def __init__(self):
        self.colors_to_face = CENTER_COLORS
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width, _ = img.shape
        grid_size = min(height, width) // 2 
        cell_size = grid_size // 3
        start_x = (width - grid_size) // 2
        start_y = (height - grid_size) // 2
        
        # 🌟 UX UPGRADE 1: Draw the Live Targeting Grid (3x3 boxes)
        cv2.rectangle(img, (start_x, start_y), (start_x + grid_size, start_y + grid_size), (255, 255, 255), 4)
        for i in range(1, 3):
            cv2.line(img, (start_x + i * cell_size, start_y), (start_x + i * cell_size, start_y + grid_size), (255, 255, 255), 2)
            cv2.line(img, (start_x, start_y + i * cell_size), (start_x + grid_size, start_y + i * cell_size), (255, 255, 255), 2)

        # Process HSV for color detection
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        frame_detected_colors = []
        
        # 🌟 LOGIC UPGRADE: More Robust Color Detection Logic
        def get_color_name(h, s, v):
            # Focus on low saturation for white, ignoring brightness (fixes shadows)
            if s < 50: return 'White' 
            
            # Improved Hue ranges based on reference scales
            if 15 <= h <= 35: return 'Yellow'
            elif 35 < h <= 85: return 'Green'
            elif 85 < h <= 130: return 'Blue'
            elif 5 <= h < 15: return 'Orange'
            # Red wraps around 0/180
            elif (0 <= h < 5 or 150 <= h <= 179): return 'Red'
            
            # Default to white if uncertain
            return 'White' 

        for row in range(3):
            for col in range(3):
                cx = start_x + (col * cell_size) + (cell_size // 2)
                cy = start_y + (row * cell_size) + (cell_size // 2)
                
                # Take 10x10 ROI median to reduce noise
                roi = hsv_img[cy-5:cy+5, cx-5:cx+5]
                avg_h, avg_s, avg_v = np.median(roi, axis=(0, 1)).astype(int)
                color_name = get_color_name(avg_h, avg_s, avg_v)
                frame_detected_colors.append(color_name)
                
                # 🌟 UX UPGRADE 2: Draw the Live Color Dot inside each grid square
                dot_color_bgr = (0, 0, 0) # Mapping would go here for real color feedback, using text for now
                # Draw black shadow for text readability
                cv2.putText(img, color_name, (cx-20, cy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                # Draw colored text (Red dots for feedback)
                cv2.circle(img, (cx, cy), 12, (0, 0, 255), -1) 
                cv2.circle(img, (cx, cy), 14, (255, 255, 255), 2) # White outline

        # Thread-safe update of the temporary shared state
        with state_lock:
            st.session_state.temp_scan['current_colors'] = frame_detected_colors

        # Return the frame with the grid and dots drawn on it for live display
        return frame.from_ndarray(img, format="bgr24")

# --- 3. Sidebar UI (Mini-Map) ---
with st.sidebar:
    st.markdown("## 🗺️ Live Cube Map")
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
            else: html += '<div></div>'
    html += '</div>'
    st.components.v1.html(html, height=250)
    st.info("💡 **Tip:** AI saw it wrong? Click the color buttons to fix them!")

# --- 4. Main User Interface ---
st.title("🧊 AI Rubik's Solver")

tabs = st.tabs([f"{COLOR_EMOJIS[CENTER_COLORS[f]]} {f}" for f in FACES])

for idx, tab in enumerate(tabs):
    current_face = FACES[idx]
    
    with tab:
        st.info(f"🧭 **HOW TO HOLD:** {ORIENTATION_GUIDE[current_face]}")
        col_camera, col_manual = st.columns([2, 1])
        
        with col_camera:
            st.write(f"### 📷 Live AR Scanner")
            st.write("Align the cube inside the white grid. Colors update in real-time below.")
            
            # 🌟 CORE UPGRADE: WebRTC Streamer provides LIVE FEED with drawn Grid/Dots
            ctx = webrtc_streamer(
                key=f"scan_{current_face}", 
                video_processor_factory=CubeFaceProcessor, 
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False}, # Video only
                # Wait for the user to start the camera explicitly
                async_processing=True 
            )

            # Button to "Snapshot" the current live colors into session state
            if st.button(f"Confirm & Lock Scanned Colors", key=f"conf_{current_face}", type="primary"):
                with state_lock:
                    live_colors = st.session_state.temp_scan['current_colors']
                
                if live_colors:
                    # Enforce fixed center
                    live_colors[4] = CENTER_COLORS[current_face]
                    st.session_state.cube_state[current_face] = live_colors
                    st.success(f"Colors locked for {current_face} face! Check the map.")
                    st.rerun() # Refresh map
                else:
                    st.warning("Camera not started or active. Please start the camera and align the cube.")
                
        with col_manual:
            st.write(f"### 🖱️ Manual Edit / Verify")
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

# --- 5. Final Step ---
if st.button("Validate & Generate optimal solution", type="primary", use_container_width=True):
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
