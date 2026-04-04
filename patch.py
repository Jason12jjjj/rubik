import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Chunk 1: CSS and State init
css_target = """# --- 🌟 CSS HACK: Overlay a Targeting Box on the Camera ---
st.markdown(\"\"\"
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
\"\"\", unsafe_allow_html=True)"""

css_replace = """if 'cube_size' not in st.session_state:
    st.session_state.cube_size = 50

# --- 🌟 CSS HACK: Overlay a Targeting Box on the Camera ---
c_size = st.session_state.get('cube_size', 50)
st.markdown(f\"\"\"
<style>
    /* This creates a 3x3 aiming grid over the Streamlit camera */
    [data-testid="stCameraInput"] > div:first-child::after {{
        content: "";
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        width: {c_size}%;
        max-width: {c_size}vh;
        aspect-ratio: 1 / 1;
        border: 3px solid rgba(0, 255, 0, 0.8);
        pointer-events: none; /* Let clicks pass through */
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5); /* Darken surroundings */
        
        /* Draw the internal 3x3 Grid lines */
        background-image: 
            linear-gradient(to right, transparent 33.33%, rgba(0, 255, 0, 0.6) 33.33%, rgba(0, 255, 0, 0.6) calc(33.33% + 2px), transparent calc(33.33% + 2px)),
            linear-gradient(to right, transparent 66.66%, rgba(0, 255, 0, 0.6) 66.66%, rgba(0, 255, 0, 0.6) calc(66.66% + 2px), transparent calc(66.66% + 2px)),
            linear-gradient(to bottom, transparent 33.33%, rgba(0, 255, 0, 0.6) 33.33%, rgba(0, 255, 0, 0.6) calc(33.33% + 2px), transparent calc(33.33% + 2px)),
            linear-gradient(to bottom, transparent 66.66%, rgba(0, 255, 0, 0.6) 66.66%, rgba(0, 255, 0, 0.6) calc(66.66% + 2px), transparent calc(66.66% + 2px));
    }}
</style>
\"\"\", unsafe_allow_html=True)"""

content = content.replace(css_target, css_replace)

# Chunk 2: Grid size logic in CV
grid_target = """    height, width, _ = img.shape
    grid_size = min(height, width) // 2 
    cell_size = grid_size // 3"""

grid_replace = """    grid_scale = st.session_state.get('cube_size', 50)
    height, width, _ = img.shape
    grid_size = int(min(height, width) * (grid_scale / 100.0))
    cell_size = grid_size // 3"""

content = content.replace(grid_target, grid_replace)

# Chunk 3: Sidebar Slider
sidebar_target = """    st.markdown("## 🧭 Navigation")
    app_mode = st.radio("Choose Mode:", ["📸 Scan & Solve", "⚙️ Tune Colors"])
    st.divider()
    
    st.markdown("## 🗺️ Live Cube Map")"""

sidebar_replace = """    st.markdown("## 🧭 Navigation")
    app_mode = st.radio("Choose Mode:", ["📸 Scan & Solve", "⚙️ Tune Colors"])
    st.divider()
    
    st.markdown("## 📐 Camera Tool")
    st.slider("📏 Viewport Grid Size", min_value=30, max_value=80, key="cube_size", help="Resize the green scanning box to perfectly fit your Rubik's Cube.")
    
    st.divider()
    st.markdown("## 🗺️ Live Cube Map")"""

content = content.replace(sidebar_target, sidebar_replace)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Replaced!")
