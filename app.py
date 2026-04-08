import os, sys, json, streamlit as st, streamlit.components.v1 as components
import numpy as np
import cv2
from rubiks_core import validate_cube_state, solve_cube

# --- 1. SYSTEM CONFIG ---
st.set_page_config(page_title="Manual Rubik's Solver", page_icon="🧊", layout="wide")

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Selected Color Glow Effect */
    .stButton > button[aria-pressed="true"], .active-color {
        border: 2px solid #00e5ff !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.6) !important;
        transform: scale(1.05);
    }
    /* Smooth transitions */
    .stButton > button { transition: all 0.2s ease; }
    /* Inventory Alerts */
    .inventory-warn { color: #ffa500; font-weight: bold; }
    .inventory-err { color: #ff4b4b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Constants
FACES = ['Up', 'Left', 'Front', 'Right', 'Back', 'Down']
HEX_COLORS = {'White':'#f8f9fa','Red':'#ff4b4b','Green':'#09ab3b','Yellow':'#ffeb3b','Orange':'#ffa500','Blue':'#1e88e5'}
COLOR_EMOJIS = {'White':'⬜','Red':'🟥','Green':'🟩','Yellow':'🟨','Orange':'🟧','Blue':'🟦'}
CENTER_COLORS = {'Up':'White','Left':'Orange','Front':'Green','Right':'Red','Back':'Blue','Down':'Yellow'}

CALIB_FILE = "calibration_profile.json"

def get_calibrated_colors():
    defaults = {'White':(0,30,220), 'Yellow':(30,160,200), 'Orange':(12,200,240), 'Red':(0,210,180), 'Green':(60,180,150), 'Blue':(110,180,160)}
    if 'custom_std_colors' in st.session_state:
        for k, v in st.session_state.custom_std_colors.items(): defaults[k] = tuple(v)
    return defaults

def run_manual_grid_extract(image_bytes, expected_center):
    """Reliable fixed-grid color extraction using erosion and centroid snapping."""
    img = cv2.imdecode(np.asarray(bytearray(image_bytes.read()), dtype=np.uint8), 1)
    if img is None: return None, None, False
    h, w = img.shape[:2]
    # Sample from the center area (0.7 scale)
    gs = int(min(h, w) * 0.7)
    sx, sy = (w - gs) // 2, (h - gs) // 2
    warped = cv2.resize(img[sy:sy+gs, sx:sx+gs], (300, 300))
    debug_view = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    std_colors = get_calibrated_colors()
    hsv_w = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    sat_w = hsv_w[:,:,1]
    
    detected = ['White'] * 9
    for r in range(3):
        for c in range(3):
            tx, ty = int((c+0.5)*100), int((r+0.5)*100)
            # Saturation Centroid Refinement
            y1, y2, x1, x2 = max(0,ty-35), min(300,ty+35), max(0,tx-35), min(300,tx+35)
            roi_sat = sat_w[y1:y2, x1:x2]
            moms = cv2.moments(roi_sat)
            fx, fy = tx, ty
            if moms["m00"] > 50:
                sx_l, sy_l = x1 + int(moms["m10"]/moms["m00"]), y1 + int(moms["m01"]/moms["m00"])
                if np.sqrt((sx_l-tx)**2 + (sy_l-ty)**2) < 30: fx, fy = sx_l, sy_l
            
            # Eroded ROI Sampling (User Suggested logic)
            roi_raw = warped[max(0,fy-8):min(300,fy+8), max(0,fx-8):min(300,fx+8)]
            if roi_raw.size > 0:
                hr, wr = roi_raw.shape[:2]
                roi_er = roi_raw[hr//4 : hr-hr//4, wr//4 : wr-wr//4]
                bgr = np.median(roi_er, axis=(0,1)).astype(np.uint8)
            else: bgr = [0,0,0]
            
            lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]
            min_d, best_c = 999.0, 'White'
            for name, (hs,ss,vs) in std_colors.items():
                t_lab = cv2.cvtColor(cv2.cvtColor(np.uint8([[[hs,ss,vs]]]), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2LAB)[0][0]
                # High-precision LAB coefficients from snippet
                dV = np.sqrt(0.1*(float(lab[0])-t_lab[0])**2 + 2.4*(float(lab[1])-t_lab[1])**2 + 2.4*(float(lab[2])-t_lab[2])**2)
                if dV < min_d: min_d, best_c = dV, name
            detected[r*3+c] = best_c
            cv2.circle(debug_view, (fx, fy), 12, (255,255,255), 2)
            cv2.putText(debug_view, best_c[0], (fx-5, fy+5), 0, 0.4, (255,255,255), 2)
            
    detected[4] = expected_center
    return detected, debug_view, True

# --- 2. SESSION STATE ---
if 'programmatic_face' not in st.session_state:
    st.session_state.programmatic_face = 'Front'
if 'cube_state' not in st.session_state:
    st.session_state.cube_state = {f: (['White']*4 + [CENTER_COLORS[f]] + ['White']*4) for f in FACES}
if 'last_solution' not in st.session_state:
    st.session_state.last_solution = None
if 'selected_color' not in st.session_state:
    st.session_state.selected_color = 'White'
if 'solve_speed' not in st.session_state:
    st.session_state.solve_speed = 1.0
if 'custom_std_colors' not in st.session_state:
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, 'r') as f:
                st.session_state.custom_std_colors = json.load(f)
        except Exception: st.session_state.custom_std_colors = {}
    else: st.session_state.custom_std_colors = {}
if 'history' not in st.session_state:
    st.session_state.history = [json.dumps(st.session_state.cube_state)]
if 'history_index' not in st.session_state:
    st.session_state.history_index = 0

def push_history():
    state_json = json.dumps(st.session_state.cube_state)
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history = st.session_state.history[:st.session_state.history_index + 1]
    st.session_state.history.append(state_json)
    st.session_state.history_index = len(st.session_state.history) - 1

# --- 3. UI COMPONENTS ---
def render_sidebar_map():
    """Renders a 2D clickable net of the cube using native buttons for speed"""
    grid = [
        [None, 'Up', None, None],
        ['Left', 'Front', 'Right', 'Back'],
        [None, 'Down', None, None]
    ]
    active = st.session_state.programmatic_face
    for row_idx, row in enumerate(grid):
        cols = st.columns(4)
        for col_idx, f_k in enumerate(row):
            if f_k:
                is_active = (f_k == active)
                # Dynamic Pointer logic
                label = f"☞{f_k[0]}" if is_active else f_k[0]
                # Button color logic via emoji prefix
                emoji = COLOR_EMOJIS[CENTER_COLORS[f_k]]
                if cols[col_idx].button(f"{emoji}\n{label}", key=f"nav_{f_k}", use_container_width=True):
                    st.session_state.programmatic_face = f_k
                    st.rerun()

def render_3d_player(solution):
    """Embeds the Twisty Player for solution visualization"""
    def inv_alg(s):
        r = []
        for m in reversed(s.split()):
            if "'" in m: r.append(m.replace("'",""))
            elif "2" in m: r.append(m)
            else: r.append(m+"'")
        return " ".join(r)
    speed = st.session_state.get('solve_speed', 1.0)
    html = f"""<div style="background:#000; border:1px solid #00e5ff; border-radius:15px; padding:20px; box-shadow:0 0 20px rgba(0,229,255,0.2);">
        <script src="https://cubing.net" type="module"></script>
        <twisty-player experimental-setup-alg="{inv_alg(solution)}" alg="{solution}" background="none" tempo-scale="{speed}" control-panel="bottom-row" style="width:100%; height:400px;"></twisty-player>
    </div>"""
    components.html(html, height=450)

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🧩 Solver Console")
    app_mode = st.radio("App Mode:", ["📸 Scan & Solve", "⚙️ Tune Colors"], horizontal=True)
    
    st.divider()
    if app_mode == "📸 Scan & Solve":
        st.markdown("### 🗺️ Context Map")
        render_sidebar_map()
        
        st.divider()
        with st.expander("🎓 Beginner Academy", expanded=False):
            tab1, tab2, tab3 = st.tabs(["Steps 1-2", "Steps 3-4", "Algs"])
            with tab1:
                st.markdown("**Step 1: The White Cross**")
                st.caption("Daisy around yellow, then flip to white.")
                st.code("Sexy: R U R' U'", language="markdown")
            with tab2:
                st.markdown("**Step 3: Mid-layer**")
                st.code("U R U' R' U' F' U F", language="markdown")
                st.markdown("**Step 4: Top Cross**")
                st.code("F R U R' U' F'", language="markdown")
            with tab3:
                st.markdown("**Sune:** R U R' U R U2 R'")
                st.markdown("**Niklas:** U R U' L' U R' U' L")

        st.divider()
        st.markdown("### 📊 Inventory Check")
        all_stickers = [s for f in FACES for s in st.session_state.cube_state[f]]
        counts = {c: all_stickers.count(c) for c in HEX_COLORS.keys()}
        cols = st.columns(3)
        for i, name in enumerate(HEX_COLORS.keys()):
            cols[i%3].markdown(f"{COLOR_EMOJIS[name]} `{counts[name]}/9`")

        st.divider()
        if st.button("🗑️ Reset All Colors", use_container_width=True):
            st.session_state.cube_state = {f: (['White']*4 + [CENTER_COLORS[f]] + ['White']*4) for f in FACES}
            st.session_state.last_solution = None; push_history(); st.rerun()

# --- 5. MAIN INTERFACE ---
if app_mode == "📸 Scan & Solve":
    curr = st.session_state.programmatic_face
    st.title("🧊 Pro Rubik's Solver")

    # Face Switching Header
    h1, h2, h3 = st.columns([1,3,1])
    with h1: 
        if st.button("⬅️ Prev", use_container_width=True):
            st.session_state.programmatic_face = FACES[(FACES.index(curr)-1)%6]; st.rerun()
    with h2:
        face_emoji = COLOR_EMOJIS[CENTER_COLORS[curr]]
        st.markdown(f"<h2 style='text-align:center; color:#00e5ff;'>☞ {face_emoji} {curr} ☜</h2>", unsafe_allow_html=True)
    with h3:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.programmatic_face = FACES[(FACES.index(curr)+1)%6]; st.rerun()

    # UNDO / REDO CONTROLS
    u1, u2, u3 = st.columns([1, 1, 4])
    with u1:
        if st.button("⏪ Undo", disabled=st.session_state.history_index <= 0, use_container_width=True):
            st.session_state.history_index -= 1
            st.session_state.cube_state = json.loads(st.session_state.history[st.session_state.history_index])
            st.rerun()
    with u2:
        if st.button("Redo ⏩", disabled=st.session_state.history_index >= len(st.session_state.history)-1, use_container_width=True):
            st.session_state.history_index += 1
            st.session_state.cube_state = json.loads(st.session_state.history[st.session_state.history_index])
            st.rerun()

    st.info(f"💡 Select a color from the Palette then click the squares to fill. Center is fixed to **{CENTER_COLORS[curr]}**.")

    c_edit, c_sol = st.columns([1, 1])

    with c_edit:
        st.markdown("#### 🎨 Color Palette")
        pal_cols = st.columns(6)
        for i, color_name in enumerate(HEX_COLORS.keys()):
            is_selected = st.session_state.selected_color == color_name
            btn_label = f"{COLOR_EMOJIS[color_name]} {'•' if is_selected else ''}"
            if pal_cols[i].button(btn_label, key=f"pal_{color_name}", use_container_width=True):
                st.session_state.selected_color = color_name
                st.rerun()

        st.divider()
        
        def paint_color(face_name, index):
            st.session_state.cube_state[face_name][index] = st.session_state.selected_color
            st.session_state.last_solution = None
            push_history()

        # Center color enforcement before rendering
        if st.session_state.cube_state[curr][4] != CENTER_COLORS[curr]:
            st.session_state.cube_state[curr][4] = CENTER_COLORS[curr]

        for r in range(3):
            rows = st.columns(3)
            for c in range(3):
                idx = r*3 + c
                color_val = st.session_state.cube_state[curr][idx]
                is_active = (idx != 4)
                if not is_active:
                    rows[c].button(f"🔒\n{COLOR_EMOJIS[CENTER_COLORS[curr]]}", disabled=True, use_container_width=True)
                else:
                    rows[c].button(f"{COLOR_EMOJIS[color_val]}\n{color_val}", key=f"btn_{curr}_{idx}", on_click=paint_color, args=(curr, idx), use_container_width=True)

        st.divider()
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🧹 Reset This Face", use_container_width=True):
                st.session_state.cube_state[curr] = (['White']*4 + [CENTER_COLORS[curr]] + ['White']*4)
                push_history(); st.rerun()
        with col_act2:
            st.toast(f"📍 Editing: {curr} Face")

        st.divider()
        st.markdown("#### 📂 Photo Assist")
        up_img = st.file_uploader("Upload face photo", type=['jpg', 'png', 'jpeg'], key=f"up_{curr}", label_visibility="collapsed")
        if up_img:
            st.image(up_img, caption="Reference Photo", use_container_width=True)
            if st.button(f"📸 Scan {curr} Face", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    d, db, ok = run_manual_grid_extract(up_img, CENTER_COLORS[curr])
                    if ok:
                        st.session_state.cube_state[curr] = d
                        push_history()
                        st.success("✨ Scanned! Advancing...")
                        # Auto-Advance Logic: Find next un-scanned face or just next in sequence
                        next_idx = (FACES.index(curr) + 1) % 6
                        st.session_state.programmatic_face = FACES[next_idx]
                        st.rerun()
                    else: st.error("❌ Scan Failed.")

    with c_sol:
        st.markdown("### 🚀 Solve Path")
        # --- Predictive Diagnostics ---
        all_stk = [s for f in FACES for s in st.session_state.cube_state[f]]
        inv = {c: all_stk.count(c) for c in HEX_COLORS.keys()}
        errors = [f"{COLOR_EMOJIS[c]} {inv[c]}/9" for c in inv if inv[c] != 9]
        
        is_ready = len(errors) == 0
        btn_label = "CALCULATE PATH" if is_ready else "⚠️ CHECK COLORS"
        
        if st.button(btn_label, type="primary" if is_ready else "secondary", use_container_width=True):
            if not is_ready:
                st.warning(f"Inventory Mismatch: {', '.join(errors)}")
            else:
                with st.spinner("Solving..."):
                    ok, msg = validate_cube_state(st.session_state.cube_state)
                    if ok: st.session_state.last_solution = solve_cube(st.session_state.cube_state)
                    else: st.error(f"❌ {msg}")
        
        if not is_ready:
            st.caption("Please ensure all colors have exactly 9 stickers.")
        
        if st.session_state.last_solution:
            sol = st.session_state.last_solution
            if sol == "!IMPOSSIBLE_STATE!":
                st.error("⚠️ IMPOSSIBLE: Check face orientations.")
            else:
                st.success(f"✅ Route: {sol}")
                st.markdown("#### 📺 3D Player Control")
                s1, s2, s3 = st.columns(3)
                if s1.button("🐢 0.5x", use_container_width=True): st.session_state.solve_speed = 0.5
                if s2.button("🏃 1.0x", use_container_width=True): st.session_state.solve_speed = 1.0
                if s3.button("🚀 2.0x", use_container_width=True): st.session_state.solve_speed = 2.0
                st.caption(f"Current Speed: {st.session_state.solve_speed}x")
                render_3d_player(sol)

elif app_mode == "⚙️ Tune Colors":
    st.title("⚙️ Color Calibration")
    st.info("💡 Upload a photo of a single face and click 'Auto-Calibrate' to sync sensors with your lighting.")
    
    c_cal = st.radio("Target Color:", list(HEX_COLORS.keys()), horizontal=True, format_func=lambda x: f"{COLOR_EMOJIS[x]} {x}")
    cal_buf = st.file_uploader("Upload Calibration Photo", type=['jpg', 'png', 'jpeg'], key="cal_up")
    
    if cal_buf:
        img_cal = cv2.imdecode(np.asarray(bytearray(cal_buf.read()), dtype=np.uint8), 1)
        if img_cal is not None:
            st.image(cv2.cvtColor(img_cal, cv2.COLOR_BGR2RGB), caption="Calibration Source", use_container_width=True)
            if st.button(f"🎯 Auto-Calibrate {c_cal} from Image Center", type="primary", use_container_width=True):
                h, w = img_cal.shape[:2]
                roi = img_cal[h//2-15:h//2+15, w//2-15:w//2+15]
                avg_bgr = np.median(roi, axis=(0,1)).astype(np.uint8)
                hsv_val = cv2.cvtColor(np.uint8([[[avg_bgr[0],avg_bgr[1],avg_bgr[0]]]]), cv2.COLOR_BGR2HSV)[0][0]
                st.session_state.custom_std_colors[c_cal] = [int(hsv_val[0]), int(hsv_val[1]), int(hsv_val[2])]
                with open(CALIB_FILE, 'w') as f: json.dump(st.session_state.custom_std_colors, f)
                st.success(f"✅ Calibrated {c_cal} to HSV: {hsv_val}")
                st.rerun()

# Footer info
st.markdown("---")
st.caption("Rubik's Solver Console: Precision logic with manual input. Verified by Kociemba Engine.")
