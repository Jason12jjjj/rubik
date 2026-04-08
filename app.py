import io, os, json, numpy as np, cv2, streamlit as st, streamlit.components.v1 as components
from streamlit_image_coordinates import streamlit_image_coordinates
from rubiks_core import validate_cube_state, solve_cube

# --- 1. CONFIG & STYLES ---
st.set_page_config(page_title="Refined Rubik's Solver", page_icon="🧊", layout="wide")
FACES = ['Up', 'Left', 'Front', 'Right', 'Back', 'Down']
CENTER_COLORS = {'Up':'White','Left':'Orange','Front':'Green','Right':'Red','Back':'Blue','Down':'Yellow'}
HEX_COLORS = {'White':'#f8f9fa','Red':'#ff4b4b','Green':'#09ab3b','Yellow':'#ffeb3b','Orange':'#ffa500','Blue':'#1e88e5'}
COLOR_EMOJIS = {'White':'⬜','Red':'🟥','Green':'🟩','Yellow':'🟨','Orange':'🟧','Blue':'🟦'}
CALIB_FILE = "calibration_profile.json"

ORIENTATION_GUIDE = {
    'Up':("⬜ White","🟦 Blue"), 'Left':("辨 Orange","⬜ White"), 'Front':("🟩 Green","⬜ White"),
    'Right':("🟥 Red","⬜ White"), 'Back':("🟦 Blue","⬜ White"), 'Down':("🟨 Yellow","🟩 Green")
}

st.markdown("""
<style>
    [data-testid="stCameraInput"] { position: relative; }
    [data-testid="stCameraInput"]::after {
        content: ""; position: absolute; top: 25px; left: 50%; transform: translateX(-50%);
        width: 260px; height: 260px; border: 4px solid #00e5ff; border-radius: 8px; z-index: 999;
        pointer-events: none; box-shadow: 0 0 0 1000px rgba(0,0,0,0.4);
        background-image: linear-gradient(to right, rgba(0,229,255,0.3) 1px, transparent 1px), linear-gradient(to bottom, rgba(0,229,255,0.3) 1px, transparent 1px);
        background-size: 86.6px 86.6px; animation: pulse 2s infinite;
    }
    [data-testid="stCameraInput"]::before {
        content: "PLACE FACE IN GRID"; position: absolute; top: -10px; width: 100%; text-align: center;
        color: #00e5ff; font-weight: bold; z-index: 1000; text-shadow: 0 0 10px #000;
    }
    @keyframes pulse { 0%,100% { opacity: 0.6; } 50% { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

# --- 2. LOGIC & SAMPLING ---
def get_calibrated_colors():
    defaults = {'White':(0,30,220), 'Yellow':(30,160,200), 'Orange':(12,200,240), 'Red':(0,210,180), 'Green':(60,180,150), 'Blue':(110,180,160)}
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE,'r') as f:
            for k,v in json.load(f).items(): defaults[k] = tuple(v)
    return defaults

def run_fixed_grid_extract(image_bytes, expected_center, scale_percent):
    img = cv2.imdecode(np.asarray(bytearray(image_bytes.read()), dtype=np.uint8), 1)
    if img is None: return None, None, False
    h, w = img.shape[:2]
    gs = int(min(h, w) * (scale_percent / 100.0))
    sx, sy = (w - gs) // 2, (h - gs) // 2
    warped = cv2.resize(img[sy:sy+gs, sx:sx+gs], (300, 300))
    debug_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    std_colors, detected = get_calibrated_colors(), ['White']*9
    sat_w = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)[:,:,1]
    
    for r in range(3):
        for c in range(3):
            tx, ty = int((c+0.5)*100), int((r+0.5)*100)
            y1, y2, x1, x2 = max(0,ty-35), min(300,ty+35), max(0,tx-35), min(300,tx+35)
            moms = cv2.moments(sat_w[y1:y2, x1:x2])
            fx, fy = (x1 + int(moms["m10"]/moms["m00"]), y1 + int(moms["m01"]/moms["m00"])) if moms["m00"]>50 else (tx, ty)
            
            roi = warped[max(0,fy-8):min(300,fy+8), max(0,fx-8):min(300,fx+8)]
            bgr = np.median(roi[2:-2, 2:-2], axis=(0,1)).astype(np.uint8) if roi.size>0 else [0,0,0]
            lab = cv2.cvtColor(np.uint8([[[bgr[0],bgr[1],bgr[2]]]]), cv2.COLOR_BGR2LAB)[0][0]
            
            min_d, best_c = 999.0, 'White'
            for name, (hs,ss,vs) in std_colors.items():
                t_lab = cv2.cvtColor(cv2.cvtColor(np.uint8([[[hs,ss,vs]]]), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2LAB)[0][0]
                dV = np.sqrt(0.1*(float(lab[0])-t_lab[0])**2 + 2.4*(float(lab[1])-t_lab[1])**2 + 2.4*(float(lab[2])-t_lab[2])**2)
                if dV < min_d: min_d, best_c = dV, name
            detected[r*3+c] = best_c
            cv2.circle(debug_warped, (fx, fy), 12, (255,255,255), 2)
    detected[4] = expected_center
    return detected, debug_warped, True

# --- 3. SESSION INITIALIZATION ---
init_defaults = {
    'cube_state': {f: (['White']*4 + [CENTER_COLORS[f]] + ['White']*4) for f in FACES},
    'processed_photos': {}, 'cube_size': 60, 'auto_advance': True, 'uploader_key_version': 0, 'programmatic_face': FACES[0], 'last_solution': None
}
for k, v in init_defaults.items():
    if k not in st.session_state: st.session_state[k] = v

if "face_click" in st.query_params:
    st.session_state.programmatic_face = st.query_params["face_click"]
    st.query_params.clear()

# --- 4. SIDEBAR & HUD ---
with st.sidebar:
    st.title("🧊 Controls")
    app_mode = st.radio("Mode:", ["📸 Scan & Solve", "⚙️ Calibration"], label_visibility="collapsed")
    if app_mode == "📸 Scan & Solve":
        st.session_state.auto_advance = st.checkbox("⏩ Auto-advance", value=st.session_state.auto_advance)
        st.markdown("### 📊 Status")
        all_stks = [s for f in FACES for s in st.session_state.cube_state[f]]
        from collections import Counter
        counts = Counter(all_stks)
        cols = st.columns(3)
        for i, color in enumerate(list(COLOR_EMOJIS.keys())):
            cnt = counts.get(color,0)
            with cols[i%3]: st.markdown(f"{COLOR_EMOJIS[color]} `{cnt}/9` {'✅' if cnt==9 else '⏳'}")
        if st.button("🗑️ Reset All", use_container_width=True):
            st.session_state.processed_photos = {}; st.rerun()

# --- 5. MAIN SCANNER ---
if app_mode == "📸 Scan & Solve":
    curr = st.session_state.programmatic_face
    st.title(f"{COLOR_EMOJIS[CENTER_COLORS[curr]]} {curr} Face")
    
    col_nav1, col_nav2, col_nav3 = st.columns([1,3,1])
    with col_nav1: 
        if st.button("⬅️"): st.session_state.programmatic_face = FACES[(FACES.index(curr)-1)%6]; st.rerun()
    with col_nav3: 
        if st.button("➡️"): st.session_state.programmatic_face = FACES[(FACES.index(curr)+1)%6]; st.rerun()

    c_cam, c_edit = st.columns([1.2, 1])
    with c_cam:
        guide = ORIENTATION_GUIDE[curr]
        st.info(f"🧭 HOLD: {guide[0]} facing you, {guide[1]} on TOP.")
        buf = st.camera_input("P", key=f"c_{st.session_state.uploader_key_version}", label_visibility="collapsed")
        if buf:
            d, db, ok = run_fixed_grid_extract(buf, CENTER_COLORS[curr], st.session_state.cube_size)
            if ok:
                st.session_state.cube_state[curr] = d
                st.image(db, caption="Sampled Grid", use_container_width=True)
                if st.button("✅ Confirm & Save", type="primary", use_container_width=True):
                    st.session_state.processed_photos[curr] = True
                    st.session_state[f"cached_{curr}"] = db
                    if st.session_state.auto_advance:
                        un = [f for f in FACES if f not in st.session_state.processed_photos]
                        if un: st.session_state.programmatic_face = un[0]
                    st.rerun()

    with c_edit:
        st.markdown("### 🖱️ Edit Colors")
        for r in range(3):
            ecols = st.columns(3)
            for c in range(3):
                idx = r*3 + c
                if idx == 4: ecols[c].button(f"{COLOR_EMOJIS[CENTER_COLORS[curr]]}\nCtr", disabled=True, use_container_width=True)
                else:
                    v = st.session_state.cube_state[curr][idx]
                    if ecols[c].button(f"{COLOR_EMOJIS[v]}\n{v}", key=f"e_{curr}_{idx}", use_container_width=True):
                        seq = list(COLOR_EMOJIS.keys())
                        st.session_state.cube_state[curr][idx] = seq[(seq.index(v)+1)%6]
                        st.rerun()

    st.divider()
    if st.button("🚀 SOLVE", type="primary", use_container_width=True):
        ok, msg = validate_cube_state(st.session_state.cube_state)
        if ok: st.session_state.last_solution = solve_cube(st.session_state.cube_state)
        else: st.error(msg)

    if st.session_state.last_solution:
        sol = st.session_state.last_solution
        st.success(f"✅ Moves: {sol}")
        html = f"""<script src="https://cdn.cubing.net/v0/js/cubing/twisty" type="module"></script>
        <twisty-player alg="{sol}" control-panel="bottom-row" background="none" style="width:100%; height:300px;"></twisty-player>"""
        components.html(html, height=320)

elif app_mode == "⚙️ Calibration":
    st.title("⚙️ Calibration")
    c_cal = st.radio("Target:", list(COLOR_EMOJIS.keys()), horizontal=True)
    buf = st.camera_input("Sample")
    if buf:
        val_xy = streamlit_image_coordinates(cv2.cvtColor(cv2.imdecode(np.asarray(bytearray(buf.read()), dtype=np.uint8), 1), cv2.COLOR_BGR2RGB))
        if val_xy:
            st.success(f"Sampled at {val_xy}. Save logic ready.")
