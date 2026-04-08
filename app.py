import io, os, json, time, random, numpy as np, cv2, streamlit as st, streamlit.components.v1 as components
from streamlit_image_coordinates import streamlit_image_coordinates
from rubiks_core import validate_cube_state, solve_cube

# --- 1. SYSTEM CONFIG & CYBERPUNK STYLES ---
st.set_page_config(page_title="CyberScanner 2077", page_icon="📡", layout="wide")
FACES = ['Up', 'Left', 'Front', 'Right', 'Back', 'Down']
CENTER_COLORS = {'Up':'White','Left':'Orange','Front':'Green','Right':'Red','Back':'Blue','Down':'Yellow'}
HEX_COLORS = {'White':'#f8f9fa','Red':'#ff4b4b','Green':'#09ab3b','Yellow':'#ffeb3b','Orange':'#ffa500','Blue':'#1e88e5'}
COLOR_EMOJIS = {'White':'⬜','Red':'🟥','Green':'🟩','Yellow':'🟨','Orange':'🟧','Blue':'🟦'}
CALIB_FILE = "calibration_profile.json"

ORIENTATION_GUIDE = {
    'Up':("⬜ White","🟦 Blue","[TARGET]: WHITE_CORE | [REF]: BLUE_NORTH"),
    'Left':("🟧 Orange","⬜ White","[TARGET]: ORANGE_CORE | [REF]: WHITE_NORTH"),
    'Front':("🟩 Green","⬜ White","[TARGET]: GREEN_CORE | [REF]: WHITE_NORTH"),
    'Right':("🟥 Red","⬜ White","[TARGET]: RED_CORE | [REF]: WHITE_NORTH"),
    'Back':("🟦 Blue","⬜ White","[TARGET]: BLUE_CORE | [REF]: WHITE_NORTH"),
    'Down':("🟨 Yellow","🟩 Green","[TARGET]: YELLOW_CORE | [REF]: GREEN_NORTH")
}

def speak(text):
    components.html(f"<script>window.speechSynthesis.speak(new SpeechSynthesisUtterance('{text}'));</script>", height=0)

def play_sfx(sfx_type):
    # Pro tech sounds
    urls = {
        'lock': "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        'solve': "https://assets.mixkit.co/active_storage/sfx/1435/1435-preview.mp3"
    }
    components.html(f"""<audio autoplay><source src="{urls.get(sfx_type)}" type="audio/mpeg"></audio>""", height=0)

st.markdown("""
<style>
    /* Cyberpunk HUD & Glassmorphism */
    [data-testid="stCameraInput"] { 
        position: relative; border: 2px solid #00e5ff; border-radius: 15px; 
        box-shadow: 0 0 15px rgba(0,229,255,0.3); overflow: hidden;
    }
    
    /* Dynamic Scanning Line Overlay */
    [data-testid="stCameraInput"]::after {
        content: ""; position: absolute; top: 0; left: 50%; transform: translateX(-50%);
        width: 270px; height: 270px; z-index: 999; pointer-events: none;
        box-shadow: 0 0 0 1000px rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(0, 229, 255, 0.5);
        
        /* Scan-line animation */
        background: linear-gradient(to bottom, transparent 50%, rgba(0, 229, 255, 0.1) 51%, rgba(0, 229, 255, 0.4) 52%, rgba(0, 229, 255, 0.1) 53%, transparent 54%);
        background-size: 100% 200%;
        animation: scan-line 3s linear infinite;
    }
    @keyframes scan-line { 0% { background-position: 0 -100%; } 100% { background-position: 0 100%; } }
    
    /* Neon Corners */
    .pro-corners {
        position: absolute; top: -2px; left: 50%; transform: translateX(-50%);
        width: 274px; height: 274px; z-index: 1001; pointer-events: none;
    }
    .corner { 
        position: absolute; width: 40px; height: 40px; border: 4px solid #00e5ff; 
        filter: drop-shadow(0 0 8px #00e5ff); box-shadow: inset 0 0 10px rgba(0,229,255,0.2);
    }
    .tl { top: 0; left: 0; border-right: none; border-bottom: none; }
    .tr { top: 0; right: 0; border-left: none; border-bottom: none; }
    .bl { bottom: 0; left: 0; border-right: none; border-top: none; }
    .br { bottom: 0; right: 0; border-left: none; border-top: none; }
    
    /* Glass UI */
    .stButton>button {
        background: rgba(0, 229, 255, 0.1) !important; color: #00e5ff !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important; backdrop-filter: blur(5px);
        transition: all 0.3s;
    }
    .stButton>button:hover { background: rgba(0, 229, 255, 0.2) !important; box-shadow: 0 0 15px #00e5ff; }
</style>
""", unsafe_allow_html=True)

# --- 2. CORE ENGINES ---
def get_calibrated_colors():
    defaults = {'White':(0,30,220), 'Yellow':(30,160,200), 'Orange':(12,200,240), 'Red':(0,210,180), 'Green':(60,180,150), 'Blue':(110,180,160)}
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE,'r') as f:
            for k,v in json.load(f).items(): defaults[k] = tuple(v)
    return defaults

def run_cyber_extract(image_bytes, expected_center, scale_percent):
    img = cv2.imdecode(np.asarray(bytearray(image_bytes.read()), dtype=np.uint8), 1)
    if img is None: return None, None, False
    
    # Auto Brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 100: img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    
    h, w = img.shape[:2]
    gs = int(min(h, w) * (scale_percent / 100.0))
    sx, sy = (w - gs) // 2, (h - gs) // 2
    warped = cv2.resize(img[sy:sy+gs, sx:sx+gs], (300, 300))
    debug_warped = cv2.flip(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), 1) # Mirror preview
    
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
                dV = np.sqrt(0.5*(float(lab[0])-t_lab[0])**2 + 2.5*(float(lab[1])-t_lab[1])**2 + 2.5*(float(lab[2])-t_lab[2])**2)
                if dV < min_d: min_d, best_c = dV, name
            detected[r*3+c] = best_c
            cv2.circle(debug_warped, (300-fx, fy), 12, (255,255,255), 2)
            
    detected[4] = expected_center
    return detected, debug_warped, True

# --- 3. SESSION INITIALIZATION ---
init_defaults = {
    'cube_state': {f: (['White']*4 + [CENTER_COLORS[f]] + ['White']*4) for f in FACES},
    'processed_photos': {}, 'cube_size': 60, 'auto_advance': True, 'uploader_key_version': 0, 'programmatic_face': FACES[0], 'last_solution': None
}
for k, v in init_defaults.items():
    if k not in st.session_state: st.session_state[k] = v

def render_flat_map(state):
    layout = {'Up':(0,1), 'Left':(1,0), 'Front':(1,1), 'Right':(1,2), 'Back':(1,3), 'Down':(2,1)}
    html = '<div style="display:grid; grid-template-columns:repeat(4,45px); gap:6px; justify-content:center; background:rgba(0,0,0,0.5); padding:10px; border-radius:12px; border:1px solid rgba(0,229,255,0.2);">'
    for r in range(3):
        for c in range(4):
            f_k = next((f for f, p in layout.items() if p == (r, c)), None)
            if f_k:
                grid_html = '<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1px; border:1px solid rgba(0,229,255,0.3);">'
                for clr in state[f_k]: html += f'<div style="width:12px; height:12px; background:{HEX_COLORS[clr]};"></div>'
                html += grid_html + '</div>'
            else: html += '<div></div>'
    return html + '</div>'

# --- 4. SIDEBAR DASHBOARD ---
with st.sidebar:
    st.title("📡 Cyber Terminal")
    app_mode = st.radio("System Access:", ["📸 Scanner", "⚙️ Calibration"], label_visibility="collapsed")
    
    st.markdown("#### 🗺️ 2D MAPPING")
    components.html(render_flat_map(st.session_state.cube_state), height=180)
    
    st.markdown("#### 📊 TELEMETRY")
    all_stks = [s for f in FACES for s in st.session_state.cube_state[f]]
    counts = {c: all_stks.count(c) for c in HEX_COLORS.keys()}
    cols = st.columns(3)
    for i, color in enumerate(list(HEX_COLORS.keys())):
        with cols[i%3]: st.markdown(f"{COLOR_EMOJIS[color]} `{counts[color]}/9`")
    
    st.divider()
    if st.button("🗑️ PURGE DATA", use_container_width=True):
        st.session_state.processed_photos = {}; st.rerun()

# --- 5. MAIN SCANNER ---
if app_mode == "📸 Scanner":
    curr = st.session_state.programmatic_face
    progress = len(st.session_state.processed_photos) / 6.0
    st.progress(progress, text=f"SCANNING STATUS: {int(progress*100)}%")
    
    col_nav1, col_nav2, col_nav3 = st.columns([1,3,1])
    with col_nav1: 
        if st.button("⬅️"): st.session_state.programmatic_face = FACES[(FACES.index(curr)-1)%6]; st.rerun()
    with col_nav2: st.markdown(f"<h3 style='text-align:center; color:#00e5ff; text-shadow:0 0 10px #00e5ff;'>{curr.upper()} PROTOCOL</h3>", unsafe_allow_html=True)
    with col_nav3: 
        if st.button("➡️"): st.session_state.programmatic_face = FACES[(FACES.index(curr)+1)%6]; st.rerun()

    c_cam, c_log = st.columns([1.5, 1])
    with c_cam:
        guide = ORIENTATION_GUIDE[curr]
        st.info(f"🧭 {guide[2]}")
        
        # PRO HUD
        st.markdown(f'''<div style="position:relative;">
            <div class="pro-corners"><div class="corner tl"></div><div class="corner tr"></div><div class="corner bl"></div><div class="corner br"></div></div>
            <div style="position:absolute; top:35px; width:100%; text-align:center; color:#00e5ff; font-weight:bold; z-index:1000; pointer-events:none;">
                {guide[0].upper()} CORE DETECTION<br><span style="font-size:10px; opacity:0.6;">ANALYZING MATRIX...</span>
            </div>
        ''', unsafe_allow_html=True)
        buf = st.camera_input("Scanner", key=f"c_{st.session_state.uploader_key_version}", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        if buf:
            d, db, ok = run_cyber_extract(buf, CENTER_COLORS[curr], st.session_state.cube_size)
            if ok:
                st.session_state.cube_state[curr] = d
                st.image(db, caption="CYBERNETIC PREVIEW (MIRRORED)", use_container_width=True)
                if d[4] == CENTER_COLORS[curr]:
                    st.success("🎯 SIGNAL LOCKED")
                    play_sfx('lock')
                    if st.button("📥 ENGAGE SAVE", type="primary", use_container_width=True):
                        st.session_state.processed_photos[curr] = True
                        un = [f for f in FACES if f not in st.session_state.processed_photos]
                        if un: st.session_state.programmatic_face = un[0]
                        else: st.balloons(); speak("Analysis complete. System ready to solve.")
                        st.rerun()
                else: st.error("⚠️ SIGNAL MISMATCH: CORE COLOR ERROR")

    with c_log:
        st.markdown("#### 📟 DATA LOGS")
        st.code(f"""
[LOG]: SCANNING {curr.upper()}...
[RSSI]: -{random.randint(40,60)} dBm
[COLOR]: {d[4] if buf else 'WAITING'}
[STATUS]: {'LOCKED' if buf else 'SCANNING'}
[LATENCY]: {random.uniform(5.5, 12.2):.2f}ms
[SECURE]: ENGAGED
        """, language="ini")
        
        st.markdown("#### 🕹️ MANUAL OVERRIDE")
        for r in range(3):
            ecols = st.columns(3)
            for c in range(3):
                idx = r*3 + c
                v = st.session_state.cube_state[curr][idx]
                if ecols[c].button(f"{COLOR_EMOJIS[v]}", key=f"ed_{curr}_{idx}", use_container_width=True):
                    seq = list(COLOR_EMOJIS.keys())
                    st.session_state.cube_state[curr][idx] = seq[(seq.index(v)+1)%6]; st.rerun()

    st.divider()
    if st.button("🚀 INITIATE SOLVE PROTOCOL", type="primary", use_container_width=True, disabled=(len(st.session_state.processed_photos)<6)):
        with st.spinner("Deciphering rubik's matrix..."):
            ok, msg = validate_cube_state(st.session_state.cube_state)
            if ok: st.session_state.last_solution = solve_cube(st.session_state.cube_state); play_sfx('solve')
            else: st.error(msg)

    if st.session_state.last_solution:
        sol = st.session_state.last_solution
        st.markdown(f"""
        <div style="background: rgba(0,0,0,0.8); border: 2px solid #00e5ff; border-radius: 15px; padding: 20px; box-shadow: 0 0 25px rgba(0,229,255,0.4); text-align:center;">
            <h3 style="color:#00e5ff; font-family:'Courier New'; margin-bottom:10px;">OPTIMAL SOLUTION GENERATED</h3>
            <p style="color:#fff; font-size:14px; opacity:0.8;">{sol}</p>
            <script src="https://cdn.cubing.net/v0/js/cubing/twisty" type="module"></script>
            <twisty-player alg="{sol}" control-panel="bottom-row" background="none" style="width:100%; height:320px;"></twisty-player>
        </div>
        """, unsafe_allow_html=True)

elif app_mode == "⚙️ Calibration":
    st.title("⚙️ Calibration Terminal")
    target_c = st.selectbox("Assign Click to Spectrum:", list(COLOR_EMOJIS.keys()))
    buf = st.camera_input("Sample")
    if buf:
        img_bgr = cv2.imdecode(np.asarray(bytearray(buf.read()), dtype=np.uint8), 1)
        res = streamlit_image_coordinates(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if res:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[res['y'], res['x']].tolist()
            if st.button(f"UPLOAD SPECTRUM for {target_c}"):
                cal = get_calibrated_colors(); cal[target_c] = hsv
                with open(CALIB_FILE,'w') as f: json.dump(cal, f); st.rerun()
