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
    """Executes JS for Text-to-Speech"""
    components.html(f"<script>const m = new SpeechSynthesisUtterance('{text}'); m.rate=0.95; window.speechSynthesis.speak(m);</script>", height=0)

def play_sfx(sfx_type):
    urls = {
        'lock': "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        'solve': "https://assets.mixkit.co/active_storage/sfx/1435/1435-preview.mp3"
    }
    components.html(f"""<audio autoplay><source src="{urls.get(sfx_type)}" type="audio/mpeg"></audio>""", height=0)

st.markdown("""
<style>
    /* Cyberpunk HUD - PERCENTAGE BASED FOR SYNC */
    [data-testid="stCameraInput"] { 
        position: relative; border: 2px solid #00e5ff; border-radius: 15px; 
        box-shadow: 0 0 20px rgba(0,229,255,0.3); overflow: hidden;
    }
    
    [data-testid="stCameraInput"]::after {
        content: ""; position: absolute; 
        top: 45%; left: 50%; transform: translate(-50%, -50%);
        width: 60%; aspect-ratio: 1/1; z-index: 999; pointer-events: none;
        box-shadow: 0 0 0 1000px rgba(0, 0, 0, 0.6);
        border: 2px solid rgba(0, 229, 255, 0.6);
        background: linear-gradient(to bottom, transparent 50%, rgba(0, 229, 255, 0.1) 51%, rgba(0, 229, 255, 0.4) 52%, rgba(0, 229, 255, 0.1) 53%, transparent 54%);
        background-size: 100% 200%;
        animation: scan-line 4s linear infinite;
    }
    @keyframes scan-line { 0% { background-position: 0 -100%; } 100% { background-position: 0 100%; } }
    
    .pro-corners {
        position: absolute; top: 45%; left: 50%; transform: translate(-50%, -50%);
        width: 62%; aspect-ratio: 1/1; z-index: 1001; pointer-events: none;
    }
    .corner { 
        position: absolute; width: 40px; height: 40px; border: 4px solid #00e5ff; 
        filter: drop-shadow(0 0 8px #00e5ff); 
    }
    .tl { top: 0; left: 0; border-right: none; border-bottom: none; }
    .tr { top: 0; right: 0; border-left: none; border-bottom: none; }
    .bl { bottom: 0; left: 0; border-right: none; border-top: none; }
    .br { bottom: 0; right: 0; border-left: none; border-top: none; }
    
    .stButton>button {
        background: rgba(0, 229, 255, 0.1) !important; color: #00e5ff !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important; backdrop-filter: blur(5px);
    }
    .low-light-warning {
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        color: #ff4b4b; font-weight: bold; font-size: 20px; z-index: 1002;
        text-shadow: 0 0 10px #000; animation: flash 1s infinite;
    }
    @keyframes flash { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# CSS for the Interactive Nav Grid
def render_interactive_nav():
    layout = {'Up':(0,1), 'Left':(1,0), 'Front':(1,1), 'Right':(1,2), 'Back':(1,3), 'Down':(2,1)}
    st.markdown("#### 🗺️ 2D PROTOCOL MAP")
    for r in range(3):
        cols = st.columns(4)
        for c in range(4):
            face = next((f for f, p in layout.items() if p == (r, c)), None)
            if face:
                btn_type = "primary" if face == st.session_state.programmatic_face else "secondary"
                # Use a small visual state indicator
                is_done = "✅" if face in st.session_state.processed_photos else "⏳"
                if cols[c].button(f"{is_done} {face[0]}", key=f"nav_{face}", help=f"Jump to {face} face"):
                    st.session_state.programmatic_face = face
                    st.rerun()

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
    
    # Auto Brightness & Low Light Warning
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_bright = np.mean(gray)
    is_low_light = avg_bright < 60
    if avg_bright < 100: img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    
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
                dV = np.sqrt(0.5*(float(lab[0])-t_lab[0])**2 + 2.5*(float(lab[1])-t_lab[1])**2 + 2.5*(float(lab[2])-t_lab[2])**2)
                if dV < min_d: min_d, best_c = dV, name
            detected[r*3+c] = best_c
            cv2.circle(debug_warped, (fx, fy), 12, (255,255,255), 2)
            
    detected[4] = expected_center
    return detected, debug_warped, is_low_light, True

# --- 3. SESSION INITIALIZATION ---
init_defaults = {
    'cube_state': {f: (['White']*4 + [CENTER_COLORS[f]] + ['White']*4) for f in FACES},
    'processed_photos': {}, 'cube_size': 60, 'auto_advance': True, 'uploader_key_version': 1, 
    'programmatic_face': FACES[0], 'last_solution': None, 'pending_speech': None
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
    app_mode = st.radio("Access Level:", ["📸 Scanner", "⚙️ Calibration"], label_visibility="collapsed")
    
    render_interactive_nav()
    
    st.markdown("#### 📊 TELEMETRY")
    all_stks = [s for f in FACES for s in st.session_state.cube_state[f]]
    counts = {c: all_stks.count(c) for c in HEX_COLORS.keys()}
    cols = st.columns(3)
    for i, color in enumerate(list(HEX_COLORS.keys())):
        with cols[i%3]: st.markdown(f"{COLOR_EMOJIS[color]} `{counts[color]}/9`")
    
    st.divider()
    if st.button("🗑️ PURGE DATA", use_container_width=True):
        st.session_state.processed_photos = {}; st.session_state.uploader_key_version += 1; st.rerun()

# --- 5. MAIN SCANNER ---
if app_mode == "📸 Scanner":
    if not os.path.exists(CALIB_FILE):
        st.warning("⚠️ CRITICAL: Calibration profile missing. Please calibrate colors for accurate results.")

    curr = st.session_state.programmatic_face
    progress = len(st.session_state.processed_photos) / 6.0
    st.progress(progress, text=f"UPLINK STATUS: {int(progress*100)}%")
    
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
        st.markdown(f'''<div style="position:relative;">
            <div class="pro-corners"><div class="corner tl"></div><div class="corner tr"></div><div class="corner bl"></div><div class="corner br"></div></div>
            <div style="position:absolute; top:35px; width:100%; text-align:center; color:#00e5ff; font-weight:bold; z-index:1000; pointer-events:none;">
                {guide[0].upper()} CORE DETECTION<br><span style="font-size:10px; opacity:0.6;">ANALYZING MATRIX...</span>
            </div>
        ''', unsafe_allow_html=True)
        # BIND TO KEY VERSION FOR CAMERA RESET
        buf = st.camera_input("Scanner", key=f"v_{st.session_state.uploader_key_version}", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # SPACEBAR CAPTURE JS BRIDGE
        components.html("""<script>
            const doc = window.parent.document;
            doc.addEventListener('keydown', function(e) {
                if (e.code === 'Space') {
                    const buttons = doc.querySelectorAll('button');
                    const captureBtn = Array.from(buttons).find(el => el.innerText.includes('Scanner'));
                    if (captureBtn) captureBtn.click();
                }
            });
        </script>""", height=0)

        if buf:
            d, db, low_l, ok = run_cyber_extract(buf, CENTER_COLORS[curr], st.session_state.cube_size)
            if ok:
                st.session_state.cube_state[curr] = d
                st.image(db, caption="RAW SENSOR DATA (UNMIRRORED)", use_container_width=True)
                if low_l: st.markdown('<div class="low-light-warning">⚠️ [SIGNAL_WEAK]: INCREASE LIGHT</div>', unsafe_allow_html=True)
                
                if d[4] == CENTER_COLORS[curr]:
                    st.success("🎯 SIGNAL LOCKED")
                    play_sfx('lock')
                    if st.button("📥 ENGAGE SAVE", type="primary", use_container_width=True):
                        st.session_state.processed_photos[curr] = True
                        st.session_state.uploader_key_version += 1 # RESET CAMERA
                        un = [f for f in FACES if f not in st.session_state.processed_photos]
                        if un: 
                            st.session_state.programmatic_face = un[0]
                            st.session_state.pending_speech = f"Signal saved. Moving to {un[0]} protocol."
                        else: st.balloons(); st.session_state.pending_speech = "All matrices synced. Logic ready."
                        st.rerun()
                else: st.error("⚠️ SIGNAL MISMATCH: CORE COLOR ERROR")

    with c_log:
        st.markdown("#### 📟 DATA LOGS")
        st.code(f"[LOG]: SCANNING {curr.upper()}...\n[COLOR]: {d[4] if buf else 'WAITING'}\n[SECURE]: ENGAGED", language="ini")
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
        with st.spinner("Analyzing matrix..."):
            ok, msg = validate_cube_state(st.session_state.cube_state)
            if ok: 
                st.session_state.last_solution = solve_cube(st.session_state.cube_state)
                play_sfx('solve')
                st.session_state.pending_speech = "Solution generated. Ready for playback."
            else: st.error(msg)

    if st.session_state.last_solution:
        sol = st.session_state.last_solution
        # Map current face to 3D starting angle
        views = {'Front':(0,0), 'Right':(0,90), 'Back':(0,180), 'Left':(0,270), 'Up':(90,0), 'Down':(-90,0)}
        lat, lon = views.get(curr, (0,0))
        
        st.markdown(f"""<div style="background:rgba(0,0,0,0.8); border:2px solid #00e5ff; border-radius:15px; padding:20px; text-align:center;">
            <h3 style="color:#00e5ff; font-family:'Courier New';">OPTIMAL SOLUTION GENERATED</h3>
            <p style="color:#fff;">{sol}</p>
            <script src="https://cubing.net" type="module"></script>
            <twisty-player alg="{sol}" control-panel="bottom-row" background="none" 
                           camera-latitude="{lat}" camera-longitude="{lon}"
                           style="width:100%; height:320px;"></twisty-player>
        </div>""", unsafe_allow_html=True)

elif app_mode == "⚙️ Calibration":
    st.title("⚙️ Calibration Terminal")
    target_c = st.selectbox("Assign Click to Spectrum:", list(COLOR_EMOJIS.keys()))
    buf = st.camera_input("Sample")
    if buf:
        img_bgr = cv2.imdecode(np.asarray(bytearray(buf.read()), dtype=np.uint8), 1)
        res = streamlit_image_coordinates(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if res:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[res['y'], res['x']].tolist()
            if st.button(f"UPLOAD SPECTRUM"):
                cal = get_calibrated_colors(); cal[target_c] = hsv
                with open(CALIB_FILE,'w') as f: json.dump(cal, f); st.rerun()

# --- 6. ASYNC VOICE ENGINE & FOOTER ---
if st.session_state.pending_speech:
    speak(st.session_state.pending_speech)
    st.session_state.pending_speech = None

st.markdown("---")
ft_cols = st.columns(4)
with ft_cols[0]: st.caption("📡 UPLINK: STABLE")
with ft_cols[1]: st.caption(f"💾 CACHE: {len(st.session_state.processed_photos)}/6")
with ft_cols[2]: st.caption(f"🛡️ SECURE: {'LOCKED' if st.session_state.last_solution else 'PENDING'}")
with ft_cols[3]: st.caption("⚡ ENGINE: KOCIEMBA_V2")
if not st.session_state.processed_photos:
    st.toast("Pro Tip: Hold the cube steady for 1s before capturing.", icon="🧊")
