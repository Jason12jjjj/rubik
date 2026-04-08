import os, sys, json, streamlit as st, streamlit.components.v1 as components
from rubiks_core import validate_cube_state, solve_cube

# --- 1. SYSTEM CONFIG ---
st.set_page_config(page_title="Manual Rubik's Solver", page_icon="🧊", layout="wide")

# Constants
FACES = ['Up', 'Left', 'Front', 'Right', 'Back', 'Down']
HEX_COLORS = {'White':'#f8f9fa','Red':'#ff4b4b','Green':'#09ab3b','Yellow':'#ffeb3b','Orange':'#ffa500','Blue':'#1e88e5'}
COLOR_EMOJIS = {'White':'⬜','Red':'🟥','Green':'🟩','Yellow':'🟨','Orange':'🟧','Blue':'🟦'}
CENTER_COLORS = {'Up':'White','Left':'Orange','Front':'Green','Right':'Red','Back':'Blue','Down':'Yellow'}

# --- 2. SESSION STATE ---
if 'cube_state' not in st.session_state:
    st.session_state.cube_state = {f: (['White']*4 + [CENTER_COLORS[f]] + ['White']*4) for f in FACES}
if 'programmatic_face' not in st.session_state:
    st.session_state.programmatic_face = 'Front'
if 'last_solution' not in st.session_state:
    st.session_state.last_solution = None

# --- 3. UI COMPONENTS ---
def render_interactive_map(active_face):
    """Renders a 2D net of the cube for navigation and overview"""
    grid = {'Up':(0,1), 'Left':(1,0), 'Front':(1,1), 'Right':(1,2), 'Back':(1,3), 'Down':(2,1)}
    html = '<div style="display:grid;grid-template-columns:repeat(4,50px);gap:6px;justify-content:center;padding:10px;background:#1e1e1e;border-radius:10px;border:1px solid #444;">'
    for r in range(3):
        for c in range(4):
            f_k = next((f for f, p in grid.items() if p == (r, c)), None)
            if f_k:
                style = "border:2px solid #00e5ff; box-shadow:0 0 8px #00e5ff;" if f_k == active_face else "border:1px solid #444;"
                html += f'<div style="text-align:center; cursor:pointer; {style} border-radius:4px; padding:2px;">'
                html += f'<div style="font-size:10px; color:#aaa;">{f_k[0]}</div>'
                html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1px;">'
                for clr in st.session_state.cube_state[f_k]:
                    html += f'<div style="width:12px;height:12px;background:{HEX_COLORS[clr]};"></div>'
                html += '</div></div>'
            else: html += '<div></div>'
    html += '</div>'
    return html

def render_3d_player(solution):
    """Embeds the Twisty Player for solution visualization"""
    # Inverse algorithm for setup (to show the cube at its scrambled state first)
    def inverse_alg(s):
        moves = []
        for m in reversed(s.split()):
            if "'" in m: moves.append(m.replace("'", ""))
            elif "2" in m: moves.append(m)
            else: moves.append(m + "'")
        return " ".join(moves)
    
    html = f"""<div style="background:#000; border:1px solid #00e5ff; border-radius:15px; padding:20px; box-shadow:0 0 20px rgba(0,229,255,0.2);">
        <script src="https://cubing.net" type="module"></script>
        <twisty-player 
            experimental-setup-alg="{inverse_alg(solution)}" 
            alg="{solution}" 
            background="none" 
            control-panel="bottom-row"
            style="width:100%; height:400px;">
        </twisty-player>
    </div>"""
    components.html(html, height=450)

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🧩 Manual Console")
    st.markdown("### 🗺️ Context Map")
    st.markdown(render_interactive_map(st.session_state.programmatic_face), unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 📊 Inventory Check")
    all_stickers = [s for f in FACES for s in st.session_state.cube_state[f]]
    counts = {c: all_stickers.count(c) for c in HEX_COLORS.keys()}
    cols = st.columns(3)
    for i, (name, hex) in enumerate(HEX_COLORS.items()):
        cols[i%3].markdown(f"{COLOR_EMOJIS[name]} `{counts[name]}/9`")

    st.divider()
    if st.button("🗑️ Reset All Colors", use_container_width=True):
        st.session_state.cube_state = {f: (['White']*4 + [CENTER_COLORS[f]] + ['White']*4) for f in FACES}
        st.session_state.last_solution = None; st.rerun()

# --- 5. MAIN INTERFACE ---
curr = st.session_state.programmatic_face
st.title("🧊 Pro Rubik's Solver")

# Face Switching Header
h1, h2, h3 = st.columns([1,3,1])
with h1: 
    if st.button("⬅️ Prev", use_container_width=True):
        st.session_state.programmatic_face = FACES[(FACES.index(curr)-1)%6]; st.rerun()
with h2:
    st.markdown(f"<h2 style='text-align:center; color:#00e5ff;'>Editing: {curr} Face</h2>", unsafe_allow_html=True)
with h3:
    if st.button("Next ➡️", use_container_width=True):
        st.session_state.programmatic_face = FACES[(FACES.index(curr)+1)%6]; st.rerun()

st.info(f"💡 Click the buttons below to change sticker colors for the **{curr}** face.")

c_edit, c_sol = st.columns([1, 1])

with c_edit:
    # Manual Grid Entry
    def cycle_color(face_name, index):
        colors = list(HEX_COLORS.keys())
        current = st.session_state.cube_state[face_name][index]
        st.session_state.cube_state[face_name][index] = colors[(colors.index(current)+1) % len(colors)]
        st.session_state.last_solution = None # Reset solution on change

    for r in range(3):
        rows = st.columns(3)
        for c in range(3):
            idx = r*3 + c
            color_val = st.session_state.cube_state[curr][idx]
            if idx == 4: # Center is fixed
                rows[c].button(f"{COLOR_EMOJIS[color_val]}\nCtr", disabled=True, use_container_width=True)
            else:
                rows[c].button(f"{COLOR_EMOJIS[color_val]}\n{color_val}", key=f"btn_{curr}_{idx}", on_click=cycle_color, args=(curr, idx), use_container_width=True)

with c_sol:
    st.markdown("### 🚀 Restore Logic")
    if st.button("INITIATE SOLVE PROTOCOL", type="primary", use_container_width=True):
        with st.spinner("Analyzing cube matrix..."):
            ok, msg = validate_cube_state(st.session_state.cube_state)
            if ok:
                st.session_state.last_solution = solve_cube(st.session_state.cube_state)
            else:
                st.error(f"❌ {msg}")
    
    if st.session_state.last_solution:
        sol = st.session_state.last_solution
        if sol == "!IMPOSSIBLE_STATE!":
            st.error("⚠️ IMPOSSIBLE STATE: Please double-check the colors. Some corners or edges might be flipped.")
        else:
            st.success(f"✅ Logic Found: {sol}")
            render_3d_player(sol)

# Footer info
st.markdown("---")
st.caption("Manual Mode: No Camera. No AI Detection. Pure Human Logic. Verified by Kociemba.")
