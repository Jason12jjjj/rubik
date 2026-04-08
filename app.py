# ==============================================================================
# RUBIK'S CUBE SOLVER - PREMIUM INTERACTIVE VERSION
# ==============================================================================
import io
import os
import json
import numpy as np
import cv2
import streamlit as st
import streamlit.components.v1 as components
from streamlit_image_coordinates import streamlit_image_coordinates
from rubiks_core import validate_cube_state, solve_cube

# --- 1. System Configuration & State Management ---
st.set_page_config(page_title="AI Rubik's Solver", page_icon="🧊", layout="wide")

# Handle Interactive Map Clicks via Query Parameters
if "face_click" in st.query_params:
    clicked_face = st.query_params["face_click"]
    st.session_state.programmatic_face = clicked_face
    # Clear the parameter to avoid loop-reruns
    st.query_params.clear()

AVAILABLE_COLORS = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
FACES = ['Up', 'Left', 'Front', 'Right', 'Back', 'Down']
COLOR_EMOJIS = {'White': '⬜', 'Red': '🟥', 'Green': '🟩', 'Yellow': '🟨', 'Orange': '🟧', 'Blue': '🟦'}
HEX_COLORS = {'White': '#f8f9fa', 'Red': '#ff4b4b', 'Green': '#09ab3b', 'Yellow': '#ffeb3b', 'Orange': '#ffa500', 'Blue': '#1e88e5'}
CENTER_COLORS = {'Up': 'White', 'Left': 'Orange', 'Front': 'Green', 'Right': 'Red', 'Back': 'Blue', 'Down': 'Yellow'}
COLOR_TO_FACE_NAME = {v: k for k, v in CENTER_COLORS.items()}
CALIB_FILE = "calibration_profile.json"

ORIENTATION_GUIDE = {
    'Up':    "Point the camera at the **⬜ White** face. Rotate the cube so the adjacent **🟦 Blue** face is on the **TOP edge ⬆️** of the camera screen.",
    'Left':  "Point the camera at the **🟧 Orange** face. Rotate the cube so the adjacent **⬜ White** face is on the **TOP edge ⬆️** of the camera screen.",
    'Front': "Point the camera at the **🟩 Green** face. Rotate the cube so the adjacent **⬜ White** face is on the **TOP edge ⬆️** of the camera screen.",
    'Right': "Point the camera at the **🟥 Red** face. Rotate the cube so the adjacent **⬜ White** face is on the **TOP edge ⬆️** of the camera screen.",
    'Back':  "Point the camera at the **🟦 Blue** face. Rotate the cube so the adjacent **⬜ White** face is on the **TOP edge ⬆️** of the camera screen.",
    'Down':  "Point the camera at the **🟨 Yellow** face. Rotate the cube so the adjacent **🟩 Green** face is on the **TOP edge ⬆️** of the camera screen."
}

# --- Initialize Session State ---
if 'cube_state' not in st.session_state:
    st.session_state.cube_state = {f: (['White'] * 4 + [CENTER_COLORS[f]] + ['White'] * 4) for f in FACES}

if 'processed_photos' not in st.session_state:
    st.session_state.processed_photos = {}

if 'custom_std_colors' not in st.session_state:
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, 'r') as f:
                st.session_state.custom_std_colors = json.load(f)
        except Exception:
            st.session_state.custom_std_colors = {}
    else:
        st.session_state.custom_std_colors = {}

if 'cube_size' not in st.session_state:
    st.session_state.cube_size = 50

if 'shared_face_images' not in st.session_state:
    st.session_state.shared_face_images = {}

if 'auto_detect' not in st.session_state:
    st.session_state.auto_detect = True

if 'auto_advance' not in st.session_state:
    st.session_state.auto_advance = True

if 'show_guide' not in st.session_state:
    st.session_state.show_guide = True

if 'last_solution' not in st.session_state:
    st.session_state.last_solution = None

if 'uploader_key_version' not in st.session_state:
    st.session_state.uploader_key_version = 0

if 'programmatic_face' not in st.session_state:
    st.session_state.programmatic_face = FACES[0]

# --- Global Styles ---
st.markdown("""
<style>
    [data-testid="stCameraInput"] { position: relative; }
    [data-testid="stCameraInput"]::after { display: none !important; }
    
    /* Camera Guide Overlay */
    .camera-container {
        position: relative;
        display: flex;
        justify-content: center;
        align-items: flex-start; /* Align to top to match video feed */
        width: 100%;
        max-width: 500px;
        margin: 0 auto;
    }
    .camera-guide {
        position: absolute;
        top: 130px; /* Precise offset for Streamlit's internal video frame */
        width: 280px;
        height: 280px;
        border: 2px dashed rgba(255, 255, 255, 0.4);
        border-radius: 12px;
        pointer-events: none;
        z-index: 10;
        box-shadow: 0 0 0 1000px rgba(0, 0, 0, 0.15);
    }
    .camera-guide::before {
        content: "CENTER CUBE HERE";
        position: absolute;
        top: -30px;
        left: 50%;
        transform: translateX(-50%);
        color: white;
        font-size: 12px;
        font-weight: bold;
        white-space: nowrap;
        text-shadow: 1px 1px 2px black;
    }
</style>
""", unsafe_allow_html=True)

# ── ADVANCED COMPUTER VISION (WARP PERSPECTIVE VERSION) ──────────────────────
def auto_detect_cube_face(image_bytes, expected_center, show_diag=False):
    # --- 1. PRE-PHASE: DECODE & SCREENING ---
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: 
        return None, None, "No Data", {}, None, None, ["Fail: No image"], 0, 0
    
    pad = 40
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    h_p, w_p = img_padded.shape[:2]
    
    # Init variables to avoid UnboundLocalError
    trace = []
    diag_imgs = {}
    best_cluster = []
    med_w = 40.0
    max_score = 0.0
    best_reg_score = 0.0
    cluster_coverage = 0.0
    pass_info = "Searching..."
    
    # Downscale for processing speed
    work_h = 650
    work_w = int(w_p * (work_h / h_p))
    work_img = cv2.resize(img_padded, (work_w, work_h))
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    
    # Environment Sensing
    bright = np.mean(gray)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    trace.append(f"📸 Env: Brightness={bright:.1f}, Sharpness={sharp:.1f}")
    
    if bright < 65 or bright > 220:
        trace.append("⚡ Extreme light detected. Applying CLAHE Contrast Boost...")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 2. DETECTION PHASE: SEARCHING FOR STICKERS ---
    candidates = []
    all_raw = [] # For Diagnostic Map
    
    def process_contours(clist, tag=""):
        found = []
        for c in clist:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            hull = cv2.convexHull(c)
            h_area = cv2.contourArea(hull)
            solidity = area / h_area if h_area > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0
            
            # BIO-GUARD
            roi_mini = work_img[max(0,y):min(work_h,y+h), max(0,x):min(work_w,x+w)]
            is_skin, is_vibrant = False, True
            if roi_mini.size > 0:
                bgr_m = np.mean(roi_mini, axis=(0,1))
                hsv_m = cv2.cvtColor(np.uint8([[bgr_m]]), cv2.COLOR_BGR2HSV)[0][0]
                is_skin = (bgr_m[2] > bgr_m[1] > bgr_m[0]) and (hsv_m[1] < 110)
                is_vibrant = hsv_m[1] > 110 or hsv_m[2] > 200

            if (work_h*0.005)**2 < area < (work_h*0.8)**2:
                if ratio > 0.45 and solidity > 0.75 and extent > 0.5:
                    if (not is_skin and is_vibrant) or ratio > 0.85:
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                            dist_c = np.sqrt((cx-work_w/2)**2 + (cy-work_h/2)**2)
                            found.append({'cnt':c, 'center':(cx,cy), 'area':area, 'weight': 1.0/(1.0+dist_c/work_w)})
            
            if tag != "SILENT":
                all_raw.append({'box':(x,y,w,h), 'status': "RED", 'center':(x+w//2, y+h//2)})
        return found

    # 🚀 ADAPTIVE SHUTTER LOOP: Try different adaptive thresholds to find best candidates
    for ws in [13, 31, 61, 101, 151]:
        thr = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ws, 2)
        c_found = process_contours(cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0], tag="SILENT" if ws > 13 else "")
        
        # Deduplicate and grow global candidates list
        for cand in c_found:
            if not any(np.sqrt((cand['center'][0]-c['center'][0])**2 + (cand['center'][1]-c['center'][1])**2) < 15 for c in candidates):
                candidates.append(cand)
        
        # Internal early exit: If we have > 9 firm candidates, we likely have enough
        if len(candidates) >= 12: break

    trace.append(f"Trace: Final Candidates={len(candidates)} (from multi-scale scan)")
    if sharp < 25: trace.append("⚠️ WARNING: Image blurry. Hold steady.")

    # --- 3. CLUSTERING PHASE: BFS CONNECTIVITY ---
    if len(candidates) >= 4:
        # Cluster logic: Growth / Connected Components
        pts = np.array([c['center'] for c in candidates])
        
        # 📏 ROBUST SCALE: Use median to ignore tiny noise/glare
        valid_areas = [c['area'] for c in candidates]
        med_w = np.sqrt(np.median(valid_areas)) if valid_areas else 1.0
        
        threshold_dist = med_w * 6.5
        min_dist = med_w * 0.55
        
        visited = [False] * len(candidates)
        all_clusters = []
        
        for i in range(len(candidates)):
            if visited[i]: continue
            curr_cluster = [i]
            queue = [i]
            visited[i] = True
            while queue:
                u = queue.pop(0)
                for v in range(len(candidates)):
                    if not visited[v]:
                        d = np.sqrt(np.sum((pts[u] - pts[v])**2))
                        if min_dist < d < threshold_dist:
                            visited[v] = True
                            curr_cluster.append(v)
                            queue.append(v)
            all_clusters.append([candidates[idx] for idx in curr_cluster])
        
        max_score = 0.1 # Very low baseline to encourage locking
        for cluster in all_clusters:
            if len(cluster) < 4: continue
            
            # 📐 Calculate Cluster Coverage (Physical size in image)
            c_pts = np.array([c['center'] for c in cluster])
            c_min_x, c_min_y = np.min(c_pts, axis=0)
            c_max_x, c_max_y = np.max(c_pts, axis=0)
            c_w, c_h = c_max_x - c_min_x, c_max_y - c_min_y
            curr_coverage = (c_w * c_h) / (work_w * work_h)
            
            # 🧬 Robust Metric: Area Consistency (Are they similar in size?)
            c_areas = [c['area'] for c in cluster]
            area_std = np.std(c_areas) / (np.mean(c_areas) + 1e-6)
            consistency_score = 1.0 / (1.0 + area_std)

            # Regularity & Bucketing Score
            reg_score = 1.0
            all_dists = []
            for p_m in cluster:
                px, py = p_m['center']
                d_to_others = sorted([np.sqrt((px-o['center'][0])**2 + (py-o['center'][1])**2) for o in cluster if o is not p_m])
                if d_to_others: all_dists.extend(d_to_others[:2])
            
            if all_dists:
                reg_score = 1.0 / (1.0 + np.std(all_dists) / (np.mean(all_dists) + 1e-6))

            # Bonus for 3x3 layout
            grid_bonus = 1.0
            if len(cluster) >= 7:
                pts_sort_y = sorted(cluster, key=lambda x: x['center'][1])
                rows = []
                if pts_sort_y:
                    curr_row = [pts_sort_y[0]]
                    for k in range(1, len(pts_sort_y)):
                        if pts_sort_y[k]['center'][1] - curr_row[-1]['center'][1] < med_w * 1.3:
                            curr_row.append(pts_sort_y[k])
                        else:
                            rows.append(curr_row); curr_row = [pts_sort_y[k]]
                    rows.append(curr_row)
                if len(rows) == 3: grid_bonus = 2.0

            # Aggressive Score: Prioritize Consistency and Count
            score = (len(cluster)**2) * consistency_score * reg_score * grid_bonus * (1.0 + curr_coverage * 5)
            
            if score > max_score:
                max_score, best_cluster = score, cluster
                best_reg_score, cluster_coverage = reg_score, curr_coverage

        status_msg = f"Locked cluster: {len(best_cluster)} stks (RegScore={best_reg_score:.2f}, FinalScore={max_score:.1f})"
        trace.append(status_msg)

    # --- ADVANCED DIAGNOSTIC DRAWING ---
    if show_diag:
        # Draw ALL raw candidates in RED
        for r in candidates:
            c = r['cnt']
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(work_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        
        # Draw INCLUDED cluster in GREEN circles
        for b in best_cluster:
            cv2.circle(work_img, b['center'], 5, (0, 255, 0), -1)
            
        diag_imgs['Sticker Candidates'] = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB)

    # --- 4. TRANSFORMATION & OUTPUT ---
    if len(best_cluster) < 4:
        return None, None, "Searching...", diag_imgs, None, None, trace, bright, sharp

    # Coordinate Extrapolation
    cluster_pts = np.array([c['center'] for c in best_cluster], dtype="float32")
    rect_min = cv2.minAreaRect(cluster_pts)
    box_raw = cv2.boxPoints(rect_min)
    center_box = np.mean(box_raw, axis=0)
    
    expanded = []
    for p in box_raw:
        vec = p - center_box
        vlen = np.linalg.norm(vec)
        # Push outward by 0.7 * sticker width to reach face edges
        p_new = p + (vec / vlen) * (med_w * 0.7) if vlen > 0 else p
        expanded.append(p_new)
    
    # Scale back to original resolution
    scale = h_p / work_h
    src_pts = np.array(expanded, dtype="float32") * scale
    
    # Sort source rectangle
    s = src_pts.sum(axis=1)
    diff = np.diff(src_pts, axis=1)
    ordered_src = np.zeros((4, 2), dtype="float32")
    ordered_src[0] = src_pts[np.argmin(s)]     # TL
    ordered_src[2] = src_pts[np.argmax(s)]     # BR
    ordered_src[1] = src_pts[np.argmin(diff)]  # TR
    ordered_src[3] = src_pts[np.argmax(diff)]  # BL
    
    dst_pts = np.array([[0,0], [299,0], [299,299], [0,299]], dtype="float32")
    M_warp = cv2.getPerspectiveTransform(ordered_src, dst_pts)
    warped = cv2.warpPerspective(img_padded, M_warp, (300, 300))
    debug_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    # Color Analysis & Refinement
    std_colors = get_calibrated_colors()
    detected = ['White'] * 9
    metrics = []
    refined_centers = []
    
    hsv_w = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    sat_w = hsv_w[:,:,1]
    
    for r in range(3):
        for c in range(3):
            tx, ty = int((c+0.5)*100), int((r+0.5)*100)
            # Refine local center with Saturation Centroid
            y1, y2, x1, x2 = max(0,ty-35), min(300,ty+35), max(0,tx-35), min(300,tx+35)
            roi_sat = sat_w[y1:y2, x1:x2]
            moms = cv2.moments(roi_sat)
            final_x, final_y, off = tx, ty, (0,0)
            if moms["m00"] > 50:
                sx, sy = x1 + int(moms["m10"]/moms["m00"]), y1 + int(moms["m01"]/moms["m00"])
                if np.sqrt((sx-tx)**2 + (sy-ty)**2) < 30:
                    final_x, final_y, off = sx, sy, (sx-tx, sy-ty)
            
            refined_centers.append((final_x, final_y))
            
            # --- COLOR SAMPLING WITH EROSION (User Suggestion) ---
            # Instead of a simple 12x12, we take a 16x16 and then keep only the central 8x8
            # This is mathematically equivalent to 'Erosion' to avoid black sticker borders.
            roi_raw = warped[max(0,final_y-8):min(300,final_y+8), max(0,final_x-8):min(300,final_x+8)]
            if roi_raw.size > 0:
                # Target the central core of the sticker (Eroded region)
                h_r, w_r = roi_raw.shape[:2]
                roi_eroded = roi_raw[h_r//4 : h_r - h_r//4, w_r//4 : w_r - w_r//4]
                bgr = np.median(roi_eroded, axis=(0,1)).astype(np.uint8)
            else:
                bgr = [0,0,0]
                
            lab = cv2.cvtColor(np.uint8([[[bgr[0],bgr[1],bgr[2]]]]), cv2.COLOR_BGR2LAB)[0][0]
            
            min_d, best_c, dists = 999.0, 'White', {}
            for name, (hs,ss,vs) in std_colors.items():
                target_lab = cv2.cvtColor(cv2.cvtColor(np.uint8([[[hs,ss,vs]]]), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2LAB)[0][0]
                dV = np.sqrt(0.1*(float(lab[0])-target_lab[0])**2 + 2.4*(float(lab[1])-target_lab[1])**2 + 2.4*(float(lab[2])-target_lab[2])**2)
                dists[name] = np.round(dV, 1)
                if dV < min_d: min_d, best_c = dV, name
            
            detected[r*3+c] = best_c
            metrics.append({'slot':r*3+c, 'final_color':best_c, 'distances':dists, 'snap_offset':off, 'confidence':np.round(100 - min_d, 1)})
            cv2.circle(debug_warped, (final_x, final_y), 5, (255,255,255), 1)
            cv2.putText(debug_warped, best_c[0], (final_x-5, final_y+5), 0, 0.4, (255,255,255), 1)

    # Final AR Step
    M_inv = cv2.getPerspectiveTransform(dst_pts, ordered_src)
    ar_overlay = img_padded.copy()
    for i, (fx, fy) in enumerate(refined_centers):
        p_orig = cv2.perspectiveTransform(np.array([[[fx, fy]]], dtype="float32"), M_inv)
        px, py = int(p_orig[0][0][0]), int(p_orig[0][0][1])
        cv2.circle(ar_overlay, (px, py), int(h_p/140), (255,255,255), -1)
        cv2.putText(ar_overlay, detected[i][0], (px+10, py+10), 0, h_p/1200, (255,255,255), 2)
    
    detected[4] = expected_center # Protect center
    return detected, debug_warped, f"Locked {len(best_cluster)} stks", diag_imgs, cv2.cvtColor(ar_overlay[pad:-pad, pad:-pad], cv2.COLOR_BGR2RGB), metrics, trace, bright, sharp

def run_manual_grid_extract(image_bytes, expected_center, scale_percent):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None: return None, None, False
    
    h, w = img.shape[:2]
    # Use the scale requested by user or 50% default
    gs = int(min(h, w) * (scale_percent / 100.0))
    sx, sy = (w - gs) // 2, (h - gs) // 2
    
    # Crop and resize to 300x300 to unify UI with Auto-detect
    cropped = img[sy:sy+gs, sx:sx+gs]
    warped = cv2.resize(cropped, (300, 300))
    debug_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    std_colors = get_calibrated_colors()
    detected = ['White'] * 9
    cell_sz = 100
    for r in range(3):
        for c in range(3):
            cx, cy = int((c+0.5)*cell_sz), int((r+0.5)*cell_sz)
            roi = warped[cy-10:cy+10, cx-10:cx+10]
            if roi.size == 0: continue
            b_a, g_a, r_a = np.median(roi, axis=(0,1)).astype(np.uint8)
            lab = cv2.cvtColor(np.uint8([[[b_a,g_a,r_a]]]), cv2.COLOR_BGR2LAB)[0][0]
            
            min_d, best = float('inf'), 'White'
            for name, (hs,ss,vs) in std_colors.items():
                sl = cv2.cvtColor(cv2.cvtColor(np.uint8([[[hs,ss,vs]]]), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2LAB)[0][0]
                d = np.sqrt(0.5*(int(lab[0])-int(sl[0]))**2 + 1.25*(int(lab[1])-int(sl[1]))**2 + (int(lab[2])-int(sl[2]))**2)
                if d < min_d: min_d, best = d, name
            detected[r*3+c] = best
            cv2.circle(debug_warped, (cx, cy), 15, (255, 255, 255), 2)
            cv2.putText(debug_warped, best, (cx-30, cy+35), 0, 0.5, (255,255,255), 1)

    detected[4] = expected_center
    return detected, debug_warped, True

# ── LEGACY CV FUNCTIONS (Kept for manual override fallback) ──────────────────
def get_calibrated_colors():
    defaults = {'White':(0,30,220), 'Yellow':(30,160,200), 'Orange':(12,200,240), 'Red':(0,210,180), 'Green':(60,180,150), 'Blue':(110,180,160)}
    for k, v in st.session_state.custom_std_colors.items(): defaults[k] = tuple(v)
    return defaults

# ── UI COMPONENTS ────────────────────────────────────────────────────────────
def render_3d_solution(solution_str, speed):
    def inv(s):
        r = []
        for m in reversed(s.split()):
            if "'" in m: r.append(m.replace("'",""))
            elif "2" in m: r.append(m)
            else: r.append(m+"'")
        return " ".join(r)
    html = f"""<script src="https://cdn.cubing.net/v0/js/cubing/twisty" type="module"></script>
    <twisty-player experimental-setup-alg="{inv(solution_str)}" alg="{solution_str}" visualization="PG3D" control-panel="bottom-row" tempo-scale="{speed}" background="none" style="width:100%; height:380px;"></twisty-player>"""
    components.html(html, height=400)

def render_interactive_map(active_face):
    grid = {'Up':(0,1), 'Left':(1,0), 'Front':(1,1), 'Right':(1,2), 'Back':(1,3), 'Down':(2,1)}
    html = '<div style="display:grid;grid-template-columns:repeat(4,60px);gap:8px;justify-content:center;font-family:sans-serif;">'
    for r in range(3):
        for c in range(4):
            f_k = next((f for f, p in grid.items() if p == (r, c)), None)
            if f_k:
                colors = st.session_state.cube_state[f_k]
                is_sc = (f_k in st.session_state.processed_photos)
                active_style = "border:3px solid #00e5ff; box-shadow:0 0 10px #00e5ff;" if f_k == active_face else "border:1px solid rgba(255,255,255,0.2);"
                html += f"""<a href="/?face_click={f_k}" target="_self" class="face-link">
                    <div style="opacity:{1.0 if (is_sc or f_k==active_face) else 0.4}; text-align:center;">
                        <div style="font-size:10px; font-weight:bold; color:{'#00e5ff' if f_k==active_face else '#ddd'}; margin-bottom:2px;">{COLOR_EMOJIS[CENTER_COLORS[f_k]]} {f_k} {'✅' if is_sc else ''}</div>
                        <div style="display:grid;grid-template-columns:repeat(3,18px);gap:2px; {active_style} border-radius:4px; padding:3px; background:rgba(255,255,255,0.05);">"""
                for clr in colors:
                    html += f'<div style="width:18px;height:18px;background:{HEX_COLORS[clr]};border:1px solid rgba(0,0,0,0.3);border-radius:2px;"></div>'
                html += '</div></div></a>'
            else: html += '<div></div>'
    html += '</div>'
    return html

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    app_mode = st.radio("Mode:", ["📸 Scan & Solve", "⚙️ Tune Colors"], label_visibility="collapsed")
    st.divider()
    if app_mode == "📸 Scan & Solve":
        st.markdown("### ⚙️ Settings")
        st.session_state.auto_advance = st.checkbox("⏩ Auto-advance on success", value=st.session_state.auto_advance)
        st.divider()
        st.markdown("### 🗺️ Interactive Map")
        st.caption("Click a face below to jump to its scanner.")
        st.markdown(render_interactive_map(st.session_state.programmatic_face), unsafe_allow_html=True)
        st.divider()
        with st.expander("🛠️ Advanced Tools"):
            st.session_state.show_diag = st.toggle("🔍 Diagnostic Vision", value=st.session_state.get('show_diag', False))
            st.session_state.show_guide = st.toggle("🎯 Show Camera Guide", value=st.session_state.show_guide)
            st.session_state.cube_size = st.slider("Manual Grid Size (%)", 10, 100, st.session_state.cube_size)
            if st.button("🗑️ Reset All Scans", use_container_width=True):
                st.session_state.processed_photos = {}
                st.session_state.shared_face_images = {}
                st.rerun()
    else:
        st.markdown("Map is disabled in Calibration mode.")

# ── MAIN SCANNER ─────────────────────────────────────────────────────────────
if app_mode == "📸 Scan & Solve":
    curr = st.session_state.programmatic_face
    st.title("🧊 AI Rubik's Solver")

    # Header Nav
    hcol1, hcol2, hcol3 = st.columns([1, 2, 1])
    with hcol1: 
        if st.button("⬅️ Previous", use_container_width=True): 
            st.session_state.programmatic_face = FACES[(FACES.index(curr)-1)%6]; st.rerun()
    with hcol2: st.markdown(f"<h2 style='text-align:center; margin-top:-10px;'>{COLOR_EMOJIS[CENTER_COLORS[curr]]} {curr} Face</h2>", unsafe_allow_html=True)
    with hcol3: 
        if st.button("Next ➡️", use_container_width=True): 
            st.session_state.programmatic_face = FACES[(FACES.index(curr)+1)%6]; st.rerun()

    if st.session_state.get('last_face') != curr:
        st.session_state.last_face = curr
        st.session_state.uploader_key_version += 1

    st.info(f"🧭 **HOLD:** {ORIENTATION_GUIDE[curr]}")
    c_cam, c_edit = st.columns([1, 1])
    
    with c_cam:
        st.markdown("### 📷 Camera Input")
        sh = st.session_state.shared_face_images.get(curr)
        if sh: st.success("♻️ Image shared/demo loaded.")
        
        imth = st.radio("Method:", ["📹 Live", "📂 Upload"], horizontal=True, label_visibility="collapsed")
        st.session_state.auto_detect = st.toggle("🤖 Auto-Detect", value=st.session_state.auto_detect)
        
        v = st.session_state.uploader_key_version
        
        # Wrapped camera in a container for the CSS guide
        if imth == "📹 Live":
            st.markdown('<div class="camera-container">', unsafe_allow_html=True)
            if st.session_state.show_guide:
                st.markdown('<div class="camera-guide"></div>', unsafe_allow_html=True)
            buf = st.camera_input("P", key=f"c_{v}", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            buf = st.file_uploader("U", type=['jpg','png'], key=f"u_{v}")
        
        act = buf or (io.BytesIO(sh) if sh and imth == "📂 Upload" else None)
        if act:
            key = f"{curr}_{getattr(act, 'file_id', id(act))}"
            if st.session_state.processed_photos.get(curr) != key:
                if hasattr(act, 'seek'): act.seek(0)
                
                # BRANCH: Auto-Detect vs Manual Alignment
                msg_success = "✨ **I flattened the cube face!** Check the orientation and colors."
                msg_fail = "❌ Cube not found. Try to hold it flatter, or switch to Manual Mode."
                
                if st.session_state.auto_detect:
                    res = auto_detect_cube_face(act, CENTER_COLORS[curr], show_diag=st.session_state.get('show_diag'))
                    # Unpack 9 values: d, db, info, diags, track, metrics, trace, bright, sharp
                    if len(res) == 9:
                        d, db, info, diags, track, metrics, trace, bright, sharp = res
                    else:
                        d, db, info, diags, track, metrics, trace, bright, sharp = None, None, "API Error", {}, None, None, ["Error: Index mismatch"], 0, 0
                    
                    success = (d is not None)
                    
                    if success:
                        st.session_state.cube_state[curr] = d
                        t_col, r_col = st.columns([1, 1])
                        with t_col: st.image(track, caption="🎯 Target Locked", use_container_width=True)
                        with r_col: st.image(db, caption="📸 Color Result", use_container_width=True)
                        
                        # --- 🚦 QUALITY BADGES (PROFEEDBACK) ---
                        # Extract metrics from trace or results
                        q1, q2 = st.columns(2)
                        with q1:
                            b_status = "🌕 Good" if bright > 75 else ("🌑 Too Dark" if bright < 50 else "🌥️ Low Light")
                            st.write(f"**Brightness:** {bright:.0f} {b_status}")
                        with q2:
                            s_status = "✅ Sharp" if sharp > 40 else "🌫️ Blurry"
                            st.write(f"**Sharpness:** {sharp:.0f} {s_status}")
                        
                        st.success(msg_success)
                        
                        if st.session_state.get('show_diag'):
                            # HUD 1: Process Trace
                            if trace:
                                with st.expander("📝 View Process Trace Log", expanded=True):
                                    st.code("\n".join([f"> {l}" for l in trace]), language="markdown")
                            
                            # HUD 2: Detailed Metrics Table
                            if metrics:
                                with st.expander("🔬 View Detailed Color Metrics", expanded=False):
                                    df_data = []
                                    for m in metrics:
                                        row = {"Slot": m['slot']+1, "Color": m['final_color'], "Conf%": m['confidence']}
                                        for c_name, dist in m['distances'].items(): row[f"d({c_name[0]})"] = dist
                                        df_data.append(row)
                                    st.table(df_data)
                    else:
                        st.error(f"❌ Cube not found. (Diag: {info})")
                        if st.session_state.get('show_diag'):
                            if trace: st.warning("\n".join(trace))
                        
                    # Show all diagnostics images (Feature Map, Candidates, Edges, etc)
                    if st.session_state.get('show_diag') and diags:
                        st.info("🖼️ **Diagnostic Intermediate Steps**")
                        n_diags = len(diags)
                        dcols = st.columns(n_diags)
                        for i, (name, img) in enumerate(diags.items()):
                            with dcols[i]: st.image(img, caption=name, use_container_width=True)
                else:
                    d, db, success = run_manual_grid_extract(act, CENTER_COLORS[curr], st.session_state.cube_size)
                    info = "Manual"
                    msg_success = "📐 **Manual Alignment Used.**"
                    if success:
                        st.session_state.cube_state[curr] = d
                        st.image(db, caption="📸 Capture Preview", use_container_width=True)
                        st.success(msg_success)
                    msg_fail = "❌ Failed to read manual grid. Check your connection."

                if success:
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        if st.button("✅ Confirm & Save", type="primary", use_container_width=True, key=f"conf_{key}"):
                            st.session_state.processed_photos[curr] = key
                            st.session_state[f"cached_img_{curr}"] = db
                            if st.session_state.auto_advance:
                                un = [f for f in FACES if f not in st.session_state.processed_photos]
                                if un: st.session_state.programmatic_face = un[0]
                            st.rerun()
                    with col_b2:
                        if st.button("🔄 Retake", use_container_width=True, key=f"ign_{key}"):
                            st.session_state.uploader_key_version += 1; st.rerun()
                else:
                    st.error(msg_fail)
                    if hasattr(act, 'seek'): act.seek(0)
                    st.image(act, use_container_width=True)
                    if not st.session_state.auto_detect:
                        st.info("💡 Adjust the 'Manual Grid Size' slider in the sidebar to match the cube in the photo.")

        # Show the permanent saved image if it exists
        elif curr in st.session_state.processed_photos and f"cached_img_{curr}" in st.session_state:
            st.image(st.session_state[f"cached_img_{curr}"], caption="Saved Detection")

    with c_edit:
        st.markdown("### 🖱️ Edit Colors")
        st.caption("Click squares to cycle selection.")
        
        def cycle(i):
            seq = ['White', 'Yellow', 'Green', 'Blue', 'Red', 'Orange']
            cur_v = st.session_state.cube_state[curr][i]
            st.session_state.cube_state[curr][i] = seq[(seq.index(cur_v)+1)%len(seq)]
            st.session_state.last_solution = None

        for r in range(3):
            ecols = st.columns(3)
            for c in range(3):
                idx = r*3 + c
                with ecols[c]:
                    if idx == 4: st.button(f"{COLOR_EMOJIS[CENTER_COLORS[curr]]}\nCtr", disabled=True, use_container_width=True)
                    else:
                        v_c = st.session_state.cube_state[curr][idx]
                        st.button(f"{COLOR_EMOJIS[v_c]}\n{v_c}", key=f"e_{curr}_{idx}", use_container_width=True, on_click=cycle, args=(idx,))

        if st.button("🧹 Reset Face", use_container_width=True):
            st.session_state.cube_state[curr] = ['White']*9
            st.session_state.cube_state[curr][4] = CENTER_COLORS[curr]
            st.session_state.processed_photos.pop(curr, None)
            st.rerun()

    st.divider()
    if st.button("🚀 VALIDATE & SOLVE", type="primary", use_container_width=True):
        if len(st.session_state.processed_photos) < 6: st.warning("⚠️ Scan all 6 faces!")
        else:
            ok, msg = validate_cube_state(st.session_state.cube_state)
            if ok: 
                with st.spinner("Solving..."): st.session_state.last_solution = solve_cube(st.session_state.cube_state)
            else: st.error(f"❌ {msg}")

    sol = st.session_state.get('last_solution')
    if sol:
        if sol == "!IMPOSSIBLE_STATE!": st.error("❌ Physical Error: Impossible State. Check alignment.")
        else:
            st.success(f"✅ moves: {sol}")
            render_3d_solution(sol, st.select_slider("Speed", [0.5, 1.0, 2.0], 1.0))

elif app_mode == "⚙️ Tune Colors":
    st.title("⚙️ Color Calibration")
    c_cal = st.radio("Target:", AVAILABLE_COLORS, horizontal=True, format_func=lambda x: f"{COLOR_EMOJIS[x]} {x}")
    v = st.session_state.uploader_key_version
    cal_meth = st.radio("Method:", ["📹 Live", "📂 Upload"], horizontal=True, label_visibility="collapsed", key="cal_meth")
    if cal_meth == "📹 Live":
        cal_buf = st.camera_input("Sample", key=f"cal_{v}")
    else:
        cal_buf = st.file_uploader("Upload Image", type=['jpg','png'], key=f"ulc_{v}")
    if cal_buf:
        img_cal = cv2.imdecode(np.asarray(bytearray(cal_buf.read()), dtype=np.uint8), 1)
        if img_cal is not None:
            rgb_cal = cv2.cvtColor(img_cal, cv2.COLOR_BGR2RGB)
            val_xy = streamlit_image_coordinates(rgb_cal, key=f"click_{c_cal}_{v}")
            cx, cy = (val_xy['x'], val_xy['y']) if val_xy else (img_cal.shape[1]//2, img_cal.shape[0]//2)
            roi = img_cal[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
            if roi.size > 0:
                avg_bgr = np.median(roi, axis=(0,1)).astype(np.uint8)
                hsv_val = cv2.cvtColor(np.uint8([[[avg_bgr[0],avg_bgr[1],avg_bgr[2]]]]), cv2.COLOR_BGR2HSV)[0][0]
                cl1, cl2 = st.columns(2)
                with cl1:
                    cv2.circle(rgb_cal, (cx, cy), 15, (255,255,255), 3)
                    st.image(rgb_cal, caption="Click to sample surface color")
                with cl2:
                    st.markdown(f'<div style="width:100px;height:100px;background:rgb({avg_bgr[2]},{avg_bgr[1]},{avg_bgr[0]});border:2px solid #fff;"></div>', unsafe_allow_html=True)
                    if st.button(f"Save HSV {hsv_val} for {c_cal}", type="primary"):
                        st.session_state.custom_std_colors[c_cal] = [int(hsv_val[0]), int(hsv_val[1]), int(hsv_val[2])]
                        with open(CALIB_FILE, 'w') as f_json: json.dump(st.session_state.custom_std_colors, f_json)
                        st.success("Calibration Saved!")
