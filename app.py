import json
import os

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from rubiks_core import solve_cube, validate_cube_state

st.set_page_config(page_title="AI Rubik's Cube Solver", page_icon="🧊", layout="wide")

FACES = ["Up", "Left", "Front", "Right", "Back", "Down"]
CENTER_COLORS = {
    "Up": "White",
    "Left": "Orange",
    "Front": "Green",
    "Right": "Red",
    "Back": "Blue",
    "Down": "Yellow",
}
HEX_COLORS = {
    "White": "#F8F9FA",
    "Yellow": "#F7DC6F",
    "Orange": "#F39C12",
    "Red": "#E74C3C",
    "Green": "#27AE60",
    "Blue": "#3498DB",
}
ORIENTATION_GUIDE = {
    "Up": "Point camera to White center. Keep Blue side on TOP.",
    "Left": "Point camera to Orange center. Keep White side on TOP.",
    "Front": "Point camera to Green center. Keep White side on TOP.",
    "Right": "Point camera to Red center. Keep White side on TOP.",
    "Back": "Point camera to Blue center. Keep White side on TOP.",
    "Down": "Point camera to Yellow center. Keep Green side on TOP.",
}
CALIB_FILE = "calibration_profile.json"
FACE_LAYOUT = {
    "Up": (0, 1),
    "Left": (1, 0),
    "Front": (1, 1),
    "Right": (1, 2),
    "Back": (1, 3),
    "Down": (2, 1),
}


def init_session():
    defaults = {
        "cube_state": {
            f: ["White", "White", "White", "White", CENTER_COLORS[f], "White", "White", "White", "White"]
            for f in FACES
        },
        "scanned_faces": set(),
        "active_face": "Front",
        "last_scan": None,
        "solution": None,
        "speed": 1.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_calibration():
    profile = {
        "White": [0, 20, 240],
        "Yellow": [30, 170, 220],
        "Orange": [15, 220, 230],
        "Red": [0, 220, 190],
        "Green": [60, 180, 170],
        "Blue": [110, 190, 170],
    }
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                for color, hsv in loaded.items():
                    if color in profile and isinstance(hsv, list) and len(hsv) == 3:
                        profile[color] = hsv
        except (json.JSONDecodeError, OSError):
            pass
    return profile


def save_calibration(profile):
    with open(CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def _hsv_to_lab(hsv):
    hsv_pixel = np.uint8([[[hsv[0], hsv[1], hsv[2]]]])
    bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0]
    return lab


def detect_face_colors(image_bytes, expected_center, manual_grid_size=65, diagnostic=False):
    raw = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    # 1) Preprocessing: padding, normalize, gray, blur
    h, w = frame.shape[:2]
    pad = int(min(h, w) * 0.08)
    padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2) Feature extraction & clustering proxy via adaptive threshold + contours
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 3
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        ratio = cw / float(ch + 1e-6)
        if 500 < area < 60000 and 0.7 < ratio < 1.3:
            candidates.append((x, y, cw, ch))

    warped = None
    debug = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    if candidates:
        # 3) Perspective warping (approximate quadrilateral by envelope)
        xs = [x for x, _, cw, _ in candidates] + [x + cw for x, _, cw, _ in candidates]
        ys = [y for _, y, _, ch in candidates] + [y + ch for _, y, _, ch in candidates]
        x0, x1 = max(0, min(xs)), min(padded.shape[1], max(xs))
        y0, y1 = max(0, min(ys)), min(padded.shape[0], max(ys))
        roi = padded[y0:y1, x0:x1]
        if roi.size > 0:
            warped = cv2.resize(roi, (300, 300))
            cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 255, 255), 2)

    # Manual fallback
    if warped is None:
        hh, ww = padded.shape[:2]
        size = int(min(hh, ww) * (manual_grid_size / 100.0))
        sx, sy = (ww - size) // 2, (hh - size) // 2
        warped = cv2.resize(padded[sy : sy + size, sx : sx + size], (300, 300))
        cv2.rectangle(debug, (sx, sy), (sx + size, sy + size), (255, 100, 100), 2)

    hsv_img = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    sat = hsv_img[:, :, 1]
    calibration = load_calibration()
    calibration_lab = {k: _hsv_to_lab(v) for k, v in calibration.items()}

    stickers = ["White"] * 9
    for r in range(3):
        for c in range(3):
            tx, ty = int((c + 0.5) * 100), int((r + 0.5) * 100)

            # 4) Sub-pixel refinement by local saturation centroid
            y1, y2 = max(0, ty - 30), min(300, ty + 30)
            x1, x2 = max(0, tx - 30), min(300, tx + 30)
            patch = sat[y1:y2, x1:x2]
            moments = cv2.moments(patch)
            if moments["m00"] > 20:
                fx = x1 + int(moments["m10"] / moments["m00"])
                fy = y1 + int(moments["m01"] / moments["m00"])
            else:
                fx, fy = tx, ty

            sample = warped[max(0, fy - 4) : min(300, fy + 4), max(0, fx - 4) : min(300, fx + 4)]
            median_bgr = np.median(sample, axis=(0, 1))
            sample_lab = cv2.cvtColor(np.uint8([[median_bgr]]), cv2.COLOR_BGR2LAB)[0][0]

            # 5) Color matching in LAB distance
            best_name = "White"
            best_dist = 10e9
            for color_name, target_lab in calibration_lab.items():
                dist = float(np.linalg.norm(sample_lab.astype(np.float32) - target_lab.astype(np.float32)))
                if dist < best_dist:
                    best_dist = dist
                    best_name = color_name

            idx = r * 3 + c
            stickers[idx] = best_name
            cv2.circle(debug, (fx, fy), 8, (0, 255, 255), 2)

    stickers[4] = expected_center
    result = {
        "stickers": stickers,
        "warped_rgb": cv2.cvtColor(warped, cv2.COLOR_BGR2RGB),
        "debug_rgb": debug if diagnostic else None,
    }
    return result


def render_face_editor(face_name):
    st.markdown("#### Click-to-Edit")
    colors_cycle = list(HEX_COLORS.keys())
    for r in range(3):
        cols = st.columns(3)
        for c in range(3):
            idx = r * 3 + c
            now_color = st.session_state.cube_state[face_name][idx]
            label = f"{now_color[:2]}"
            if cols[c].button(label, key=f"edit_{face_name}_{idx}", use_container_width=True):
                if idx == 4:
                    st.warning("Center color is fixed by cube orientation.")
                else:
                    next_idx = (colors_cycle.index(now_color) + 1) % len(colors_cycle)
                    st.session_state.cube_state[face_name][idx] = colors_cycle[next_idx]
                    st.session_state.solution = None
                    st.rerun()


def render_sidebar():
    with st.sidebar:
        st.header("Global Sidebar")
        mode = st.radio("App Mode", ["Scan & Solve", "Tune Colors"], index=0)
        st.divider()

        st.markdown("#### Interactive Map")
        for row in range(3):
            cols = st.columns(4)
            for col in range(4):
                face = next((f for f, pos in FACE_LAYOUT.items() if pos == (row, col)), None)
                if not face:
                    cols[col].write("")
                    continue
                done = face in st.session_state.scanned_faces
                flag = " ✅" if done else ""
                if cols[col].button(f"{face[0]}{flag}", key=f"goto_{face}", use_container_width=True):
                    st.session_state.active_face = face
                    st.rerun()

        st.markdown("#### Advanced Tools")
        diagnostic = st.checkbox("Diagnostic Vision", value=False)
        manual_grid_size = st.slider("Manual Grid Size", 45, 85, 65, 1)
        auto_advance = st.checkbox("Auto-advance", value=True)
        speed = st.slider("3D Speed", 0.25, 2.0, st.session_state.speed, 0.05)
        st.session_state.speed = speed

        if st.button("Reset Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    return mode, diagnostic, manual_grid_size, auto_advance


def render_scan_solve(diagnostic, manual_grid_size, auto_advance):
    active = st.session_state.active_face
    progress = len(st.session_state.scanned_faces) / 6.0

    st.title("🧊 Scan & Solve")
    st.progress(progress, text=f"Scanned {len(st.session_state.scanned_faces)} / 6 faces")
    st.subheader(f"Current Face: {active}")
    st.info(ORIENTATION_GUIDE[active])

    left, right = st.columns([1.5, 1])
    with left:
        image = st.camera_input("Capture current face")
        if image is not None:
            result = detect_face_colors(
                image,
                expected_center=CENTER_COLORS[active],
                manual_grid_size=manual_grid_size,
                diagnostic=diagnostic,
            )
            if result:
                st.session_state.last_scan = result
                st.session_state.cube_state[active] = result["stickers"]
                st.image(result["warped_rgb"], caption="Warped 300x300 face", use_container_width=True)
                if result["debug_rgb"] is not None:
                    st.image(result["debug_rgb"], caption="Diagnostic Vision", use_container_width=True)
            else:
                st.error("Failed to parse image. Try better lighting and alignment.")

    with right:
        render_face_editor(active)
        if st.button("Save This Face", type="primary", use_container_width=True):
            st.session_state.scanned_faces.add(active)
            st.session_state.solution = None
            if auto_advance:
                remains = [f for f in FACES if f not in st.session_state.scanned_faces]
                if remains:
                    st.session_state.active_face = remains[0]
            st.rerun()

        st.markdown("#### Face Preview")
        preview_cols = st.columns(3)
        stickers = st.session_state.cube_state[active]
        for idx, color in enumerate(stickers):
            preview_cols[idx % 3].markdown(
                f"<div style='height:34px;background:{HEX_COLORS[color]};border:1px solid #333;border-radius:4px;margin:3px;'></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    solve_disabled = len(st.session_state.scanned_faces) < 6
    if st.button("VALIDATE & SOLVE", type="primary", use_container_width=True, disabled=solve_disabled):
        valid, message = validate_cube_state(st.session_state.cube_state)
        if not valid:
            st.error(message)
        else:
            ok, solution_or_error = solve_cube(st.session_state.cube_state)
            if not ok:
                st.error(solution_or_error)
            else:
                st.session_state.solution = solution_or_error
                st.success("Solution generated successfully.")

    if st.session_state.solution:
        st.markdown("### 3D Solution Player")
        sp = st.session_state.speed
        st.code(st.session_state.solution, language="text")
        st.markdown(
            f"""
            <script src="https://cdn.cubing.net/js/cubing/twisty" type="module"></script>
            <twisty-player
                alg="{st.session_state.solution}"
                visualization="PG3D"
                control-panel="bottom-row"
                tempo-scale="{sp}"
                style="width:100%;height:420px;">
            </twisty-player>
            """,
            unsafe_allow_html=True,
        )


def render_tune_colors():
    st.title("⚙️ Tune Colors")
    st.caption("Click cube sticker on snapshot to update HSV calibration profile.")

    profile = load_calibration()
    target_color = st.selectbox("Target color", list(HEX_COLORS.keys()))
    image = st.camera_input("Capture calibration sample")

    if image is not None:
        raw = np.asarray(bytearray(image.read()), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Failed to load camera frame.")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        selected = streamlit_image_coordinates(rgb, key="calibration_pick")
        st.image(rgb, caption="Click a sticker area", use_container_width=True)
        if selected:
            x, y = selected["x"], selected["y"]
            y1, y2 = max(0, y - 4), min(frame.shape[0], y + 4)
            x1, x2 = max(0, x - 4), min(frame.shape[1], x + 4)
            roi = frame[y1:y2, x1:x2]
            hsv_value = cv2.cvtColor(np.median(roi, axis=(0, 1)).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[
                0
            ][0].tolist()
            st.write(f"Sample HSV: `{hsv_value}`")
            if st.button(f"Save HSV to {target_color}", type="primary"):
                profile[target_color] = hsv_value
                save_calibration(profile)
                st.success("Saved to calibration_profile.json")

    st.markdown("#### Current Profile")
    st.json(profile)


init_session()
mode, diagnostic, manual_grid_size, auto_advance = render_sidebar()

if mode == "Scan & Solve":
    render_scan_solve(diagnostic, manual_grid_size, auto_advance)
else:
    render_tune_colors()
