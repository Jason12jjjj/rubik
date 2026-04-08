"""Core validation and solving helpers for Rubik's Cube."""

from collections import Counter

import kociemba

COLORS = ("White", "Red", "Green", "Yellow", "Orange", "Blue")
FACES = ("Up", "Left", "Front", "Right", "Back", "Down")
COLOR_TO_FACE = {
    "White": "U",
    "Red": "R",
    "Green": "F",
    "Yellow": "D",
    "Orange": "L",
    "Blue": "B",
}


def validate_cube_state(faces_data):
    """Return (is_valid, message) after strict physical consistency checks."""
    for face in FACES:
        if face not in faces_data:
            return False, f"Missing face: {face}"
        if len(faces_data[face]) != 9:
            return False, f"{face} face has {len(faces_data[face])} stickers, expected 9."

    counts = Counter()
    locations = {c: set() for c in COLORS}
    for face, stickers in faces_data.items():
        for sticker in stickers:
            if sticker not in COLOR_TO_FACE:
                return False, f"Invalid color '{sticker}' detected on {face} face."
            counts[sticker] += 1
            locations[sticker].add(face)

    issues = []
    for color in COLORS:
        if counts[color] != 9:
            issues.append(f"{color}={counts[color]} (expected 9)")

    if issues:
        suspect_faces = sorted({f for c in COLORS if counts[c] != 9 for f in locations[c]})
        suffix = f" Check faces: {', '.join(suspect_faces)}." if suspect_faces else ""
        return False, "Color count mismatch: " + "; ".join(issues) + "." + suffix

    return True, "Cube state passed validation."


def to_kociemba_string(faces_data):
    """Convert face-color dictionary into URFDLB 54-char string."""
    face_order = ("Up", "Right", "Front", "Down", "Left", "Back")
    chars = []
    for face in face_order:
        for color in faces_data[face]:
            chars.append(COLOR_TO_FACE[color])
    return "".join(chars)


def solve_cube(faces_data):
    """Run Kociemba and return (ok, solution_or_error)."""
    try:
        cube_string = to_kociemba_string(faces_data)
        solution = kociemba.solve(cube_string)
        return True, solution
    except ValueError:
        return False, "Impossible cube state. Please recheck scanned stickers."
    except Exception as exc:
        return False, f"Solver error: {exc}"
