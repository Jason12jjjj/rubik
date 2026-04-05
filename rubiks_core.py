# ==============================================================================
# RUBIK'S CUBE CORE LOGIC & ALGORITHM INTERFACE
# ==============================================================================
import kociemba

# Standard Western Color Scheme Mapping
# U (Up) = White, R (Right) = Red, F (Front) = Green
# D (Down) = Yellow, L (Left) = Orange, B (Back) = Blue
COLOR_TO_FACE = {
    'White': 'U',
    'Red': 'R',
    'Green': 'F',
    'Yellow': 'D',
    'Orange': 'L',
    'Blue': 'B'
}

def validate_cube_state(faces_data):
    """
    Business Rule Validation: Checks if the scanned cube is mathematically possible.
    A valid cube must have exactly 9 squares of each of the 6 colors.
    """
    color_counts = {'White': 0, 'Red': 0, 'Green': 0, 'Yellow': 0, 'Orange': 0, 'Blue': 0}
    
    # Track which faces contain which colors to help the user find errors
    color_locations = {c: set() for c in color_counts.keys()}

    for face, colors in faces_data.items():
        if len(colors) != 9:
            return False, f"{face} face has {len(colors)} squares — expected exactly 9."
        for color in colors:
            if color in color_counts:
                color_counts[color] += 1
                color_locations[color].add(face)
            else:
                return False, f"Invalid color detected on {face} face: '{color}'"

    too_many = []
    too_few  = []
    for color, count in color_counts.items():
        if count > 9:     too_many.append((color, count))
        elif count < 9:   too_few.append((color, count))

    if not too_many and not too_few:
        return True, "Cube state is valid."

    # Build a highly descriptive error message pinpointing the exact issue
    err_parts = []
    if too_many:
        err_parts.append("太多了(>9): " + ", ".join([f"{c}有{cnt}个" for c, cnt in too_many]))
    if too_few:
        err_parts.append("太少了(<9): " + ", ".join([f"{c}有{cnt}个" for c, cnt in too_few]))
        
    problem_colors = [c for c, cnt in too_many] + [c for c, cnt in too_few]
    suspect_faces  = set()
    for pc in problem_colors:
        suspect_faces.update(color_locations[pc])
        
    err_msg = " | ".join(err_parts)
    if suspect_faces:
        err_msg += f" ➡️ 请重点检查这几个面是否有错: {', '.join(suspect_faces)} 面。"

    return False, err_msg

def solve_cube(faces_data):
    """
    Translates the 6 scanned faces into the URFDLB string format
    and uses the Kociemba algorithm to find the optimal solution.
    """
    try:
        # Order required by Kociemba: U, R, F, D, L, B
        # Assume faces_data is a dictionary with keys matching the faces
        cube_string = ""
        face_order = ['Up', 'Right', 'Front', 'Down', 'Left', 'Back']
        
        for face in face_order:
            if face not in faces_data or len(faces_data[face]) != 9:
                return "Error: Missing or incomplete face data."
                
            # Convert color names to Face notation (U, R, F, D, L, B)
            for color in faces_data[face]:
                cube_string += COLOR_TO_FACE[color]
                
        # Execute Kociemba Algorithm
        solution = kociemba.solve(cube_string)
        return solution

    except ValueError:
        return "!IMPOSSIBLE_STATE!"
    except Exception as e:
        return f"System Error: {str(e)}"
