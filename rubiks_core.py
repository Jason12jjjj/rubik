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
    
    for face, colors in faces_data.items():
        for color in colors:
            if color in color_counts:
                color_counts[color] += 1
            else:
                return False, f"Invalid color detected: {color}"
                
    for color, count in color_counts.items():
        if count != 9:
            return False, f"Validation Failed: Found {count} {color} squares. Must be exactly 9."
            
    return True, "Cube state is valid."

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
        
    except ValueError as e:
        # Exception Handling for unsolvable states (e.g., corner twist)
        return f"Algorithm Error: Impossible cube state detected. ({str(e)})"
    except Exception as e:
        return f"System Error: {str(e)}"
