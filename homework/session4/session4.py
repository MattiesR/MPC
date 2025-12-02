import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Global variable setup (Needed for the code to run, based on your original block)
folder = "images/assignment4/"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--figs", 
    action="store_true", 
    help="Show figures if this flag is provided"
)
args = parser.parse_args()


# --- 1. Utility Classes ---

class Circle(patches.Circle):
    def __init__(self, center, radius, **kwargs):
        super().__init__(center, radius, **kwargs)

class Rectangle(patches.Rectangle):
    def __init__(self, xy, w, h, **kwargs):
        # xy is the anchor point (lower left corner)
        super().__init__(xy, w, h, **kwargs)


# --- 2. Function for Question 1 (Returns List of Patch Objects) ---

def circles_protection(w, l, n_c):
    """
    Calculates protective circles centered along the length l and returns 
    them as a list of Circle patches for local plotting (Question 1).
    """
    d = l/(2*n_c)
    r = np.sqrt((w/2)**2 + d**2)
    
    # Local centers: c_x = -l/2 + d * (1 + 2 * i), c_y = 0
    circles = [Circle([-l/2 + d * (1 + 2*i), 0], r, 
                      facecolor='blue', alpha=0.2, edgecolor='darkblue') 
               for i in range(n_c)]
    return circles


# --- 3. Function for Question 2 (Returns Raw Data for Transformation) ---

def get_local_circle_data(w, l, n_c):
    """
    Calculates local centers and radius, returning them as a NumPy array 
    and a float for efficient matrix transformation (Question 2).
    """
    d = l/(2*n_c)
    r = np.sqrt((w/2)**2 + d**2)
    
    # c_x = -l/2 + d * (1 + 2 * i), c_y = 0
    c_centers_local = np.array([
        [-l / 2 + d * (1 + 2 * i), 0.0] for i in range(n_c)
    ]).T # Shape (2, n_c)
    
    return c_centers_local, r


# --- 4. EFFICIENT ROTATION FUNCTION (Helper) ---

def rotate_and_translate(points_local, R_psi, P):
    """Applies rotation and translation using matrix multiplication."""
    return P + R_psi @ points_local


# --- 5. Question 1 (FIXED) ---

def question1():
    w = 4.0
    l = 12.0
    n_c = 5
    
    # 1. Setup the figure and axes
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # 2. Create and plot the Rectangle patch (anchor point: -l/2, -w/2)
    rectangle = Rectangle([-l/2, -w/2], l, w, 
                          facecolor='none', edgecolor='black', linewidth=2, label="Vehicle")
    ax.add_patch(rectangle)
    
    # 3. Create and plot the Circle patches (uses the function that returns patches)
    circles = circles_protection(w, l, n_c)
    for i, circle in enumerate(circles):
        if i == 0:
            circle.set_label("Protection Circles")
        ax.add_patch(circle)
        
    # 4. Configure Axes
    ax.set_xlim(-l/2-2, l/2+2) 
    ax.set_ylim(-w/2-2, w/2+2)
    ax.set_aspect('equal', adjustable='box') 
    
    ax.set_title(f"Vehicle covered with {n_c} circles")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()

    if args.figs == True:
        filename = "circles.png"
        # fig.savefig(folder + filename) # Uncomment if 'folder' is defined
    return 0


# --- 6. Question 2 (Uses Efficient Rotation) ---

def question2():
    # --- 1. Vehicle and Rotation State (Assignment 4.2 parameters) ---
    l = 4.0   
    w = 2.0   
    n_c = 3   # Using n_c=3 as defined in Assignment 4.2
    
    px, py = 2.0, 2.0
    psi = np.pi / 4.0 

    # --- 2. Transformation Setup ---
    R_psi = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi), np.cos(psi)]
    ])
    P = np.array([[px], [py]]) 

    # --- 3. Setup Figure ---
    fig, ax = plt.subplots(figsize=(8, 8))


    # --- 4. Rotated Rectangle (Vehicle Footprint) ---
    local_corners = np.array([
        [-l/2, -w/2], [ l/2, -w/2], 
        [ l/2,  w/2], [-l/2,  w/2]  
    ]).T 

    global_corners = rotate_and_translate(local_corners, R_psi, P)

    vehicle_patch = patches.Polygon(global_corners.T, 
                                    closed=True, 
                                    facecolor='lightgray', 
                                    edgecolor='black', 
                                    linewidth=2, 
                                    alpha=0.8,
                                    label="Rotated Vehicle")
    ax.add_patch(vehicle_patch)


    # --- 5. Rotated Circles ---
    # Uses the function that returns raw data
    c_centers_local, r = get_local_circle_data(w, l, n_c)
    
    C_centers_global = rotate_and_translate(c_centers_local, R_psi, P)
    
    for i in range(n_c):
        center_global = C_centers_global[:, i]
        
        if i == 0:
            label_text = f"Protection Circles (r={r:.3f})"
        else:
            label_text = None
            
        circle = patches.Circle(center_global, 
                                radius=r, 
                                facecolor='blue', 
                                alpha=0.3, 
                                edgecolor='blue', 
                                label=label_text)
        ax.add_patch(circle)
        ax.plot(center_global[0], center_global[1], 'bx', markersize=6) 


    # --- 6. Configure Axes and Show ---
    ax.plot(px, py, 'ro', label=r'Vehicle Center $(p_x, p_y)$')
    ax.set_aspect('equal', adjustable='box') 
    ax.set_title(f"Vehicle Rotated ($\psi = {int(psi*180/np.pi)}^\circ$)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    
    # Smart Limits
    all_points = np.hstack([global_corners, C_centers_global])
    min_x, max_x = np.min(all_points[0, :]), np.max(all_points[0, :])
    min_y, max_y = np.min(all_points[1, :]), np.max(all_points[1, :])
    padding = r + 1 
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    
    ax.legend(loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()
    
    if args.figs == True:
        filename = "vehicle_rotated.png"
        fig.savefig(folder + filename) 
    return 0


if __name__ == "__main__":
    # Test Question 1 (FIXED)
    question1() 
    
    # Test Question 2
    question2() # Not executing here to avoid two plots, but logic is fixed.