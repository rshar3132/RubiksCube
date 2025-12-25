#pylint: disable=no-member
import cv2
import numpy as np
import kociemba

# --- CONFIGURATION ---

# Color ranges for the colorful stickers (Hue, Saturation, Value)
# Note: White is NOT here; it is handled specially in logic below.
COLOR_RANGES = {
    "red1":   [(0, 70, 50), (10, 255, 255)],      # Red part 1
    "red2":   [(170, 70, 50), (180, 255, 255)],   # Red part 2 (wrap around)
    "orange": [(11, 70, 50), (25, 255, 255)],
    "yellow": [(26, 70, 50), (35, 255, 255)],
    "green":  [(36, 50, 50), (86, 255, 255)],
    "blue":   [(90, 50, 50), (130, 255, 255)]
}

def bgr_to_hsv(bgr):
    """Convert a single BGR tuple (0-255) to HSV."""
    color = np.uint8([[bgr]])
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]
    return hsv

def hsv_to_color_name(hsv):
    """
    Decide the color name based on HSV values.
    Priority: White -> Black -> Colors
    """
    h, s, v = hsv

    # --- 1. PRIORITY CHECK: WHITE ---
    # White is defined by LOW Saturation. 
    # If S is low (< 80) and it's not pitch black (V > 60), it's White.
    if s < 80 and v > 60:
        return "white"

    # --- 2. PRIORITY CHECK: BLACK/VOID ---
    # If it's too dark, ignore it (likely the gap between stickers)
    if v < 50:
        return "unknown"

    # --- 3. CHECK COLORS ---
    for name, ranges in COLOR_RANGES.items():
        lower, upper = ranges
        l_h, l_s, l_v = lower
        u_h, u_s, u_v = upper
        
        if l_h <= h <= u_h and l_s <= s <= u_s and l_v <= v <= u_v:
            # Combine red1 and red2 into just "red"
            if "red" in name: return "red"
            return name

    return "unknown"

def get_blob_positions(roi_size=300, rows=3, cols=3):
    """Calculate the 9 grid points relative to the ROI."""
    step = roi_size // rows
    offset = step // 2
    centers = []
    
    for r in range(rows):
        for c in range(cols):
            cx = (c * step) + offset
            cy = (r * step) + offset
            centers.append((cx, cy))
            
    # Return centers and a dynamic radius based on box size
    return centers, step // 4 

def average_color_in_blob(img, center_roi, x_global, y_global, radius):
    """
    Compute average BGR color in a circular area.
    Arguments:
      img: The CLEAN image frame (no drawings on it)
      center_roi: (x,y) of the blob relative to the ROI box
      x_global, y_global: Top-left coordinates of the ROI box
    """
    cx, cy = center_roi
    # Calculate absolute coordinates on the full frame
    absolute_center = (x_global + cx, y_global + cy)
    
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.circle(mask, absolute_center, radius, 255, -1)
    
    # Calculate mean color only where the mask is white
    mean = cv2.mean(img, mask=mask)[:3]
    return tuple(map(int, mean))

def scan_cube(frame, roi_size):
    """Scans the 9 stickers and returns their color names."""
    h, w = frame.shape[:2]
    x0 = w//2 - roi_size//2
    y0 = h//2 - roi_size//2
    
    centers, radius = get_blob_positions(roi_size)
    detected_colors = []
    
    print("-" * 30)
    for i, center in enumerate(centers):
        # 1. Get raw color
        bgr = average_color_in_blob(frame, center, x0, y0, radius)
        hsv = bgr_to_hsv(bgr)
        
        # 2. Convert to name
        name = hsv_to_color_name(hsv)
        detected_colors.append(name)
        
        # Debug: Print the center sticker's raw data to help tuning
        if i == 4: 
            print(f"CENTER STICKER DATA -> HSV: {hsv} | Detected: {name.upper()}")

    print("-" * 30)
    return detected_colors

def solve_cube(cube_string):
    # Use the kociemba library to solve the Rubik's Cube
    solution = kociemba.solve(cube_string)
    return solution

def main():
    cam = cv2.VideoCapture(0)
    colors = []
    # Reduce size slightly so it fits better on screen
    roi_size = 300 
    current_face_colors = []
    
    print("Cube Scanner Started.")
    print("Press 's' to scan a face.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Flip horizontally for mirror effect (easier to aim)
        frame = cv2.flip(frame, 1)
        
        # IMPORTANT: Make a copy for the UI. 
        # We must keep 'frame' clean so we don't scan our own blue circles!
        ui_frame = frame.copy()
        
        h, w = ui_frame.shape[:2]
        x0 = w//2 - roi_size//2
        y0 = h//2 - roi_size//2
        x1, y1 = x0 + roi_size, y0 + roi_size

        # Draw the big green box
        cv2.rectangle(ui_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Draw the 9 small target circles
        centers, radius = get_blob_positions(roi_size)
        for cx, cy in centers:
            # Draw on ui_frame, NOT on frame
            cv2.circle(ui_frame, (x0+cx, y0+cy), radius, (255, 255, 255), 2)

        # Draw detection results
        if current_face_colors:
            # Format text nicely
            col1 = ", ".join(current_face_colors[:3])
            col2 = ", ".join(current_face_colors[3:6])
            col3 = ", ".join(current_face_colors[6:])
            
            cv2.putText(ui_frame, "Detected:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(ui_frame, col1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(ui_frame, col2, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(ui_frame, col3, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Cube Scanner", ui_frame)
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            # Scan the CLEAN frame
            current_face_colors = scan_cube(frame, roi_size)
            if(len(current_face_colors) != 9):
                print("Scan incomplete, press 's' again.")
                continue
            print("Captured Face:", current_face_colors)
            if(key == ord('y')):
                colors.append(current_face_colors)
                print("Face confirmed.")
                if len(colors) == 6:
                    print("All 6 faces scanned.")
                    print(solve_cube(colors))
                    break
            
        elif key == ord('q') or key == 27:
            print(colors) # q or ESC
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()