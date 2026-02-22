#pylint: disable=no-member
import cv2
import numpy as np
import kociemba

# --- CONFIGURATION ---

# Kociemba expects faces in order: U R F D L B
# Center sticker at position [4] defines the face color
# We map scanned color names to kociemba face letters
FACE_SCAN_ORDER = ["F", "U", "D", "B", "L", "R"]
FACE_INSTRUCTIONS = {
    # (face title,  color facing camera,  color on top,  action text)
    "F": ("FRONT face", "GREEN",  "YELLOW", "Green facing camera  |  Yellow on top"),
    "U": ("UP face",    "YELLOW", "BLUE",   "Tilt cube back: Yellow facing camera  |  Blue on top"),
    "D": ("DOWN face",  "WHITE",  "GREEN",  "Tilt cube forward: White facing camera  |  Green on top"),
    "B": ("BACK face",  "BLUE",   "YELLOW", "Rotate cube: Blue facing camera  |  Yellow on top"),
    "L": ("LEFT face",  "RED",    "YELLOW", "Rotate cube: Red facing camera  |  Yellow on top"),
    "R": ("RIGHT face", "ORANGE", "YELLOW", "Rotate cube: Orange facing camera  |  Yellow on top"),
}

# Color name → BGR for instruction badge highlights
FACE_LABEL_BGR = {
    "WHITE":  (255, 255, 255),
    "YELLOW": (0,   210, 210),
    "GREEN":  (0,   200,  40),
    "BLUE":   (220,  80,   0),
    "RED":    (0,    0,  220),
    "ORANGE": (0,   140, 255),
}
COLOR_TO_FACE = {
    "yellow":  "U",
    "orange":    "R",
    "green":  "F",
    "white": "D",
    "red": "L",
    "blue":   "B",
}
FACE_COLOR_BGR = {
    "D": (255, 255, 255),  # white
    "L": (0, 0, 200),      # red
    "F": (0, 180, 0),      # green
    "U": (0, 220, 220),    # yellow
    "R": (0, 100, 255),    # orange
    "B": (200, 80, 0),     # blue
}

COLOR_RANGES = {
    "red1":    [(0,   70,  50), (10,  255, 255)],
    "red2":    [(170, 70,  50), (180, 255, 255)],
    "orange1": [(7, 120, 120), (20, 255, 255)],   # normal orange
    "orange2": [(170, 150, 150), (179, 255, 255)],
    "yellow":  [(26,  70,  50), (35,  255, 255)],
    "green":   [(36,  50,  50), (86,  255, 255)],
    "blue":    [(90,  50,  50), (130, 255, 255)],
}

# ─── COLOR DETECTION ─────────────────────────────────────────────────────────

def bgr_to_hsv(bgr):
    color = np.uint8([[bgr]])
    return cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]

def hsv_to_color_name(hsv):
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if v < 40:
        return "unknown"
    if s < 70 and v > 80:
        return "white"
    for name, (lower, upper) in COLOR_RANGES.items():
        if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
            if "red" in name:
                return "red"
            if "orange" in name:
                return "orange"
            return name
    return "unknown"

def average_color_in_blob(img, abs_cx, abs_cy, radius):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.circle(mask, (abs_cx, abs_cy), radius, 255, -1)
    mean = cv2.mean(img, mask=mask)[:3]
    return tuple(map(int, mean))

def get_grid_centers(x0, y0, roi_size=300, rows=3, cols=3):
    step = roi_size // rows
    offset = step // 2
    centers = []
    for r in range(rows):
        for c in range(cols):
            cx = x0 + c * step + offset
            cy = y0 + r * step + offset
            centers.append((cx, cy))
    return centers, step // 4

def scan_face(frame, x0, y0, roi_size=300):
    centers, radius = get_grid_centers(x0, y0, roi_size)
    detected = []
    raw_bgr = []
    for cx, cy in centers:
        bgr = average_color_in_blob(frame, cx, cy, radius)
        hsv = bgr_to_hsv(bgr)
        name = hsv_to_color_name(hsv)
        detected.append(name)
        raw_bgr.append(bgr)
    return detected, centers, radius, raw_bgr

# ─── CUBE STRING BUILDER ──────────────────────────────────────────────────────

def build_cube_string(face_data):
    """
    Build the 54-char kociemba string from scanned face data.

    Kociemba identifies each face by its CENTER sticker, not by the label we
    assigned during scanning (which depends on how the user held the cube).
    So we route each scanned face into the kociemba slot that matches its
    center sticker color — ignoring the scan key entirely.

    Kociemba slot order: U R F D L B
    """
    # Build a lookup: kociemba_slot → 9-color list, keyed by center color
    slot_data: dict[str, list[str]] = {}
    for _scan_key, colors in face_data.items():
        center_color  = colors[4]                      # position 4 = center sticker
        kociemba_slot = COLOR_TO_FACE.get(center_color)
        if kociemba_slot is None:
            raise ValueError(
                f"Center sticker of a scanned face is '{center_color}' which is not a "
                f"recognised cube color. Check that the center sticker was detected correctly."
            )
        if kociemba_slot in slot_data:
            raise ValueError(
                f"Two scanned faces both have a '{center_color}' center sticker. "
                f"Each face must have a unique center color."
            )
        slot_data[kociemba_slot] = colors

    # Verify all 6 slots are present
    for slot in "URFDLB":
        if slot not in slot_data:
            color_name = {v: k for k, v in COLOR_TO_FACE.items()}[slot]
            raise ValueError(f"No scanned face has a '{color_name}' center sticker (slot {slot}).")

    # Assemble in kociemba order
    cube_str = ""
    for slot in ["U", "R", "F", "D", "L", "B"]:
        for color in slot_data[slot]:
            letter = COLOR_TO_FACE.get(color)
            if letter is None:
                raise ValueError(f"Unknown color '{color}' in face slot {slot}.")
            cube_str += letter
    return cube_str

def validate_cube_string(cube_str, face_data):
    """
    Check that each face letter appears exactly 9 times.
    Returns (is_valid, human_readable_error, bad_faces)
    bad_faces: list of face keys the user should re-scan.
    """
    from collections import Counter
    counts = Counter(cube_str)
    errors = []
    bad_faces = []

    # Map face letter → color name for readable messages
    letter_to_color = {v: k for k, v in COLOR_TO_FACE.items()}
    # Map face letter → which scan face it most likely belongs to
    # (over-counted letter → faces that contain too many of it)
    letter_to_scan_faces = {v: [] for v in COLOR_TO_FACE.values()}
    for face_key, colors in face_data.items():
        for color in colors:
            letter = COLOR_TO_FACE.get(color)
            if letter:
                letter_to_scan_faces[letter].append(face_key)

    for letter in "URFDLB":
        count = counts.get(letter, 0)
        if count != 9:
            color_name = letter_to_color.get(letter, letter)
            diff = count - 9
            direction = f"+{diff}" if diff > 0 else str(diff)
            errors.append(f"{color_name.upper()} ({letter}): found {count}/9  [{direction}]")
            # The faces that contributed too many/few of this letter are suspect
            for fk in letter_to_scan_faces[letter]:
                if fk not in bad_faces:
                    bad_faces.append(fk)

    if errors:
        msg = "Color count mismatch — each color must appear exactly 9 times:\n" + "\n".join(errors)
        return False, msg, bad_faces
    return True, "", []

# ─── DRAWING HELPERS ──────────────────────────────────────────────────────────

COLOR_NAME_BGR = {
    "white":   (255, 255, 255),
    "red":     (0,   0,   220),
    "orange":  (0,   140, 255),
    "yellow":  (0,   220, 220),
    "green":   (0,   200,  0),
    "blue":    (200, 80,   0),
    "unknown": (60,  60,   60),
}

def draw_mini_cube_face(ui_frame, colors, top_left, cell=28):
    """Draw a 3x3 colored grid of detected stickers."""
    tx, ty = top_left
    for i, c in enumerate(colors):
        r, col = divmod(i, 3)
        x1, y1 = tx + col * cell, ty + r * cell
        bgr = COLOR_NAME_BGR.get(c, (60, 60, 60))
        cv2.rectangle(ui_frame, (x1, y1), (x1+cell-2, y1+cell-2), bgr, -1)
        cv2.rectangle(ui_frame, (x1, y1), (x1+cell-2, y1+cell-2), (30, 30, 30), 1)

def draw_solution_overlay(ui_frame, solution_moves, current_move_idx):
    """Draw solution steps on the bottom of the frame."""
    h, w = ui_frame.shape[:2]
    overlay = ui_frame.copy()
    cv2.rectangle(overlay, (0, h-120), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, ui_frame, 0.25, 0, ui_frame)

    total = len(solution_moves)
    if total == 0:
        cv2.putText(ui_frame, "Cube is already solved!", (10, h-80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return

    done = current_move_idx
    remaining = total - done

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 10, h-20, w-20, 12
    cv2.rectangle(ui_frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (80,80,80), -1)
    filled = int(bar_w * done / total) if total > 0 else 0
    cv2.rectangle(ui_frame, (bar_x, bar_y), (bar_x+filled, bar_y+bar_h), (0,220,100), -1)
    cv2.putText(ui_frame, f"{done}/{total}", (bar_x+bar_w//2-20, bar_y+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # Current move (big)
    if done < total:
        move = solution_moves[done]
        cv2.putText(ui_frame, f"Move {done+1}: {move}", (10, h-80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)

    # Next few moves
    preview = solution_moves[done+1:done+5]
    preview_text = "  Next: " + "  ".join(preview) if preview else "  (last move)"
    cv2.putText(ui_frame, preview_text, (10, h-50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

    cv2.putText(ui_frame, f"Remaining: {remaining} moves | [N] next  [P] prev  [R] restart",
                (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

def draw_color_pill(img, text, color_bgr, x, y, font_scale=0.52, pad_x=10, pad_y=6):
    """Draw a rounded-rect pill badge with colored background. Returns right-edge x."""
    thickness = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, thickness, font_scale, 2)
    x1, y1 = x, y - th - pad_y
    x2, y2 = x + tw + pad_x * 2, y + pad_y
    # Filled rect (cv2 has no rounded rect fill before 4.x so we stack rects + circles)
    cv2.rectangle(img, (x1 + 6, y1), (x2 - 6, y2), color_bgr, -1)
    cv2.rectangle(img, (x1, y1 + 6), (x2, y2 - 6), color_bgr, -1)
    cv2.circle(img, (x1 + 6, y1 + 6), 6, color_bgr, -1)
    cv2.circle(img, (x2 - 6, y1 + 6), 6, color_bgr, -1)
    cv2.circle(img, (x1 + 6, y2 - 6), 6, color_bgr, -1)
    cv2.circle(img, (x2 - 6, y2 - 6), 6, color_bgr, -1)
    # Text color: dark on light backgrounds
    lum = 0.299 * color_bgr[2] + 0.587 * color_bgr[1] + 0.114 * color_bgr[0]
    txt_color = (20, 20, 20) if lum > 140 else (240, 240, 240)
    cv2.putText(img, text, (x1 + pad_x, y), thickness, font_scale, txt_color, 2)
    return x2 + 8  # gap after pill

def draw_scanning_ui(ui_frame, face_idx, current_face_colors, centers, radius, x0, y0):
    """Draw scanning overlay: grid dots, rich instruction banner, mini-preview."""
    h, w = ui_frame.shape[:2]
    face_key = FACE_SCAN_ORDER[face_idx]
    title, cam_color, top_color, action_text = FACE_INSTRUCTIONS[face_key]

    # ── Banner background (tall enough for 3 lines + pills) ──────────────────
    BANNER_H = 100
    overlay = ui_frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, BANNER_H), (18, 18, 28), -1)
    cv2.addWeighted(overlay, 0.88, ui_frame, 0.12, 0, ui_frame)
    # Accent left bar in the camera-facing color
    accent_bgr = FACE_LABEL_BGR.get(cam_color, (180, 180, 180))
    cv2.rectangle(ui_frame, (0, 0), (5, BANNER_H), accent_bgr, -1)

    # ── Line 1: Face counter + title ─────────────────────────────────────────
    counter_text = f"Face {face_idx+1} / 6"
    cv2.putText(ui_frame, counter_text, (14, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (130, 130, 130), 1)
    cv2.putText(ui_frame, title.upper(), (14 + 90, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 240, 255), 2)

    # ── Line 2: "FACING CAMERA" pill + label ─────────────────────────────────
    cv2.putText(ui_frame, "FACING CAMERA:", (14, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1)
    cam_bgr = FACE_LABEL_BGR.get(cam_color, (180, 180, 180))
    draw_color_pill(ui_frame, cam_color, cam_bgr, 155, 52)

    # ── Line 3: "ON TOP" pill + label ────────────────────────────────────────
    cv2.putText(ui_frame, "ON TOP:        ", (14, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1)
    top_bgr = FACE_LABEL_BGR.get(top_color, (180, 180, 180))
    draw_color_pill(ui_frame, top_color, top_bgr, 155, 80)

    # ── Action hint (right side of banner) ───────────────────────────────────
    (aw, _), _ = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    ax = w - aw - 14
    cv2.putText(ui_frame, action_text, (ax, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 180, 120), 1)

    # ── ROI box ───────────────────────────────────────────────────────────────
    cv2.rectangle(ui_frame, (x0, y0), (x0 + 300, y0 + 300), accent_bgr, 2)

    # ── Grid sampling circles ─────────────────────────────────────────────────
    for cx, cy in centers:
        cv2.circle(ui_frame, (cx, cy), radius, (255, 255, 255), 2)

    # ── Mini-preview of locked scan (bottom-right) ────────────────────────────
    if current_face_colors:
        draw_mini_cube_face(ui_frame, current_face_colors, (w - 100, h - 115))
        cv2.putText(ui_frame, "Locked scan", (w - 106, h - 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # ── Saved faces progress dots ─────────────────────────────────────────────
    dot_y = BANNER_H + 18
    cv2.putText(ui_frame, "Saved:", (14, dot_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (130, 130, 130), 1)
    for i, fk in enumerate(FACE_SCAN_ORDER):
        cx_dot = 75 + i * 30
        done = i < face_idx
        current = i == face_idx
        dot_bgr = FACE_COLOR_BGR.get(fk, (80, 80, 80))
        if done:
            cv2.circle(ui_frame, (cx_dot, dot_y - 5), 11, dot_bgr, -1)
            cv2.circle(ui_frame, (cx_dot, dot_y - 5), 11, (0, 0, 0), 1)
            # checkmark
            cv2.putText(ui_frame, fk, (cx_dot - 5, dot_y - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        elif current:
            cv2.circle(ui_frame, (cx_dot, dot_y - 5), 11, dot_bgr, 2)
            cv2.circle(ui_frame, (cx_dot, dot_y - 5), 5, dot_bgr, -1)
        else:
            cv2.circle(ui_frame, (cx_dot, dot_y - 5), 11, (50, 50, 50), -1)
            cv2.circle(ui_frame, (cx_dot, dot_y - 5), 11, (80, 80, 80), 1)

    # ── Bottom controls bar ───────────────────────────────────────────────────
    overlay2 = ui_frame.copy()
    cv2.rectangle(overlay2, (0, h - 30), (w, h), (18, 18, 28), -1)
    cv2.addWeighted(overlay2, 0.8, ui_frame, 0.2, 0, ui_frame)
    cv2.putText(ui_frame, "[S] Scan    [E] Edit colors    [Y] Confirm & save    [Q] Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 160), 1)

def draw_live_color_dots(ui_frame, centers, raw_bgr):
    """Overlay small dots showing detected BGR color on each sampling point."""
    for (cx, cy), bgr in zip(centers, raw_bgr):
        cv2.circle(ui_frame, (cx, cy), 7, bgr, -1)
        cv2.circle(ui_frame, (cx, cy), 7, (0, 0, 0), 1)

# ─── MANUAL OVERRIDE ──────────────────────────────────────────────────────────

COLOR_CYCLE = ["white", "red", "orange", "yellow", "green", "blue", "unknown"]
# Key shortcuts when a sticker is selected in edit mode
KEY_COLOR_MAP = {
    ord('w'): "white",
    ord('o'): "orange",
    ord('g'): "green",
    ord('b'): "blue",
}

def find_clicked_sticker(mx, my, centers, radius):
    """Return index of the sticker whose circle contains (mx, my), else -1."""
    hit_radius = max(radius + 10, 20)
    for i, (cx, cy) in enumerate(centers):
        if (mx - cx) ** 2 + (my - cy) ** 2 <= hit_radius ** 2:
            return i
    return -1

def draw_edit_overlay(ui_frame, colors, centers, radius, selected_idx):
    """Draw the edit-mode UI on top of the locked scan."""
    h, w = ui_frame.shape[:2]

    # Dark top banner
    overlay = ui_frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (40, 10, 10), -1)
    cv2.addWeighted(overlay, 0.82, ui_frame, 0.18, 0, ui_frame)
    cv2.putText(ui_frame, "EDIT MODE — Click a sticker, then press key to change color",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 200, 255), 2)
    cv2.putText(ui_frame, "W=white  R=red  O=orange  Y=yellow  G=green  B=blue",
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    cv2.putText(ui_frame, "Scroll wheel over sticker to cycle  |  [E] done editing  [S] rescan",
                (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    # Draw all sticker circles with current color
    for i, (cx, cy) in enumerate(centers):
        name = colors[i]
        bgr  = COLOR_NAME_BGR.get(name, (60, 60, 60))
        is_selected = (i == selected_idx)
        ring_thick = 4 if is_selected else 2
        ring_color = (0, 255, 255) if is_selected else (30, 30, 30)
        inner_r = radius + 6
        cv2.circle(ui_frame, (cx, cy), inner_r, ring_color, ring_thick)
        cv2.circle(ui_frame, (cx, cy), inner_r - ring_thick, bgr, -1)

        # Color initial label
        initial = name[0].upper() if name != "unknown" else "?"
        font_color = (0, 0, 0) if name in ("white", "yellow", "orange") else (240, 240, 240)
        cv2.putText(ui_frame, initial, (cx - 6, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, font_color, 2)

    # Bottom status
    if selected_idx >= 0:
        sel_name = colors[selected_idx]
        sel_bgr  = COLOR_NAME_BGR.get(sel_name, (60, 60, 60))
        cv2.putText(ui_frame, f"Selected sticker #{selected_idx + 1}: {sel_name.upper()}   (press key to change)",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sel_bgr, 2)
    else:
        cv2.putText(ui_frame, "Click a sticker to select it",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 150), 1)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    cam = cv2.VideoCapture(0)
    roi_size = 300

    STATE_SCANNING = "scanning"
    STATE_EDITING  = "editing"
    STATE_SOLUTION = "solution"
    STATE_ERROR    = "error"

    state            = STATE_SCANNING
    face_idx         = 0
    face_data        = {}
    current_colors   = []
    current_centers  = []
    current_radius   = 10
    current_raw_bgr  = []

    solution_moves   = []
    current_move_idx = 0
    error_msg        = ""
    error_faces      = []   # face keys flagged as likely bad during validation

    # Edit-mode state
    edit_colors      = []   # mutable copy of colors being edited
    edit_selected    = -1   # which sticker is selected (-1 = none)
    mouse_event_buf  = []   # [(event, x, y, flags)]

    WINDOW = "Rubik's Cube Solver"
    cv2.namedWindow(WINDOW)

    def on_mouse(event, mx, my, flags, param):
        mouse_event_buf.append((event, mx, my, flags))

    cv2.setMouseCallback(WINDOW, on_mouse)

    print("=== Rubik's Cube Solver ===")
    print("S=scan  Y=confirm  E=edit scanned face  Q=quit")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        ui_frame = frame.copy()
        h, w     = ui_frame.shape[:2]
        x0 = w // 2 - roi_size // 2
        y0 = h // 2 - roi_size // 2

        # ── Drain mouse buffer ────────────────────────────────────────────────
        events = mouse_event_buf[:]
        mouse_event_buf.clear()

        # ── SCANNING STATE ────────────────────────────────────────────────────
        if state == STATE_SCANNING:
            live_colors, live_centers, live_radius, live_raw = scan_face(frame, x0, y0, roi_size)
            draw_live_color_dots(ui_frame, live_centers, live_raw)

            draw_scanning_ui(ui_frame, face_idx, current_colors,
                             live_centers, live_radius, x0, y0)

            if current_colors:
                for i, (cx, cy) in enumerate(current_centers):
                    name = current_colors[i]
                    bgr  = COLOR_NAME_BGR.get(name, (60, 60, 60))
                    cv2.circle(ui_frame, (cx, cy), current_radius + 3, bgr, -1)
                    cv2.circle(ui_frame, (cx, cy), current_radius + 3, (0, 0, 0), 1)

        # ── EDITING STATE ─────────────────────────────────────────────────────
        elif state == STATE_EDITING:
            # Process mouse events
            for (event, mx, my, flags) in events:
                if event == cv2.EVENT_LBUTTONDOWN:
                    idx = find_clicked_sticker(mx, my, current_centers, current_radius)
                    edit_selected = idx  # -1 if clicked empty space
                elif event == cv2.EVENT_MOUSEWHEEL:
                    # Scroll over a sticker cycles its color
                    idx = find_clicked_sticker(mx, my, current_centers, current_radius)
                    if idx >= 0:
                        edit_selected = idx
                        cur = COLOR_CYCLE.index(edit_colors[idx]) if edit_colors[idx] in COLOR_CYCLE else 0
                        direction = 1 if flags > 0 else -1
                        edit_colors[idx] = COLOR_CYCLE[(cur + direction) % len(COLOR_CYCLE)]

            draw_edit_overlay(ui_frame, edit_colors, current_centers, current_radius, edit_selected)

        # ── SOLUTION STATE ────────────────────────────────────────────────────
        elif state == STATE_SOLUTION:
            draw_solution_overlay(ui_frame, solution_moves, current_move_idx)
            cv2.putText(ui_frame, "SOLUTION MODE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ── ERROR STATE ───────────────────────────────────────────────────────
        elif state == STATE_ERROR:
            overlay = ui_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 60), -1)
            cv2.addWeighted(overlay, 0.55, ui_frame, 0.45, 0, ui_frame)

            # Header
            cv2.rectangle(ui_frame, (0, 0), (w, 50), (0, 0, 160), -1)
            cv2.putText(ui_frame, "INVALID CUBE — Cannot solve", (14, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 80, 80), 2)

            # Error lines
            y_cur = 80
            for line in error_msg.split("\n"):
                color = (255, 160, 80) if ":" in line and "/" in line else (210, 210, 210)
                cv2.putText(ui_frame, line, (20, y_cur),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
                y_cur += 26

            # Suspect faces callout
            if error_faces:
                y_cur += 10
                cv2.putText(ui_frame, "Likely mis-scanned faces:", (20, y_cur),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 80), 2)
                y_cur += 28
                for fk in error_faces:
                    fi_title, fi_cam, fi_top, _ = FACE_INSTRUCTIONS[fk]
                    cam_bgr = FACE_LABEL_BGR.get(fi_cam, (180, 180, 180))
                    top_bgr = FACE_LABEL_BGR.get(fi_top, (180, 180, 180))
                    # Face name
                    cv2.putText(ui_frame, f"  {fi_title}:", (20, y_cur),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
                    # Camera pill
                    nx = draw_color_pill(ui_frame, fi_cam, cam_bgr, 175, y_cur, font_scale=0.46)
                    cv2.putText(ui_frame, "facing camera,", (nx, y_cur),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1)
                    nx2 = nx + 125
                    draw_color_pill(ui_frame, fi_top, top_bgr, nx2, y_cur, font_scale=0.46)
                    cv2.putText(ui_frame, "on top", (nx2 + len(fi_top) * 10 + 22, y_cur),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1)
                    y_cur += 30

            # Instructions
            cv2.rectangle(ui_frame, (0, h - 40), (w, h), (30, 30, 30), -1)
            cv2.putText(ui_frame, "[R] Re-scan suspect faces    [F] Full restart    [Q] Quit",
                        (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow(WINDOW, ui_frame)
        key = cv2.waitKey(1) & 0xFF

        # ── GLOBAL KEYS ───────────────────────────────────────────────────────
        if key == ord('q') or key == 27:
            break

        # ── SCANNING KEYS ─────────────────────────────────────────────────────
        if state == STATE_SCANNING:
            if key == ord('s'):
                scanned, centers, radius, raw = scan_face(frame, x0, y0, roi_size)
                unknown_count = scanned.count("unknown")
                if unknown_count > 2:
                    print(f"[WARN] Too many unknowns ({unknown_count}). Improve lighting and retry.")
                else:
                    current_colors  = scanned
                    current_centers = centers
                    current_radius  = radius
                    current_raw_bgr = raw
                    face_key = FACE_SCAN_ORDER[face_idx]
                    print(f"Face {face_idx+1} ({face_key}) scanned: {scanned}")
                    print("Y=confirm  E=edit colors  S=rescan")

            elif key == ord('e'):
                if not current_colors:
                    print("[WARN] Scan a face first (press S).")
                else:
                    edit_colors   = current_colors[:]
                    edit_selected = -1
                    state = STATE_EDITING
                    print("Edit mode. Click sticker + press key, or scroll wheel to cycle.")

            elif key == ord('y'):
                if not current_colors:
                    print("[WARN] No scan yet. Press S first.")
                else:
                    face_key = FACE_SCAN_ORDER[face_idx]
                    face_data[face_key] = current_colors[:]
                    print(f"Face {face_key} confirmed: {current_colors}")
                    current_colors  = []
                    current_centers = []
                    current_raw_bgr = []
                    face_idx += 1

                    if face_idx == 6:
                        print("All 6 faces captured. Validating...")
                        try:
                            cube_str = build_cube_string(face_data)
                            print(f"Cube string: {cube_str}")

                            # ── Pre-validate before calling kociemba ──────────
                            is_valid, val_msg, bad_faces = validate_cube_string(cube_str, face_data)
                            if not is_valid:
                                print(f"[INVALID] {val_msg}")
                                print(f"Suspect faces to re-scan: {bad_faces}")
                                error_msg   = val_msg
                                error_faces = bad_faces  # faces user should re-scan
                                state       = STATE_ERROR
                            else:
                                raw_sol = kociemba.solve(cube_str)
                                print(f"Solution: {raw_sol}")
                                solution_moves   = raw_sol.strip().split()
                                current_move_idx = 0
                                state            = STATE_SOLUTION
                        except Exception as e:
                            error_msg   = str(e)
                            error_faces = []
                            print(f"[ERROR] {e}")
                            state = STATE_ERROR

        # ── EDITING KEYS ──────────────────────────────────────────────────────
        elif state == STATE_EDITING:
            # Color keys — apply to selected sticker
            if edit_selected >= 0:
                # Direct letter shortcuts
                color_for_key = KEY_COLOR_MAP.get(key)

                # Y and R are special: in edit mode only, map them to yellow/red
                if key == ord('y'):
                    color_for_key = "yellow"
                elif key == ord('r'):
                    color_for_key = "red"

                if color_for_key:
                    edit_colors[edit_selected] = color_for_key
                    print(f"Sticker #{edit_selected+1} → {color_for_key}")

                # Arrow keys or Tab to move selection
                elif key == 9:  # Tab
                    edit_selected = (edit_selected + 1) % 9
                elif key == 82 or key == ord('k'):  # up arrow / k
                    edit_selected = (edit_selected - 3) % 9
                elif key == 84 or key == ord('j'):  # down arrow / j
                    edit_selected = (edit_selected + 3) % 9
                elif key == 81 or key == ord('h'):  # left arrow / h
                    edit_selected = (edit_selected - 1) % 9
                elif key == 83 or key == ord('l'):  # right arrow / l
                    edit_selected = (edit_selected + 1) % 9

            # E = done editing → apply and return to scanning state
            if key == ord('e'):
                current_colors = edit_colors[:]
                state = STATE_SCANNING
                print(f"Edits applied: {current_colors}")
                print("Y=confirm  S=rescan  E=edit again")

            # S = discard edits, rescan
            elif key == ord('s'):
                edit_colors   = []
                edit_selected = -1
                state = STATE_SCANNING
                current_colors = []
                print("Rescan mode.")

        # ── SOLUTION KEYS ─────────────────────────────────────────────────────
        elif state == STATE_SOLUTION:
            if key == ord('n') or key == ord(' '):
                if current_move_idx < len(solution_moves):
                    current_move_idx += 1
            elif key == ord('p'):
                if current_move_idx > 0:
                    current_move_idx -= 1
            elif key == ord('r'):
                state = STATE_SCANNING; face_idx = 0; face_data = {}
                current_colors = []; solution_moves = []; current_move_idx = 0
                print("Restarting scan...")

        # ── ERROR KEYS ────────────────────────────────────────────────────────
        elif state == STATE_ERROR:
            if key == ord('r'):
                # Jump back to re-scan the first suspect face
                if error_faces:
                    target = error_faces[0]
                    face_idx = FACE_SCAN_ORDER.index(target)
                    print(f"Re-scanning from face {target} ({FACE_INSTRUCTIONS[target][0]})")
                else:
                    face_idx = 0
                state          = STATE_SCANNING
                current_colors = []
                error_msg      = ""
                error_faces    = []
            elif key == ord('f'):
                # Full restart — wipe everything
                state     = STATE_SCANNING
                face_idx  = 0
                face_data = {}
                current_colors = []
                error_msg      = ""
                error_faces    = []
                print("Full restart.")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
