import pygame, sys, random, os, json, time, math, numpy as np
from PIL import Image

pygame.init()
try:
    snap_sound = pygame.mixer.Sound("snap.mp3")
except Exception:
    snap_sound = None

# --- Global Settings ---
SCREEN_WIDTH, SCREEN_HEIGHT = 2400, 1350
FPS = 60
GROUP_SNAP_THRESHOLD = 20
SOLVED_WIDTH = int(SCREEN_WIDTH * 0.5)
SOLVED_HEIGHT = int(SCREEN_HEIGHT * 0.5)
SOLVED_RECT = pygame.Rect((SCREEN_WIDTH - SOLVED_WIDTH)//2,
                          (SCREEN_HEIGHT - SOLVED_HEIGHT)//2,
                          SOLVED_WIDTH, SOLVED_HEIGHT)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Jigsaw Puzzle")
clock = pygame.time.Clock()

# --- Game State & Puzzle Settings ---
game_state = "menu_main"  # possible: menu_main, menu_new, menu_load, puzzle
new_puzzle_settings = {"pieces": 300, "image_list": [], "image_index": 0}
if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("completed"):
    os.makedirs("completed")
if not os.path.exists("saves"):
    os.makedirs("saves")
new_puzzle_settings["image_list"] = [f for f in os.listdir("images")
                                     if f.lower().endswith(('.png','.jpg','.jpeg'))]
if not new_puzzle_settings["image_list"]:
    print("No images found in 'images' folder. Please add some images and restart.")
    pygame.quit(); sys.exit()
if not os.path.exists("saves"):
    os.makedirs("saves")
load_puzzle_list = [f for f in os.listdir("saves") if f.lower().endswith(".json")]

image_file = None
rows = None
cols = None
cell_width = None
cell_height = None
tab_size = None
piece_edges = None
pieces = []
groups = []
dragging_group = None
drag_offsets = {}
puzzle_solved = False
puzzle_auto_saved = False

# Global to store the original input image (full resolution) and the scaled puzzle area.
original_img = None
puzzle_area_rect = None

# --- Global Caches for Performance ---
FONTS = {}          # Cache for fonts keyed by size.
IMAGE_CACHE = {}    # Cache for loaded images keyed by filename.

def get_font(size):
    """Return a cached font of the given size."""
    if size not in FONTS:
        FONTS[size] = pygame.font.SysFont("arial", size)
    return FONTS[size]

def load_image_cached(path):
    """Return a cached loaded image (with convert_alpha) for the given path."""
    if path not in IMAGE_CACHE:
        IMAGE_CACHE[path] = pygame.image.load(path).convert_alpha()
    return IMAGE_CACHE[path]

# --- Helper Functions ---
def draw_text(text, pos, font_size=30, color=(255,255,255), shadow_color=(0,0,0), shadow_offset=(2,2)):
    """Draws text with a subtle drop shadow for improved legibility."""
    font = get_font(font_size)
    # Draw shadow:
    shadow_surface = font.render(text, True, shadow_color)
    shadow_rect = shadow_surface.get_rect(center=(pos[0]+shadow_offset[0], pos[1]+shadow_offset[1]))
    screen.blit(shadow_surface, shadow_rect)
    # Draw main text:
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=pos)
    screen.blit(text_surface, text_rect)

def draw_rounded_button(button, mouse_pos=None):
    """Draws a button with a border and a subtle text shadow."""
    rect = button["rect"]
    base_color = button["color"]
    # Lighten on hover:
    if mouse_pos and rect.collidepoint(mouse_pos):
        color = tuple(min(255, c + 30) for c in base_color)
    else:
        color = base_color
    # Draw a black border around the button:
    border_rect = rect.inflate(4, 4)
    pygame.draw.rect(screen, (0, 0, 0), border_rect, border_radius=8)
    pygame.draw.rect(screen, color, rect, border_radius=8)
    font = get_font(28)
    # Draw text shadow:
    shadow_offset = (1, 1)
    shadow_surface = font.render(button["label"], True, (0, 0, 0))
    shadow_rect = shadow_surface.get_rect(center=(rect.centerx + shadow_offset[0], rect.centery + shadow_offset[1]))
    screen.blit(shadow_surface, shadow_rect)
    # Draw the main label:
    text_surface = font.render(button["label"], True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)


def get_mouse_pos():
    return pygame.mouse.get_pos()

# --- Fade Transition ---
def fade_in(duration=250):
    fade = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    fade.fill((0,0,0))
    # The fade surface is created once and then used in the loop.
    for alpha in range(255, -1, -5):
        fade.set_alpha(alpha)
        redraw_all()
        screen.blit(fade, (0,0))
        pygame.display.update()
        pygame.time.delay(duration // 51)

# --- Helper: Scale an Image to Fit a Target Rect While Preserving Its Aspect Ratio ---
def scale_image_preserve_aspect(image, target_rect):
    orig_width, orig_height = image.get_width(), image.get_height()
    target_width, target_height = target_rect.width, target_rect.height
    scale_factor = min(target_width / orig_width, target_height / orig_height)
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)
    scaled_image = pygame.transform.smoothscale(image, (new_width, new_height))
    new_x = target_rect.x + (target_rect.width - new_width) // 2
    new_y = target_rect.y + (target_rect.height - new_height) // 2
    new_rect = pygame.Rect(new_x, new_y, new_width, new_height)
    return scaled_image, new_rect

# --- UI Drawing Functions ---
def redraw_all():
    if game_state=="menu_main":
        draw_main_menu()
    elif game_state=="menu_new":
        draw_new_menu()
    elif game_state=="menu_load":
        draw_load_menu()
    elif game_state=="puzzle":
        draw_puzzle()

def draw_background_gradient():
    """Draws a vertical gradient background for a more polished look."""
    color_top = (30, 30, 30)
    color_bottom = (60, 60, 60)
    for y in range(SCREEN_HEIGHT):
        ratio = y / SCREEN_HEIGHT
        r = int(color_top[0] * (1 - ratio) + color_bottom[0] * ratio)
        g = int(color_top[1] * (1 - ratio) + color_bottom[1] * ratio)
        b = int(color_top[2] * (1 - ratio) + color_bottom[2] * ratio)
        pygame.draw.line(screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))

def draw_header(text, pos, font_size, main_color, outline_color, outline_thickness=2):
    """
    Draws a header with a bold outline to give it a polished, official look.
    """
    font = get_font(font_size)
    # Draw an outline by rendering the text multiple times offset in all directions.
    for dx in range(-outline_thickness, outline_thickness + 1):
        for dy in range(-outline_thickness, outline_thickness + 1):
            if dx == 0 and dy == 0:
                continue
            outline_surface = font.render(text, True, outline_color)
            outline_rect = outline_surface.get_rect(center=(pos[0] + dx, pos[1] + dy))
            screen.blit(outline_surface, outline_rect)
    # Draw the main text on top.
    main_surface = font.render(text, True, main_color)
    main_rect = main_surface.get_rect(center=pos)
    screen.blit(main_surface, main_rect)

def draw_main_menu():
    """Draws the main menu with a gradient background and styled text/buttons."""
    global puzzle_solved
    puzzle_solved = False
    draw_background_gradient()  # Use our new gradient background.
    draw_header("Jigsaw Madness", (SCREEN_WIDTH // 2, 100), font_size=96, main_color=(255, 215, 0), outline_color=(0, 0, 0), outline_thickness=3)
    buttons = [
        {"label": "New Puzzle", "rect": pygame.Rect(SCREEN_WIDTH // 2 - 100, 250, 200, 50), "color": (70, 130, 180)},
        {"label": "Load Puzzle", "rect": pygame.Rect(SCREEN_WIDTH // 2 - 100, 320, 200, 50), "color": (70, 130, 180)},
        {"label": "Exit", "rect": pygame.Rect(SCREEN_WIDTH // 2 - 100, 390, 200, 50), "color": (178, 34, 34)}
    ]
    for btn in buttons:
        draw_rounded_button(btn, get_mouse_pos())
    return {btn["label"]: btn for btn in buttons}

def get_candidate_configs(desired, aspect_ratio, delta=4):
    rows_est = math.sqrt(desired / aspect_ratio)
    candidates = []
    start = max(1, int(math.floor(rows_est)) - delta)
    end = int(math.floor(rows_est)) + delta
    for r in range(start, end+1):
        c = max(1, round(r * aspect_ratio))
        total = r * c
        diff = abs(total - desired)
        candidates.append((r, c, total, diff))
    candidates.sort(key=lambda x: (x[3], x[2]))
    return candidates

def draw_new_menu():
    """Draws the New Puzzle menu using the new background and updated text/buttons."""
    draw_background_gradient()
    center_x = SCREEN_WIDTH // 2
    y_offset = 50
    # Draw a bold header for the New Puzzle screen.
    draw_header("New Puzzle", (center_x, y_offset), font_size=64, main_color=(0, 255, 0), outline_color=(0, 0, 0), outline_thickness=2)
    buttons = {}
    # Pieces controls:
    pieces_y = y_offset + 80
    btn_width = 40
    btn_height = 40
    buttons["pieces_dec"] = {"label": "-", "rect": pygame.Rect(center_x - 150, pieces_y, btn_width, btn_height), "color": (100, 100, 100)}
    buttons["pieces_inc"] = {"label": "+", "rect": pygame.Rect(center_x + 110, pieces_y, btn_width, btn_height), "color": (100, 100, 100)}
    draw_rounded_button(buttons["pieces_dec"], get_mouse_pos())
    draw_rounded_button(buttons["pieces_inc"], get_mouse_pos())
    draw_text(f"Pieces: {new_puzzle_settings['pieces']}", (center_x, pieces_y + 20), font_size=36)
    # Preset options:
    preset_options = [20, 50, 100, 200, 300, 500]
    preset_buttons = {}
    preset_button_width = 80
    preset_button_height = 40
    spacing = 10
    total_width = len(preset_options) * preset_button_width + (len(preset_options) - 1) * spacing
    start_x = center_x - total_width // 2
    preset_y = pieces_y + 50
    for i, opt in enumerate(preset_options):
        rect = pygame.Rect(start_x + i * (preset_button_width + spacing), preset_y, preset_button_width, preset_button_height)
        key = f"preset_{opt}"
        preset_buttons[key] = {"label": str(opt), "rect": rect, "color": (150, 150, 150), "piece_count": opt}
        draw_rounded_button(preset_buttons[key], get_mouse_pos())
    buttons.update(preset_buttons)
    # Image picker:
    arrow_size = 50
    spacing = 100
    img_y = preset_y + 70
    text_rect = pygame.Rect(center_x - 150, img_y, 300, arrow_size)
    left_button_rect = pygame.Rect(text_rect.left - arrow_size - spacing, img_y, arrow_size, arrow_size)
    right_button_rect = pygame.Rect(text_rect.right + spacing, img_y, arrow_size, arrow_size)
    buttons["img_left"] = {"label": "<", "rect": left_button_rect, "color": (100, 100, 100)}
    buttons["img_right"] = {"label": ">", "rect": right_button_rect, "color": (100, 100, 100)}
    draw_rounded_button(buttons["img_left"], get_mouse_pos())
    draw_rounded_button(buttons["img_right"], get_mouse_pos())
    current_img = new_puzzle_settings["image_list"][new_puzzle_settings["image_index"]]
    draw_text(f"Image: {current_img}", text_rect.center, font_size=36)
    # Candidate configurations:
    draw_text("Select Puzzle Configuration:", (center_x, img_y + 150), font_size=32)
    candidate_configs = get_candidate_configs(new_puzzle_settings["pieces"], 
                                               load_image_cached(os.path.join("images", current_img)).get_width() /
                                               load_image_cached(os.path.join("images", current_img)).get_height(),
                                               delta=4)
    candidate_buttons = {}
    num_candidates = len(candidate_configs)
    num_columns = 2 if num_candidates > 4 else 1
    button_width = 250
    button_height = 50
    col_spacing = 20
    row_spacing = 10
    if num_columns == 2:
        col1_x = center_x - button_width - col_spacing // 2
        col2_x = center_x + col_spacing // 2
    else:
        col1_x = center_x - button_width // 2
    for i, (r, c, total, diff) in enumerate(candidate_configs):
        col = i % num_columns
        row = i // num_columns
        candidate_x = col1_x if col == 0 else col2_x
        candidate_y = img_y + 170 + row * (button_height + row_spacing)
        btn_rect = pygame.Rect(candidate_x, candidate_y, button_width, button_height)
        key = f"candidate_{i}"
        candidate_buttons[key] = {"label": f"{r} x {c} = {total} pieces", "rect": btn_rect, "color": (100, 100, 100), "config": (r, c)}
        draw_rounded_button(candidate_buttons[key], get_mouse_pos())
    buttons.update(candidate_buttons)
    # Back button:
    buttons["back"] = {"label": "Back", "rect": pygame.Rect(20, SCREEN_HEIGHT - 70, 100, 40), "color": (105, 105, 105)}
    draw_rounded_button(buttons["back"], get_mouse_pos())
    return buttons

def draw_load_menu():
    """Draws the Load Puzzle menu with the new gradient background and styled buttons."""
    draw_background_gradient()
    draw_header("Load Puzzle", (SCREEN_WIDTH // 2, 50), font_size=64, main_color=(0, 255, 255), outline_color=(0, 0, 0), outline_thickness=2)
    buttons = {}
    start_y = 150
    global load_puzzle_list
    load_puzzle_list = [f for f in os.listdir("saves") if f.lower().endswith(".json")]
    for i, fname in enumerate(load_puzzle_list):
        btn = {"label": fname, "rect": pygame.Rect(200, start_y + i * 60, 600, 50), "color": (70, 130, 180)}
        draw_rounded_button(btn, get_mouse_pos())
        buttons[f"save_{i}"] = btn
    buttons["back"] = {"label": "Back", "rect": pygame.Rect(20, SCREEN_HEIGHT - 70, 100, 40), "color": (105, 105, 105)}
    draw_rounded_button(buttons["back"], get_mouse_pos())
    return buttons

def draw_puzzle():
    """Draws the puzzle screen. (The game logic is unchanged, but the background now uses the gradient.)"""
    draw_background_gradient()
    for p in pieces:
        screen.blit(p["image"], p["rect"])
    menu_btn = {"label": "Main Menu", "rect": pygame.Rect(20, 20, 150, 40), "color": (128, 128, 128)}
    save_btn = {"label": "Save Puzzle", "rect": pygame.Rect(20, 70, 150, 40), "color": (34, 139, 34)}
    draw_rounded_button(menu_btn, get_mouse_pos())
    draw_rounded_button(save_btn, get_mouse_pos())
    return {"menu": menu_btn, "save": save_btn}

# --- Puzzle Piece Generation Functions ---
def point_in_rect(point, rect):
    return rect.collidepoint(point)

def get_random_position_outside(solved_rect, piece_width, piece_height):
    candidates = []
    if solved_rect.x - piece_width >= 0:
        candidates.append((0, solved_rect.x - piece_width, 0, SCREEN_HEIGHT - piece_height))
    if SCREEN_WIDTH - solved_rect.right >= piece_width:
        candidates.append((solved_rect.right, SCREEN_WIDTH - piece_width, 0, SCREEN_HEIGHT - piece_height))
    if solved_rect.y - piece_height >= 0:
        candidates.append((0, SCREEN_WIDTH - piece_width, 0, solved_rect.y - piece_height))
    if SCREEN_HEIGHT - solved_rect.bottom >= piece_height:
        candidates.append((0, SCREEN_WIDTH - piece_width, solved_rect.bottom, SCREEN_HEIGHT - piece_height))
    if not candidates:
        return (0,0)
    region = random.choice(candidates)
    x_min, x_max, y_min, y_max = region
    if x_max < x_min: x_max = x_min
    if y_max < y_min: y_max = y_min
    return random.randint(x_min, x_max), random.randint(y_min, y_max)
  
def lerp(a, b, t):
    return (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t)

def generate_edge(start, end, edge_type, tab_size, bump_direction, num_points=20):
    if edge_type == 0:
        return [start, end]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    L_edge = math.hypot(dx, dy)
    if L_edge == 0:
        return [start, end]
    u = (dx / L_edge, dy / L_edge)
    v = bump_direction
    neck_length  = tab_size * 0.25
    along_offset = tab_size * 0.25
    bump_radius  = tab_size
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    if edge_type > 0:
        p1 = (mid[0] - u[0] * along_offset, mid[1] - u[1] * along_offset)
        p4 = (mid[0] + u[0] * along_offset, mid[1] + u[1] * along_offset)
    else:
        p1 = (mid[0] + u[0] * along_offset, mid[1] + u[1] * along_offset)
        p4 = (mid[0] - u[0] * along_offset, mid[1] - u[1] * along_offset)
    p2 = (p1[0] + edge_type * neck_length * v[0], p1[1] + edge_type * neck_length * v[1])
    p3 = (p4[0] + edge_type * neck_length * v[0], p4[1] + edge_type * neck_length * v[1])
    tip = (mid[0] + edge_type * bump_radius * v[0], mid[1] + edge_type * bump_radius * v[1])
    X = (along_offset**2 + neck_length**2 - bump_radius**2) / (2 * (neck_length - bump_radius))
    center = (mid[0] + edge_type * X * v[0], mid[1] + edge_type * X * v[1])
    r = abs(bump_radius - X)
    def angle(point):
        return math.atan2(point[1] - center[1], point[0] - center[0])
    angle2 = angle(p2)
    angle3 = angle(p3)
    angle_tip = angle(tip)
    def normalize_angle(a):
        while a < 0:
            a += 2 * math.pi
        while a >= 2 * math.pi:
            a -= 2 * math.pi
        return a
    a2 = normalize_angle(angle2)
    a3 = normalize_angle(angle3)
    a_tip = normalize_angle(angle_tip)
    def is_between(a, b, c):
        a = normalize_angle(a)
        b = normalize_angle(b)
        c = normalize_angle(c)
        if a < b:
            return a <= c <= b
        return a <= c or c <= b
    if not is_between(a2, a3, a_tip):
        a2, a3 = a3, a2
    num_arc = 20
    delta_angle = a3 - a2
    if delta_angle < 0:
        delta_angle += 2 * math.pi
    arc_points = []
    for i in range(num_arc + 1):
        t = i / num_arc
        theta = a2 + t * delta_angle
        arc_points.append((center[0] + r * math.cos(theta),
                           center[1] + r * math.sin(theta)))
    if edge_type < 0:
        arc_points.reverse()
    if edge_type > 0:
        first_neck = p1
        last_neck  = p4
        A1, B1 = p1, p2
        A2, B2 = p3, p4
    else:
        first_neck = p4
        last_neck  = p1
        A1, B1 = p4, p3
        A2, B2 = p2, p1
    points = []
    points.append(start)
    points.append(first_neck)
    num_trans = 5
    for i in range(1, num_trans):
        t = i / num_trans
        interp_point = (A1[0] * (1 - t) + B1[0] * t, A1[1] * (1 - t) + B1[1] * t)
        points.append(interp_point)
    points.append(B1)
    for pt in arc_points[1:]:
        points.append(pt)
    for i in range(1, num_trans):
        t = i / num_trans
        interp_point = (A2[0] * (1 - t) + B2[0] * t, A2[1] * (1 - t) + B2[1] * t)
        points.append(interp_point)
    points.append(B2)
    points.append(end)
    return points

def generate_piece_polygon(base_x, base_y, base_width, base_height, edges, tab_size):
    tl = (base_x, base_y)
    tr = (base_x+base_width, base_y)
    br = (base_x+base_width, base_y+base_height)
    bl = (base_x, base_y+base_height)
    top_edge = generate_edge(tl, tr, edges['top'], tab_size, (0,-1))
    right_edge = generate_edge(tr, br, edges['right'], tab_size, (1,0))
    bottom_edge = generate_edge(br, bl, edges['bottom'], tab_size, (0,1))
    left_edge = generate_edge(bl, tl, edges['left'], tab_size, (-1,0))
    return top_edge[:-1] + right_edge[:-1] + bottom_edge[:-1] + left_edge[:-1]

def average_color(surface):
    arr = pygame.surfarray.array3d(surface)
    avg = np.mean(arr, axis=(0,1))
    return (int(avg[0]), int(avg[1]), int(avg[2]))

def darken_color(color, amount=50):
    return (max(color[0]-amount,0), max(color[1]-amount,0), max(color[2]-amount,0))

# --- Puzzle State Persistence ---
def save_surface_as_jpg(surface, filename):
    data = pygame.image.tostring(surface, 'RGB')
    img = Image.frombytes('RGB', surface.get_size(), data)
    img.save(filename, 'JPEG')
    print(f"Saved completed puzzle as {filename}")

def save_puzzle_state():
    state = {
        "image_file": image_file,
        "rows": rows,
        "cols": cols,
        "tab_size": tab_size,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "piece_edges": piece_edges,
        "pieces": []
    }
    for p in pieces:
        rct = p["rect"]
        state["pieces"].append({
            "grid_pos": p["grid_pos"],
            "rect": (rct.x, rct.y, rct.width, rct.height),
            "placed": p.get("placed", False)
        })
    return state

def merge_groups(group1, group2, adjust=(0,0)):
    for p in group2:
        p["rect"].x += adjust[0]
        p["rect"].y += adjust[1]
        p["group"] = group1
        group1.append(p)
    if group2 in groups:
        groups.remove(group2)
    return group1

def try_group_snap(drag_group):
    merged = False
    for p1 in drag_group:
        for other in pieces:
            if other in drag_group: continue
            r1, c1 = p1["grid_pos"]
            r2, c2 = other["grid_pos"]
            if abs(r1-r2)+abs(c1-c2) != 1: continue
            exp_dx = other["correct_pos"][0] - p1["correct_pos"][0]
            exp_dy = other["correct_pos"][1] - p1["correct_pos"][1]
            act_dx = other["rect"].x - p1["rect"].x
            act_dy = other["rect"].y - p1["rect"].y
            diff = (act_dx - exp_dx, act_dy - exp_dy)
            dist = (diff[0]**2+diff[1]**2)**0.5
            if dist < GROUP_SNAP_THRESHOLD:
                adjust = (-diff[0], -diff[1])
                g1 = p1["group"]
                g2 = other["group"]
                if g1 is not g2:
                    merge_groups(g1, g2, adjust)
                    merged = True
    if merged and snap_sound is not None:
        snap_sound.play(fade_ms=1)
    return merged

def rebuild_groups(pieces):
    changed = True
    while changed:
        changed = False
        for i in range(len(pieces)):
            for j in range(i+1, len(pieces)):
                p1 = pieces[i]
                p2 = pieces[j]
                if abs(p1["grid_pos"][0]-p2["grid_pos"][0]) + abs(p1["grid_pos"][1]-p2["grid_pos"][1]) == 1:
                    exp_dx = p2["correct_pos"][0] - p1["correct_pos"][0]
                    exp_dy = p2["correct_pos"][1] - p1["correct_pos"][1]
                    act_dx = p2["rect"].x - p1["rect"].x
                    act_dy = p2["rect"].y - p1["rect"].y
                    diff = (act_dx - exp_dx, act_dy - exp_dy)
                    dist = (diff[0]**2+diff[1]**2)**0.5
                    if dist < GROUP_SNAP_THRESHOLD:
                        if p1["group"] is not p2["group"]:
                            merge_groups(p1["group"], p2["group"])
                            changed = True
    return pieces

def load_puzzle_state(state):
    global image_file, rows, cols, cell_width, cell_height, tab_size, piece_edges, groups, puzzle_area_rect, original_img
    image_file = state["image_file"]
    rows = state["rows"]
    cols = state["cols"]
    tab_size = state["tab_size"]
    cell_width = state["cell_width"]
    cell_height = state["cell_height"]
    piece_edges = state["piece_edges"]
    original_img = load_image_cached(image_file)
    full_img, puzzle_area_rect = scale_image_preserve_aspect(original_img, SOLVED_RECT)
    new_pieces = []
    new_groups = []
    for r in range(rows):
        for c in range(cols):
            ext_left = tab_size if c>0 else 0
            ext_top = tab_size if r>0 else 0
            ext_right = tab_size if c<cols-1 else 0
            ext_bottom = tab_size if r<rows-1 else 0
            cell_x = c * cell_width
            cell_y = r * cell_height
            crop_x = cell_x - ext_left
            crop_y = cell_y - ext_top
            crop_width = cell_width + ext_left + ext_right
            crop_height = cell_height + ext_top + ext_bottom
            crop_rect = pygame.Rect(crop_x, crop_y, crop_width, crop_height)
            try:
                piece_img = full_img.subsurface(crop_rect).copy()
            except ValueError:
                piece_img = pygame.Surface((crop_width, crop_height), pygame.SRCALPHA)
            piece_surface = pygame.Surface((crop_width, crop_height), pygame.SRCALPHA)
            piece_surface.blit(piece_img, (0,0))
            poly = generate_piece_polygon(ext_left, ext_top, cell_width, cell_height, piece_edges[r][c], tab_size)
            mask_surface = pygame.Surface((crop_width, crop_height), pygame.SRCALPHA)
            mask_surface.fill((0,0,0,0))
            pygame.draw.polygon(mask_surface, (255,255,255,255), poly)
            mask = pygame.mask.from_surface(mask_surface)
            mask_image = mask.to_surface(setcolor=(255,255,255,255), unsetcolor=(0,0,0,0))
            piece_surface.blit(mask_image, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
            pygame.draw.polygon(piece_surface, (0,0,0), poly, 1)
            correct_x = puzzle_area_rect.x + c*cell_width - ext_left
            correct_y = puzzle_area_rect.y + r*cell_height - ext_top
            piece = {"image": piece_surface, "correct_pos": (correct_x, correct_y),
                     "rect": pygame.Rect(0,0,crop_width, crop_height), "grid_pos": (r,c)}
            piece["group"] = [piece]
            new_pieces.append(piece)
            new_groups.append(piece["group"])
    for saved_piece in state["pieces"]:
        for piece in new_pieces:
            if piece["grid_pos"] == tuple(saved_piece["grid_pos"]):
                x,y,w,h = saved_piece["rect"]
                piece["rect"] = pygame.Rect(x,y,w,h)
                piece["placed"] = saved_piece["placed"]
                break
    groups = new_groups
    rebuild_groups(new_pieces)
    return new_pieces, groups

def create_puzzle(loaded_state=None, new_image_file=None, new_rows=None, new_cols=None):
    global image_file, rows, cols, cell_width, cell_height, tab_size, piece_edges, pieces, groups, puzzle_area_rect, original_img
    if loaded_state:
        pieces, groups = load_puzzle_state(loaded_state)
    else:
        image_file = os.path.join("images", new_image_file)
        rows = new_rows
        cols = new_cols
        original_img = load_image_cached(image_file)
        full_img, puzzle_area_rect = scale_image_preserve_aspect(original_img, SOLVED_RECT)
        cell_width = puzzle_area_rect.width // cols
        cell_height = puzzle_area_rect.height // rows
        tab_size = min(cell_width, cell_height) // 4
        piece_edges = [[None for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                cell = {}
                cell['top'] = 0 if r==0 else -piece_edges[r-1][c]['bottom']
                cell['left'] = 0 if c==0 else -piece_edges[r][c-1]['right']
                cell['right'] = 0 if c==cols-1 else random.choice([1,-1])
                cell['bottom'] = 0 if r==rows-1 else random.choice([1,-1])
                piece_edges[r][c] = cell
        pieces = []
        groups = []
        for r in range(rows):
            for c in range(cols):
                ext_left = tab_size if c>0 else 0
                ext_top = tab_size if r>0 else 0
                ext_right = tab_size if c<cols-1 else 0
                ext_bottom = tab_size if r<rows-1 else 0
                cell_x = c * cell_width
                cell_y = r * cell_height
                crop_x = cell_x - ext_left
                crop_y = cell_y - ext_top
                crop_width = cell_width + ext_left + ext_right
                crop_height = cell_height + ext_top + ext_bottom
                crop_rect = pygame.Rect(crop_x, crop_y, crop_width, crop_height)
                try:
                    piece_img = full_img.subsurface(crop_rect).copy()
                except ValueError:
                    piece_img = pygame.Surface((crop_width, crop_height), pygame.SRCALPHA)
                piece_surface = pygame.Surface((crop_width, crop_height), pygame.SRCALPHA)
                piece_surface.blit(piece_img, (0,0))
                poly = generate_piece_polygon(ext_left, ext_top, cell_width, cell_height, piece_edges[r][c], tab_size)
                mask_surface = pygame.Surface((crop_width, crop_height), pygame.SRCALPHA)
                mask_surface.fill((0,0,0,0))
                pygame.draw.polygon(mask_surface, (255,255,255,255), poly)
                mask = pygame.mask.from_surface(mask_surface)
                mask_image = mask.to_surface(setcolor=(255,255,255,255), unsetcolor=(0,0,0,0))
                piece_surface.blit(mask_image, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
                pygame.draw.polygon(piece_surface, (0,0,0), poly, 1)
                correct_x = puzzle_area_rect.x + c*cell_width - ext_left
                correct_y = puzzle_area_rect.y + r*cell_height - ext_top
                piece = {"image": piece_surface, "correct_pos": (correct_x, correct_y),
                         "rect": None, "grid_pos": (r,c)}
                init_x, init_y = get_random_position_outside(puzzle_area_rect, crop_width, crop_height)
                piece["rect"] = pygame.Rect(init_x, init_y, crop_width, crop_height)
                piece["group"] = [piece]
                pieces.append(piece)
                groups.append(piece["group"])
    return pieces, groups

# --- New Function: Assemble a Full-Resolution Solved Puzzle Image ---
def assemble_full_resolution_solved_image():
    full_width = original_img.get_width()
    full_height = original_img.get_height()
    full_cell_width = full_width // cols
    full_cell_height = full_height // rows
    full_tab_size = min(full_cell_width, full_cell_height) // 4
    completed = pygame.Surface((full_width, full_height))
    completed.fill((50,50,50))
    for r in range(rows):
        for c in range(cols):
            ext_left = full_tab_size if c > 0 else 0
            ext_top = full_tab_size if r > 0 else 0
            ext_right = full_tab_size if c < cols-1 else 0
            ext_bottom = full_tab_size if r < rows-1 else 0
            cell_x = c * full_cell_width
            cell_y = r * full_cell_height
            crop_x = cell_x - ext_left
            crop_y = cell_y - ext_top
            crop_width = full_cell_width + ext_left + ext_right
            crop_height = full_cell_height + ext_top + ext_bottom
            crop_rect = pygame.Rect(crop_x, crop_y, crop_width, crop_height)
            try:
                piece_img = original_img.subsurface(crop_rect).copy()
            except ValueError:
                piece_img = pygame.Surface((crop_width, crop_height), pygame.SRCALPHA)
            piece_surface = pygame.Surface((crop_width, crop_height), pygame.SRCALPHA)
            piece_surface.blit(piece_img, (0,0))
            poly = generate_piece_polygon(ext_left, ext_top, full_cell_width, full_cell_height, piece_edges[r][c], full_tab_size)
            mask_surface = pygame.Surface((crop_width, crop_height), pygame.SRCALPHA)
            mask_surface.fill((0,0,0,0))
            pygame.draw.polygon(mask_surface, (255,255,255,255), poly)
            mask = pygame.mask.from_surface(mask_surface)
            mask_image = mask.to_surface(setcolor=(255,255,255,255), unsetcolor=(0,0,0,0))
            piece_surface.blit(mask_image, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
            pygame.draw.polygon(piece_surface, (0,0,0), poly, 1)
            correct_x = cell_x - ext_left
            correct_y = cell_y - ext_top
            completed.blit(piece_surface, (correct_x, correct_y))
    return completed

def handle_puzzle_events(buttons, event):
    global game_state, dragging_group, drag_offsets, puzzle_solved, puzzle_auto_saved, image_file, original_img
    if event.type == pygame.MOUSEBUTTONDOWN:
        pos = pygame.mouse.get_pos()
        if point_in_rect(pos, buttons["menu"]["rect"]):
            game_state = "menu_main"
            return
        elif point_in_rect(pos, buttons["save"]["rect"]):
            name = '.'.join(image_file.split('\\')[-1].split('.')[:-1]) if '.' in image_file else image_file.split('\\')[-1]
            filename = os.path.join("saves", f"puzzle_{name}.json")
            state = save_puzzle_state()
            with open(filename, "w") as f:
                json.dump(state, f)
            print(f"Puzzle saved to {filename}")
            return
        for p in reversed(pieces):
            if p["rect"].collidepoint(pos):
                dragging_group = p["group"]
                drag_offsets = {}
                for piece in dragging_group:
                    ox = piece["rect"].x - pos[0]
                    oy = piece["rect"].y - pos[1]
                    drag_offsets[id(piece)] = (ox, oy)
                pieces.sort(key=lambda x: 0 if x in dragging_group else 1)
                break
    elif event.type == pygame.MOUSEBUTTONUP:
        if dragging_group is not None:
            try_group_snap(dragging_group)
            dragging_group = None
            drag_offsets = {}
            if len(groups)==1 and len(groups[0])==len(pieces):
                puzzle_solved = True
    elif event.type == pygame.MOUSEMOTION:
        if dragging_group is not None:
            pos = pygame.mouse.get_pos()
            for piece in dragging_group:
                ox, oy = drag_offsets.get(id(piece), (0,0))
                piece["rect"].x = pos[0] + ox
                piece["rect"].y = pos[1] + oy
    if puzzle_solved:
        font = get_font(72)
        text = font.render("Congratulations!", True, (255,215,0))
        text_rect = text.get_rect(center=SOLVED_RECT.center)
        screen.blit(text, text_rect)
        if not puzzle_auto_saved:
            completed = assemble_full_resolution_solved_image()
            name = "completed/" + str(len(pieces)) + "-Pieces-" + ('.'.join(image_file.split('\\')[-1].split('.')[:-1]) if '.' in image_file else image_file.split('\\')[-1]) + ".jpeg"
            save_surface_as_jpg(completed, name)
            puzzle_auto_saved = True

def handle_load_menu_events(buttons, event):
    global game_state, pieces, groups
    if event.type == pygame.MOUSEBUTTONDOWN:
        pos = pygame.mouse.get_pos()
        for key, btn in buttons.items():
            if btn["rect"].collidepoint(pos):
                if btn["label"]=="Back":
                    game_state = "menu_main"
                else:
                    file_path = os.path.join("saves", btn["label"])
                    with open(file_path, "r") as f:
                        state = json.load(f)
                    create_puzzle(loaded_state=state)
                    game_state = "puzzle"
                break

def handle_main_menu_events(buttons, event):
    global game_state
    if event.type == pygame.MOUSEBUTTONDOWN:
        pos = pygame.mouse.get_pos()
        for key, btn in buttons.items():
            if btn["rect"].collidepoint(pos):
                if btn["label"]=="New Puzzle":
                    game_state = "menu_new"
                elif btn["label"]=="Load Puzzle":
                    game_state = "menu_load"
                elif btn["label"]=="Exit":
                    pygame.quit(); sys.exit()

def handle_new_menu_events(buttons, event):
    global game_state, new_puzzle_settings
    if event.type == pygame.MOUSEBUTTONDOWN:
        pos = pygame.mouse.get_pos()
        if buttons["pieces_dec"]["rect"].collidepoint(pos):
            if new_puzzle_settings["pieces"] > 20:
                new_puzzle_settings["pieces"] -= 50
        elif buttons["pieces_inc"]["rect"].collidepoint(pos):
            if new_puzzle_settings["pieces"] < 1200:
                new_puzzle_settings["pieces"] += 50
        else:
            for key in buttons:
                if key.startswith("preset_") and buttons[key]["rect"].collidepoint(pos):
                    new_puzzle_settings["pieces"] = buttons[key]["piece_count"]
                    break
            if buttons["img_left"]["rect"].collidepoint(pos):
                new_puzzle_settings["image_index"] = (new_puzzle_settings["image_index"] - 1) % len(new_puzzle_settings["image_list"])
            elif buttons["img_right"]["rect"].collidepoint(pos):
                new_puzzle_settings["image_index"] = (new_puzzle_settings["image_index"] + 1) % len(new_puzzle_settings["image_list"])
            else:
                for key in buttons:
                    if key.startswith("candidate_") and buttons[key]["rect"].collidepoint(pos):
                        config = buttons[key].get("config")
                        if config:
                            r, c = config
                            create_puzzle(new_image_file=new_puzzle_settings["image_list"][new_puzzle_settings["image_index"]],
                                          new_rows=r, new_cols=c)
                            game_state = "puzzle"
                            return
                if buttons["back"]["rect"].collidepoint(pos):
                    game_state = "menu_main"

def main_loop():
    global game_state
    while True:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit(); sys.exit()
            if game_state=="menu_main":
                btns = draw_main_menu()
                handle_main_menu_events(btns, event)
            elif game_state=="menu_new":
                btns = draw_new_menu()
                handle_new_menu_events(btns, event)
            elif game_state=="menu_load":
                btns = draw_load_menu()
                handle_load_menu_events(btns, event)
            elif game_state=="puzzle":
                btns = draw_puzzle()
                handle_puzzle_events(btns, event)
        pygame.display.flip()
        clock.tick(FPS)

if __name__=="__main__":
    fade_in()
    main_loop()
