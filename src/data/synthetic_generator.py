from PIL import Image, ImageDraw
from typing import Tuple, List, Optional
import math

def generate_custom_image(
    image_size: Tuple[int, int] = (336, 336),
    cols: int = 2,
    rows: int = 1,
    shapes: List[str] = None,
    colors: List[str] = None,
    coords: List[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> Image.Image:
    """
    Generates an image with an arbitrary number of colored shapes.
    Uses `cols` and `rows` to dynamically scale the object bounding boxes.
    """
    # 1. Create the blank canvas
    width, height = image_size
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # 2. Calculate dynamic shape size based on the grid
    cell_width = width / cols
    cell_height = height / rows
    box_size = min(cell_width, cell_height) * 0.6
    half_size = int(box_size / 2)

    # 3. Draw each object
    for shape, color, coord in zip(shapes, colors, coords):
        row_idx, col_idx = coord
        cx = int((col_idx + 0.5) * cell_width)
        cy = int((row_idx + 0.5) * cell_height)
        
        x0 = cx - half_size
        y0 = cy - half_size
        x1 = cx + half_size
        y1 = cy + half_size

        # Bounding box for circles and squares
        bbox = [x0, y0, x1, y1]
        w = x1 - x0
        h = y1 - y0
        
        shape_type = shape.lower()
        if shape_type == "circle":
            draw.ellipse(bbox, fill=color)

        elif shape_type == "square":
            draw.rectangle(bbox, fill=color)

        elif shape_type == "triangle":
            # An upward-pointing equilateral-ish triangle
            points = [
                (cx, cy - half_size),                 # Top vertex
                (cx - half_size, cy + half_size),     # Bottom left
                (cx + half_size, cy + half_size)      # Bottom right
            ]
            draw.polygon(points, fill=color)

        elif shape_type == "cross":
            thickness = int(w * 0.3)
            delta = int(thickness / 1.4)
            # Top-left to bottom-right
            draw.line([(x0+delta, y0+delta), (x1-delta, y1-delta)], fill=color, width=thickness)
            # Bottom-left to top-right
            draw.line([(x0+delta, y1-delta), (x1-delta, y0+delta)], fill=color, width=thickness)
        
        elif shape_type == "star":
            # 5-pointed star using trigonometry
            points = []
            outer_radius = w / 2
            inner_radius = outer_radius * 0.4
            start_angle = -math.pi / 2 # Start at top
            
            for i in range(10):
                r = outer_radius if i % 2 == 0 else inner_radius
                angle = start_angle + i * (math.pi / 5)
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                points.append((x, y))
                
            draw.polygon(points, fill=color)

        elif shape_type == "heart":
            r = w / 4
            # Centers of the left and right upper circles
            xl, yc = x0 + r, y0 + r
            xr = x1 - r
            # Draw the two top circles
            draw.ellipse([x0, y0, cx, y0 + 2*r], fill=color) # Left bump
            draw.ellipse([cx, y0, x1, y0 + 2*r], fill=color) # Right bump
            # 1. Calculate distance from left circle center to the bottom tip
            dx, dy = cx - xl, y1 - yc
            dist = math.hypot(dx, dy)    # Euclidean norm
            # 2. Calculate the angles for the tangent points
            theta_p = math.atan2(dy, dx) # arc tangent of dy/dx in radians
            theta_t = math.acos(r / dist) # Offset angle for the tangent
            # 3. Find the exact left tangent coordinate
            tx_l = xl + r * math.cos(theta_p + theta_t)
            ty_l = yc + r * math.sin(theta_p + theta_t)
            # 4. Find the exact right tangent coordinate (mirrored)
            theta_p_r = math.atan2(dy, cx - xr)
            tx_r = xr + r * math.cos(theta_p_r - theta_t)
            ty_r = yc + r * math.sin(theta_p_r - theta_t)
            # 5. Fill the body of the heart connecting the tangents to the bottom tip
            # We route through the centers (xl, yc) and (xr, yc) to ensure no empty gaps
            draw.polygon([
                (cx, y1),      # Bottom tip
                (tx_l, ty_l),  # Left tangent point
                (xl, yc),      # Center of left circle
                (xr, yc),      # Center of right circle
                (tx_r, ty_r)   # Right tangent point
            ], fill=color)

        elif shape_type == "sun":
            # Central circle
            sun_r = w / 5
            draw.ellipse([cx - sun_r, cy - sun_r, cx + sun_r, cy + sun_r], fill=color)
            
            # 8 outer rays
            ray_inner = sun_r * 1.4
            ray_outer = w / 2
            ray_thickness = max(2, int(w / 15))
            
            for i in range(8):
                angle = i * math.pi / 4
                rx0 = cx + ray_inner * math.cos(angle)
                ry0 = cy + ray_inner * math.sin(angle)
                rx1 = cx + ray_outer * math.cos(angle)
                ry1 = cy + ray_outer * math.sin(angle)
                draw.line([(rx0, ry0), (rx1, ry1)], fill=color, width=ray_thickness)

        elif shape_type == "umbrella":
            # 1. Draw the Canopy (semi-circle top, scalloped bottom)
            y_offset = w * 0.05
            _cy = cy + y_offset
            canopy_pts = []
            # Top arc (Left to Right)
            for i in range(180, 361):
                rad = math.radians(i)
                canopy_pts.append((cx + half_size * math.cos(rad), _cy + half_size * math.sin(rad)))
            
            # Bottom scallops (Right back to Left)
            scallops_centers = [(cx + box_size / 3, _cy), (cx, _cy), (cx - box_size / 3, _cy)]
            for ctr_x, ctr_y in scallops_centers:
                for i in range(360, 179, -1):
                    rad = math.radians(i)
                    canopy_pts.append((ctr_x + box_size * math.cos(rad) / 6, ctr_y + box_size * math.sin(rad) / 6))

            draw.polygon(canopy_pts, fill=color)
            
            # 2. Draw the Handle (Shaft + J-hook)
            thickness = max(2, int(w * 0.05))
            hook_radius = half_size * 0.25
            shaft_bottom = y1 - hook_radius
            
            # Straight shaft down the middle
            draw.line([(cx, y0), (cx, shaft_bottom)], fill=color, width=thickness)
            
            # J-Hook curving to the left
            hook_bbox = [
                cx - 2 * hook_radius,       # x0_hook
                shaft_bottom - hook_radius, # y0_hook
                cx,                         # x1_hook
                shaft_bottom + hook_radius  # y1_hook
            ]
            # Arc from 0 to 180 degrees draws the bottom half of the circle
            draw.arc(hook_bbox, 0, 180, fill=color, width=thickness)

        elif shape_type == "plane":
            # Define a plane profile using normalized coordinates (-1.0 to 1.0)
            plane_profile = [
                (0.00, -1.00),  # Nose
                (0.08, -0.70),  # Right nose curve
                (0.10, -0.30),  # Right wing root front
                (0.80,  0.20),  # Right wing tip front
                (0.80,  0.25),  # Right wing tip back
                (0.10,  0.10),  # Right wing root back
                (0.08,  0.70),  # Right tail root front
                (0.35,  0.85),  # Right tail tip front
                (0.35,  1.00),  # Right tail tip back
                (0.05,  0.95),  # Right tail inner
                (0.00,  1.00),  # Tail center bottom (exhaust)
                (-0.05, 0.95),  # Left tail inner
                (-0.35, 1.00),  # Left tail tip back
                (-0.35, 0.85),  # Left tail tip front
                (-0.08, 0.70),  # Left tail root front
                (-0.10, 0.10),  # Left wing root back
                (-0.80, 0.25),  # Left wing tip back
                (-0.80, 0.20),  # Left wing tip front
                (-0.10, -0.30), # Left wing root front
                (-0.08, -0.70)  # Left nose curve
            ]
            
            angle = math.radians(90)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            points = []
            for px, py in plane_profile:
                # Apply 2D Rotation matrix
                rx = px * cos_a - py * sin_a
                ry = px * sin_a + py * cos_a
                
                # Scale the normalized points by half_size and translate to grid center
                points.append((cx + rx * half_size, cy + ry * half_size))
                
            draw.polygon(points, fill=color)

        else:
            print(f"Warning: Unknown shape '{shape}'. Defaulting to square.")
            draw.rectangle(bbox, fill=color)

    # 4. Save to disk if a path is provided
    if save_path:
        img.save(save_path)

    return img

if __name__ == "__main__":
    # test_img = generate_custom_image(
    # image_size=(336, 336),
    # cols=3,
    # rows=3,
    # shapes=['circle', 'square', 'circle','square'],
    # colors=['blue', 'red','red','blue'],
    # coords=[(0,0), (0,1),(1,0),(1,1)],
    #     save_path="data/test_samples/blue_circle_red_square.png"
    # )
    test_img = generate_custom_image(
        image_size=(336, 336),
        cols=3,
        rows=3,
        shapes=['circle', 'star', 'plane', 'square', 'umbrella', 'triangle', 'sun', 'heart', 'cross'],
        colors=['red', 'gold', 'grey', 'blue', 'hotpink', 'lime', 'black', 'purple', 'darkorange'],
        coords=[(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)],
        save_path="src/data/test_9_shapes.png"
    )