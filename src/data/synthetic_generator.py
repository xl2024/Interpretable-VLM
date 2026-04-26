from PIL import Image, ImageDraw
from typing import Tuple, List, Optional

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
    box_size = min(cell_width, cell_height) * 0.8
    half_size = int(box_size / 2)

    # 3. Draw each object
    for shape, color, coord in zip(shapes, colors, coords):
        row_idx, col_idx = coord
        cx = int((col_idx + 0.5) * cell_width)
        cy = int((row_idx + 0.5) * cell_height)

        # Bounding box for circles and squares
        bbox = [
            cx - half_size, 
            cy - half_size, 
            cx + half_size, 
            cy + half_size
        ]

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
        else:
            print(f"Warning: Unknown shape '{shape}'. Defaulting to square.")
            draw.rectangle(bbox, fill=color)

    # 4. Save to disk if a path is provided
    if save_path:
        img.save(save_path)

    return img

if __name__ == "__main__":
    # A quick local test to verify the drawing math works before running the full model
    test_img = generate_custom_image(
    image_size=(336, 336),
    cols=3,
    rows=3,
    shapes=['circle', 'square', 'circle','square'],
    colors=['blue', 'red','red','blue'],
    coords=[(0,0), (0,1),(1,0),(1,1)],
        save_path="data/test_samples/blue_circle_red_square.png"
    )