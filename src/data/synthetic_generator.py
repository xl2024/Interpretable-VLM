from PIL import Image, ImageDraw
import os

def generate_spatial_binding_image(
    image_size: tuple = (336, 336), 
    left_shape: str = "circle", 
    left_color: str = "blue",
    right_shape: str = "square", 
    right_color: str = "red",
    save_path: str = None
) -> Image.Image:
    """
    Generates a synthetic image with two distinct shapes on a white background.
    This creates perfect, noise-free geometry for testing Spatial IDs.
    
    Args:
        image_size: Tuple of (width, height). Defaults to 336x336 for CLIP ViT.
        left_shape: "circle" or "square".
        left_color: String color name (e.g., "blue", "red", "green").
        right_shape: "circle" or "square".
        right_color: String color name (e.g., "blue", "red", "green").
        save_path: Optional path to save the generated image to disk.
        
    Returns:
        A PIL Image object ready to be passed into the Hugging Face processor.
    """
    width, height = image_size
    
    # Create a pure white background
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    
    # Define bounding boxes to keep the shapes strictly separated (Left vs. Right)
    # use proportional math so the shapes stay centered even if image_size changes
    box_size = int(width * 0.25)
    center_y = height // 2
    
    # Left Object Coordinates
    left_x_center = int(width * 0.3)
    left_bbox = [
        left_x_center - box_size // 2, center_y - box_size // 2,
        left_x_center + box_size // 2, center_y + box_size // 2
    ]
    
    # Right Object Coordinates
    right_x_center = int(width * 0.7)
    right_bbox = [
        right_x_center - box_size // 2, center_y - box_size // 2,
        right_x_center + box_size // 2, center_y + box_size // 2
    ]
    
    # Helper function to draw the requested shape
    def draw_shape(bbox, shape_type, color):
        if shape_type.lower() == "circle":
            draw.ellipse(bbox, fill=color, outline="black", width=2)
        elif shape_type.lower() == "square":
            draw.rectangle(bbox, fill=color, outline="black", width=2)
        else:
            raise ValueError(f"Unsupported shape: {shape_type}. Use 'circle' or 'square'.")

    # Draw the shapes onto the canvas
    draw_shape(left_bbox, left_shape, left_color)
    draw_shape(right_bbox, right_shape, right_color)
    
    # Save to disk if a path is provided (useful for verifying the dataset manually)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
        print(f"Synthetic image saved to: {save_path}")
        
    return img

if __name__ == "__main__":
    # A quick local test to verify the drawing math works before running the full model
    test_img = generate_spatial_binding_image(
        save_path="data/test_samples/blue_circle_red_square.png"
    )