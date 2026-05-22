import os
import random
import numpy as np
from tqdm import tqdm
from unrealcv import client


class PUGDatasetGenerator:
    def __init__(self, output_dir="dataset/figure_3"):
        self.output_dir = output_dir
        
        # Create the disjoint estimation and evaluation directories
        os.makedirs(os.path.join(self.output_dir, "est"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "eval"), exist_ok=True)
        
        # Connect to the engine running in the background shell
        client.connect()
        if not client.isconnected():
            raise ConnectionError("Could not connect to Unreal Engine. Ensure the PUG binary is running.")
        print("Successfully connected to PUG Unreal Environment.")

    def apply_material(self, object_id, color):
        """Map color strings to Unreal Engine Material paths."""
        material_map = {
            "red": "Material'/Game/Materials/M_Red.M_Red'",
            "green": "Material'/Game/Materials/M_Green.M_Green'",
            "white": "Material'/Game/Materials/M_White.M_White'"
        }
        client.request(f'vset /object/{object_id}/material {material_map[color]}')

    def render_image(self, left_color, left_animal, right_color, right_animal, env_name, filepath):
        """Executes the Unreal Engine commands to manipulate the scene and capture the image."""
        client.request(f'vset /action/game/level {env_name}')
        
        mesh_map = {
            "camel": "StaticMesh'/Game/Meshes/SM_Camel.SM_Camel'",
            "dolphin": "StaticMesh'/Game/Meshes/SM_Dolphin.SM_Dolphin'",
            "elephant": "StaticMesh'/Game/Meshes/SM_Elephant.SM_Elephant'"
        }
        
        client.request(f'vset /object/LeftSlot/mesh {mesh_map[left_animal]}')
        client.request(f'vset /object/RightSlot/mesh {mesh_map[right_animal]}')

        self.apply_material("LeftSlot", left_color)
        self.apply_material("RightSlot", right_color)

        # Base Y (Left/Right). Left stays strictly negative, Right stays positive
        left_base_y = -150.0    # centimeters in the 3D world
        right_base_y = 150.0

        # Jitter X (Depth: Closer to or further from the camera)
        left_jitter_x = np.random.uniform(-40, 40)
        right_jitter_x = np.random.uniform(-40, 40)

        # Jitter Y (Horizontal: Move slightly left and right)
        left_jitter_y = left_base_y + np.random.uniform(-20, 20)
        right_jitter_y = right_base_y + np.random.uniform(-20, 20)

        # Jitter Z (Up/Down: Sink slightly into the ground or float slightly)
        # We keep this range a bit smaller (-15 to 15) so they don't completely disappear under the map
        left_jitter_z = np.random.uniform(-15, 15)
        right_jitter_z = np.random.uniform(-15, 15)

        # Apply the fully jittered 3D coordinates (X, Y, Z)
        client.request(f'vset /object/LeftSlot/location {left_jitter_x} {left_jitter_y} {left_jitter_z}')
        client.request(f'vset /object/RightSlot/location {right_jitter_x} {right_jitter_y} {right_jitter_z}')

        # Snap the photo
        abs_filepath = os.path.abspath(filepath)
        client.request(f'vget /camera/0/lit {abs_filepath}')

def main():
    animals = ["camel", "dolphin", "elephant"]
    colors = ["green", "red", "white"]
    environments = ["beach", "salt_desert", "museum"]

    # 1. Generate the 162 distinct-color combinations
    base_combinations = []
    for lc in colors:
        for la in animals:
            for rc in colors:
                if lc == rc:
                    continue  # Enforce distinct colors
                for ra in animals:
                    for env in environments:
                        base_combinations.append((lc, la, rc, ra, env))

    # 2. Shuffle ensuring a random mix of animals/environments across the splits
    random.seed(42) # Keeps the split reproducible if you run it again
    random.shuffle(base_combinations)

    # 3. Split 1/3 for estimation, 2/3 for evaluation
    est_combinations = base_combinations[:54]
    eval_combinations = base_combinations[54:]

    # 4. Build the task queue with 2 variations (x=1, 2) per combination
    all_tasks = []
    
    for combo in est_combinations:
        all_tasks.append((combo, "est", 1))
        all_tasks.append((combo, "est", 2))
        
    for combo in eval_combinations:
        all_tasks.append((combo, "eval", 1))
        all_tasks.append((combo, "eval", 2))

    generator = PUGDatasetGenerator()

    # 5. Run the rendering loop with a progress bar
    for combo, folder, x_val in tqdm(all_tasks, desc="Rendering PUG Images"):
        lc, la, rc, ra, env = combo
        
        # Naming convention: pug_leftcolor_leftanimal_rightcolor_rightanimal_env_x
        filename = f"pug_{lc}_{la}_{rc}_{ra}_{env}_{x_val}.png"
        filepath = os.path.join(generator.output_dir, folder, filename)
        
        generator.render_image(lc, la, rc, ra, env, filepath)

    client.disconnect()
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()