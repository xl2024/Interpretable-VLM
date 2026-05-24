import os
import random
import numpy as np
from tqdm import tqdm
import time
from unrealcv import Client

client = Client(('127.0.0.1', 9000))
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

    def render_character(self):
        os.makedirs(os.path.join(self.output_dir, "characters"), exist_ok=True)
        char_list = [("camel", "salt_desert", "StaticMeshActor_2", "0 0 0"), 
                     ("dolphin", "beach", "StaticMeshActor_4", "0 0 240"), 
                     ("elephant", "museum", "StaticMeshActor_8", "0 0 -50")
        ]
        for animal, env, actor_id, loc in char_list:
            client.request(f'vset /action/game/level {env}')
            client.request(f'vset /object/{actor_id}/location {loc}')
            time.sleep(5)    # fit low vram
            # Snap the photo
            filepath = os.path.join(self.output_dir, "characters", f"{animal}_at_{env}.png")
            abs_filepath = os.path.abspath(filepath)
            client.request(f'vget /camera/0/lit {abs_filepath}')

    def apply_material(self, object_id, color):
        """Map color strings to Unreal Engine Material paths."""
        material_map = {
            # multiple ways to refer the path to a material
            # "red": "/Game/Materials/M_Red.M_Red",
            # "red": "/Script/Engine.Material'/Game/Materials/M_Red.M_Red'",
            # "red": "MaterialInstanceConstant'/Game/Materials/M_Red.M_Red'",
            "red": "Material'/Game/Materials/M_Red.M_Red'",
            "green": "Material'/Game/Materials/M_Green.M_Green'",
            "white": "Material'/Game/Materials/M_White.M_White'"
        }
        client.request(f'vset /object/{object_id}/material 0 {material_map[color]}')
        # time.sleep(0.05)
        # actual_material = client.request(f'vget /object/{object_id}/material')
        # actual_color = client.request(f'vget /object/{object_id}/color')
        # print(f"Verification - {object_id} material is: {actual_material} (set to {color})")
        # print(f"Verification - {object_id} color is: {actual_color} (set to {color})")    # the color attr doesn't matter

    def render_image(self, left_color, left_animal, right_color, right_animal, env_name, filepath):
        """Executes the Unreal Engine commands to manipulate the scene and capture the image."""
        client.request(f'vset /action/game/level {env_name}')
        # print(client.request('vget /objects'))
        animal_ids = {
            "beach": {"camel": {"left": "StaticMeshActor_2", "right": "StaticMeshActor_7"},
                      "dolphin": {"left": "StaticMeshActor_4", "right": "StaticMeshActor_8"},
                      "elephant": {"left": "StaticMeshActor_6", "right": "StaticMeshActor_9"}
                      },
            "salt_desert": {"camel": {"left": "StaticMeshActor_2", "right": "StaticMeshActor_7"},
                            "dolphin": {"left": "StaticMeshActor_4", "right": "StaticMeshActor_8"},
                            "elephant": {"left": "StaticMeshActor_6", "right": "StaticMeshActor_9"}
                            },
            "museum": {"camel": {"left": "StaticMeshActor_2", "right": "StaticMeshActor_9"},
                       "dolphin": {"left": "StaticMeshActor_6", "right": "StaticMeshActor_10"},
                       "elephant": {"left": "StaticMeshActor_8", "right": "StaticMeshActor_11"}
                       }
        }
        # x: front/behind, y: left/right, z: up/down
        animal_z_offset = {    # leave 10 cm for jitters
            "camel": {"beach": 0, "salt_desert": 0, "museum": -50},
            "dolphin": {"beach": 100, "salt_desert": 100, "museum": 50},    # +20 height to match other animals
            "elephant": {"beach": 0, "salt_desert": 0, "museum": -50},
        }
        for sm in animal_ids[env_name].values():
            for actor_id in sm.values():
                client.request(f'vset /object/{actor_id}/location 0 0 -500')
                # time.sleep(0.05)
                # actual_position = client.request(f'vget /object/{actor_id}/location')
                # print(f"Verification - {actor_id} position is: {actual_position}")

        left_actor = animal_ids[env_name][left_animal]["left"]
        right_actor = animal_ids[env_name][right_animal]["right"]

        self.apply_material(left_actor, left_color)
        self.apply_material(right_actor, right_color)
        # Base Y (Left/Right). Left stays strictly negative, Right stays positive
        left_base_y = -200.0    # centimeters in the 3D world
        right_base_y = 200.0

        # Jitter X (Depth: Closer to or further from the camera)
        left_jitter_x = np.random.uniform(-15, 15)
        right_jitter_x = np.random.uniform(-15, 15)

        # Jitter Y (Horizontal: Move slightly left and right)
        left_jitter_y = left_base_y + np.random.uniform(-15, 15)
        right_jitter_y = right_base_y + np.random.uniform(-15, 15)

        # Jitter Z (Up/Down: Sink slightly into the ground or float slightly)
        # We keep this range a bit smaller (-15 to 15) so they don't completely disappear under the map
        left_jitter_z = animal_z_offset[left_animal][env_name] + 15 + np.random.uniform(-15, 15)
        right_jitter_z = animal_z_offset[right_animal][env_name] + 15 + np.random.uniform(-15, 15)

        # Apply the fully jittered 3D coordinates (X, Y, Z)
        client.request(f'vset /object/{left_actor}/location {left_jitter_x} {left_jitter_y} {left_jitter_z}')
        client.request(f'vset /object/{right_actor}/location {right_jitter_x} {right_jitter_y} {right_jitter_z}')
        # time.sleep(0.05)
        # actual_left_pos = client.request(f'vget /object/{left_actor}/location')
        # print(f"Verification - {left_actor}({left_animal}) position is: {actual_left_pos}")
        # actual_right_pos = client.request(f'vget /object/{right_actor}/location')
        # print(f"Verification - {right_actor}({right_animal}) position is: {actual_right_pos}")
        
        base_roll = {"camel": 0, "dolphin": 90, "elephant": 90}
        left_base_yaw = np.random.choice([0, 180])
        right_base_yaw = np.random.choice([0, 180])
        left_jitter_yaw = left_base_yaw + np.random.uniform(-10, 10)
        right_jitter_yaw = right_base_yaw + np.random.uniform(-10, 10)
        # rotations: y - Pitch, z - Yaw, x - Roll -> Roll, Pitch, Yaw
        client.request(f'vset /object/{left_actor}/rotation 0 {left_jitter_yaw} {base_roll[left_animal]}')
        client.request(f'vset /object/{right_actor}/rotation 0 {right_jitter_yaw} {base_roll[right_animal]}')

        time.sleep(1)    # fit low vram
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
    # generator.render_character()

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