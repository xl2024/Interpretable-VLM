import gc
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Tuple, Any

from src.model.loader import load_vlm
from src.utils.tools import load_config, _resolve_text_model_dims, get_text_prompt, get_num_hidden_layers, get_model_id, to_kv_heads
from src.plots.cma_1d import run_mediation_analysis
from src.mech_interp.cma import cma_head_patching_by_logits, get_head_embeddings, get_top_k_heads
from src.data.synthetic_generator import generate_custom_image

# test_img = generate_custom_image(
#     image_size=(336, 336),
#     cols=3,
#     rows=3,
#     shapes=['circle', 'star', 'plane', 'square', 'umbrella', 'triangle', 'sun', 'heart', 'cross'],
#     colors=['red', 'gold', 'grey', 'blue', 'hotpink', 'lime', 'black', 'purple', 'darkorange'],
#     coords=[(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)],
#     save_path="src/data/test_9_shapes.png"
# )

def get_coord_from_index(index):
    return (index // 3, index % 3)

def cma_position_IDs_in_desc(model, processor, num_heads, top_k_heads, color_list, shape_list):
    ids_in_desc = {}
    for pos in range(9):
        text_prompts = []
        image_list = []
        for obj in range(9):
            prompt = "In this image there is a"
            colors = []
            shapes = []
            coords = []
            for i in range(9):
                if i != obj:
                    prompt += f" {color_list[i]} {shape_list[i]}, a"
                    colors.append(color_list[i])
                    shapes.append(shape_list[i])
                if i != pos:
                    coords.append(get_coord_from_index(i))

            prompt = prompt[:-3] + " and a"
            colors.append(color_list[obj])
            shapes.append(shape_list[obj])
            coords.append(get_coord_from_index(pos))
            image = generate_custom_image(cols=3, rows=3, shapes=shapes, colors=colors, coords=coords)
            text_prompts.append(get_text_prompt(model, prompt, image, processor))
            image_list.append(image)
                                    
        ids_in_desc[pos] = get_head_embeddings(
            model=model, 
            processor=processor, 
            num_heads=num_heads, 
            prompt_list=text_prompts,
            image_list=image_list, 
            top_k_heads=top_k_heads
        )

    return ids_in_desc

def get_rel_ref(colors, shapes, coords, pos):
    def get_index_from_coord(coords, p):
        for i in range(9):
            if coords[i] == get_coord_from_index(p):
                return i
        raise ValueError(f"Could not find coords from {coords} for position {p}")

    relations = {"above", "below", "left", "right"}
    row, col = get_coord_from_index(pos)
    if row == 0:
        relations.remove("below")
    elif row == 2:
        relations.remove("above")

    if col == 0:
        relations.remove("right")
    elif col == 2:
        relations.remove("left")

    rel = random.choice(list(relations))
    if rel == "above":
        index = get_index_from_coord(coords, pos+3)
    elif rel == "below":
        index = get_index_from_coord(coords, pos-3)
    elif rel == "left":
        index = get_index_from_coord(coords, pos+1)
    else:    # "right"
        index = get_index_from_coord(coords, pos-1)
    
    return rel, f"{colors[index]} {shapes[index]}"

def get_patching_results(model, processor, num_layers, num_heads, top_k_heads, ids_in_desc, color_list, shape_list):
    all_patching_results = {}
    
    for pos in range(9):
        all_patching_results[pos] = []

        for obj in range(9):
            colors, shapes, coords = [], [], []
            for i in range(9):
                if i != obj:
                    colors.append(color_list[i])
                    shapes.append(shape_list[i])
                if i != pos:
                    coords.append(get_coord_from_index(i))

                colors.append(color_list[obj])
                shapes.append(shape_list[obj])
                coords.append(get_coord_from_index(pos))

            RELATION, REF = get_rel_ref(colors, shapes, coords, pos)
            prompt = f"In this image, what is the color of the object that is directly {RELATION} of {REF}. Answer with the relevant color only."
            image = generate_custom_image(cols=3, rows=3, shapes=shapes, colors=colors, coords=coords)
            text_prompt = get_text_prompt(model, prompt, image, processor)
            d_t = ids_in_desc[pos]
            d_o = torch.zeros_like(d_t)

            predicted_word = cma_head_patching_by_logits(
                model=model,
                processor=processor,
                num_layers=num_layers,
                num_heads=num_heads,
                prompt_c1=text_prompt,
                image_c1=image,
                d_t_head_cache=d_t,
                top_k_heads=top_k_heads,
                alpha=2,
                d_o_head_cache=d_o
            )

            print(f"pos={pos}, i={i}, target={color_list[shuffle[pos]]}, before={color_list[i][pos]}, after={predicted_word}")
            all_patching_results[key].append([color_list[i][1-pos], predicted_word.lower()])
        

        print_results(all_patching_results[key], key)

    return all_patching_results["left"], all_patching_results["right"]


def main():
    model_id_list = [
        "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        #  "Qwen/Qwen2.5-VL-32B-Instruct",
        "llava-hf/llava-1.5-7b-hf"
        #  "llava-hf/llava-1.5-13b-hf"
    ]

    image_list, color_list, shape_list = generate_dataset()
    print(f"Generated {len(image_list)} images.")
    image_dataset = {"est": [], "eval": []}
    color_dataset = {"est": [], "eval": []}
    shape_dataset = {"est": [], "eval": []}
    # random.seed(42)
    for i in range(len(image_list)):
        if random.random() < 0.5:
            image_dataset["est"].append(image_list[i])
            color_dataset["est"].append(color_list[i])
            shape_dataset["est"].append(shape_list[i])
        else:
            image_dataset["eval"].append(image_list[i])
            color_dataset["eval"].append(color_list[i])
            shape_dataset["eval"].append(shape_list[i])
    print(f"Split: {len(image_dataset['est'])} in estimation set, {len(image_dataset['eval'])} in evaluation set.")

    patching_results = {}
    for model_id in model_id_list:
        model_name = model_id.replace('/', '_')
        filename = f"src/data/cma/color/{model_name}.json"
        file_path = Path(filename)
        if file_path.exists():
            print(f"Found {filename}! Loading results for keys intervention...")
            with open(filename, 'r') as f:
                patching_results[model_id] = json.load(f)
            continue

        config = load_config()
        tier = config['pipeline']['tier']
        model, processor = load_vlm(model_id, tier)    
        num_layers = get_num_hidden_layers(model)
        _, num_heads = _resolve_text_model_dims(model)
        _, num_kv_heads = _resolve_text_model_dims(model, kv_heads=True)
        mediation_scores_list = run_mediation_analysis(model_id)
        mediation_scores = mediation_scores_list[2]
        top_k_heads = get_top_k_heads(mediation_scores, 20)
        top_k_kv_heads = to_kv_heads(top_k_heads, num_heads, num_kv_heads)
        # print("top_k_kv_heads:", top_k_kv_heads)

        print(f"Calculating binding embeddings...")
        left_binding_embs, right_binding_embs = cma_position_keys(
            model=model, 
            processor=processor, 
            num_heads=num_kv_heads, 
            top_k_heads=top_k_kv_heads,
            image_list=image_dataset["est"],
            shape_list=shape_dataset["est"]
        )

        print(f"Patching embeddings...")
        left_patching_results, right_patching_results = get_patching_results(
            model=model, 
            processor=processor, 
            num_layers=num_layers, 
            num_heads=num_kv_heads, 
            top_k_heads=top_k_kv_heads, 
            left_binding_embs=left_binding_embs, 
            right_binding_embs=right_binding_embs,
            image_list=image_dataset["eval"],
            shape_list=shape_dataset["eval"],
            color_list=color_dataset["eval"]
        )
        
        patching_results[model_id] = {"left": left_patching_results, "right": right_patching_results}
        
        with open(filename, 'w') as f:
            # indent=4 formats it nicely to read it in a text editor
            json.dump(patching_results[model_id], f, indent=4)
        print(f"Keys intervention results successfully saved in {filename}.")

        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()

    # print("final patching_results: ", patching_results)


if __name__ == "__main__":
    main()