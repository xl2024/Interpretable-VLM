import gc
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Tuple, Any

from src.model.loader import load_vlm
from src.utils.tools import load_config, _resolve_text_model_dims, get_text_prompt, get_num_hidden_layers, predict
from src.plots.cma_1d import run_mediation_analysis
from src.mech_interp.cma import cma_head_patching_by_logits, get_head_embeddings, get_top_k_heads
from src.data.synthetic_generator import generate_custom_image


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

def get_intervention_results(model, processor, num_layers, num_heads, top_k_heads, ids_in_desc, color_list, shape_list):
    all_patching_results = {}
    
    before_correct = 0
    after_correct = 0
    all_count = 0

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
            d_t_head_cache = ids_in_desc[pos]
            first_tensor = next(iter(d_t_head_cache.values()))
            print(first_tensor.shape)
            d_o_head_cache = torch.zeros_like(first_tensor)
            print(d_o_head_cache.shape)

            prediction = predict(model, processor, image, text_prompt, new_only=True)
            pred_before = prediction.split()[0]

            predicted_word = cma_head_patching_by_logits(
                model=model,
                processor=processor,
                num_layers=num_layers,
                num_heads=num_heads,
                prompt_c1=text_prompt,
                image_c1=image,
                d_t_head_cache=d_t_head_cache,
                top_k_heads=top_k_heads,
                alpha=2,
                d_o_head_cache=d_o_head_cache
            )

            print(f"pos={pos}, i={i}, target={color_list[obj]}, before={pred_before}, after={predicted_word}")
            all_patching_results[pos].append([color_list[obj], pred_before.lower(), predicted_word.lower()])

            all_count += 1
            if pred_before.lower() == color_list[obj]:
                before_correct += 1
            if predicted_word.lower() == color_list[obj]:
                after_correct += 1

    print(f"Before: {before_correct}/{all_count}. After: {after_correct}/{all_count}")

    return all_patching_results


def main():
    model_id_list = [
        "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        #  "Qwen/Qwen2.5-VL-32B-Instruct",
        "llava-hf/llava-1.5-7b-hf"
        #  "llava-hf/llava-1.5-13b-hf"
    ]

    shape_list=['circle', 'star', 'plane', 'square', 'umbrella', 'triangle', 'sun', 'heart', 'cross']
    color_list=['red', 'gold', 'grey', 'blue', 'hotpink', 'lime', 'black', 'purple', 'darkorange']

    patching_results = {}
    for model_id in model_id_list:
        # model_name = model_id.replace('/', '_')
        # filename = f"src/data/cma/color/{model_name}.json"
        # file_path = Path(filename)
        # if file_path.exists():
        #     print(f"Found {filename}! Loading results for keys intervention...")
        #     with open(filename, 'r') as f:
        #         patching_results[model_id] = json.load(f)
        #     continue

        config = load_config()
        tier = config['pipeline']['tier']
        model, processor = load_vlm(model_id, tier)    
        num_layers = get_num_hidden_layers(model)
        _, num_heads = _resolve_text_model_dims(model)
        mediation_scores_list = run_mediation_analysis(model_id)
        mediation_scores = mediation_scores_list[1]
        top_k_heads = get_top_k_heads(mediation_scores, 100)

        print(f"Calculating position IDs in scene description task...")
        ids_in_desc = cma_position_IDs_in_desc(
            model=model, 
            processor=processor, 
            num_heads=num_heads, 
            top_k_heads=top_k_heads,
            color_list=color_list,
            shape_list=shape_list
        )

        print(f"Intervening with position IDs...")
        all_patching_results = get_intervention_results(
            model=model, 
            processor=processor, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            top_k_heads=top_k_heads, 
            ids_in_desc=ids_in_desc, 
            color_list=color_list,
            shape_list=shape_list
        )
        
        patching_results[model_id] = all_patching_results
        
        # with open(filename, 'w') as f:
        #     # indent=4 formats it nicely to read it in a text editor
        #     json.dump(patching_results[model_id], f, indent=4)
        # print(f"Keys intervention results successfully saved in {filename}.")

        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()

    # print("final patching_results: ", patching_results)


if __name__ == "__main__":
    main()