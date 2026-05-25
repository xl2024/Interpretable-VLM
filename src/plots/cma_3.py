import os
import glob
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path

from src.model.loader import load_vlm
from src.utils.tools import load_config, _resolve_text_model_dims, get_text_prompt, get_num_hidden_layers
from src.plots.cma_1d import run_mediation_analysis
from src.mech_interp.cma import cma_head_patching_by_logits, get_head_embeddings, get_top_k_heads

# Reproduces Figure 3 and 37-43

def cma_loading_ue5_dataset(split, dataset_dir="dataset/figure_3"):
    """
    Scans the dataset directory, parses filenames, and returns a list of dictionaries.
    The dataset was generated using UE5 in PUG-style as described in the paper. Details in dataset/figure_3/README.md.
    """
    dataset = []
    
    # Recursively find all .png files in the est/ and eval/ folders
    search_pattern = os.path.join(dataset_dir, split, "*.png")
    image_paths = glob.glob(search_pattern)
    
    for filepath in image_paths:
        # Get just the filename (e.g., "pug_red_camel_green_dolphin_salt_desert_1.png")
        filename = os.path.basename(filepath)
        
        # Remove the ".png" extension
        name_only = os.path.splitext(filename)[0]
        
        # Split the string by underscores
        parts = name_only.split('_')
        
        # parse and assign vars
        left_color = parts[1]    # parts[0] == "pug"
        left_animal = parts[2]
        right_color = parts[3]
        right_animal = parts[4]
        variation = parts[-1] 
        environment = "_".join(parts[5:-1]) # Joins 'salt' and 'desert' back together
        try:
            # use 'RGB' to ensure all images have 3 color channels (no alpha/transparency issues)
            img_data = Image.open(filepath).convert('RGB')
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            continue

        # Build the dictionary
        image_data = {
            "left_animal": left_animal,
            "left_color": left_color,
            "right_animal": right_animal,
            "right_color": right_color,
            "environment": environment,
            "variation": int(variation),
            "image": img_data
        }
        
        dataset.append(image_data)

    return dataset

def cma_binding_embeddings(model, processor, num_heads, top_k_heads, stage, est_dataset):
    text_prompts = {"left_target": [], "right_target": []}
    image_list = []
    for image_data in est_dataset:
        image_list.append(image_data["image"])
        left_prompt = f"In this image there is a {image_data['right_color']} {image_data['right_animal']} and a"
        text_prompts["left_target"].append(get_text_prompt(model, left_prompt, image_data["image"], processor))
        right_prompt = f"In this image there is a {image_data['left_color']} {image_data['left_animal']} and a"
        text_prompts["right_target"].append(get_text_prompt(model, right_prompt, image_data["image"], processor))

    left_binding_embs = get_head_embeddings(
        model=model, 
        processor=processor, 
        num_heads=num_heads, 
        prompt_list=text_prompts["left_target"], 
        image_list=image_list, 
        top_k_heads=top_k_heads,
        stage=stage
    )

    right_binding_embs = get_head_embeddings(
        model=model, 
        processor=processor, 
        num_heads=num_heads, 
        prompt_list=text_prompts["right_target"], 
        image_list=image_list, 
        top_k_heads=top_k_heads,
        stage=stage
    )

    return left_binding_embs, right_binding_embs

def get_patching_results(model, processor, num_layers, num_heads, top_k_heads, left_binding_embs, right_binding_embs, stage, alpha_list, eval_dataset):
    def print_results(alpha, pos, patching_results):
        matchings = sum(1 for pairs in patching_results if len(set(pairs)) == 1)
        patching_acc = matchings / len(patching_results)
        print(f"patching_acc (alpha={alpha}, position={pos}): {patching_acc}, patching_results: {patching_results}")

    left_patching_results = {}
    right_patching_results = {}
    for alpha in alpha_list:
        left_patching_results[alpha] = []
        right_patching_results[alpha] = []
        for image_data in eval_dataset:
            left_prompt = f"In this image there is a {image_data['right_color']} {image_data['right_animal']} and a"
            left_prompt_text = get_text_prompt(model, left_prompt, image_data["image"], processor)

            predicted_word = cma_head_patching_by_logits(
                model=model,
                processor=processor,
                num_layers=num_layers,
                num_heads=num_heads,
                prompt_c1=left_prompt_text,
                image_c1=image_data["image"],
                d_t_head_cache=right_binding_embs,
                top_k_heads=top_k_heads,
                stage=stage,
                alpha=alpha,
                d_o_head_cache=left_binding_embs
            )
            left_patching_results[alpha].append([image_data["right_color"], predicted_word])
        
            right_prompt = f"In this image there is a {image_data['left_color']} {image_data['left_animal']} and a"
            right_prompt_text = get_text_prompt(model, right_prompt, image_data["image"], processor)

            predicted_word = cma_head_patching_by_logits(
                model=model,
                processor=processor,
                num_layers=num_layers,
                num_heads=num_heads,
                prompt_c1=right_prompt_text,
                image_c1=image_data["image"],
                d_t_head_cache=left_binding_embs,
                top_k_heads=top_k_heads,
                stage=stage,
                alpha=alpha,
                d_o_head_cache=right_binding_embs
            )
            right_patching_results[alpha].append([image_data["left_color"], predicted_word])

        print_results(alpha, "left", left_patching_results[alpha])
        print_results(alpha, "right", right_patching_results[alpha])

    return left_patching_results, right_patching_results

def main():
    # model_id_list = ["Qwen/Qwen2-VL-7B-Instruct",                 # figure 40
    #                  "Qwen/Qwen2.5-VL-3B-Instruct",               # figure 37
    #                  "Qwen/Qwen2.5-VL-7B-Instruct",               # figure 38
    #                  "Qwen/Qwen2.5-VL-32B-Instruct",              # figure 39
    #                  "llava-hf/llava-1.5-7b-hf",                  # figure 41
    #                  "llava-hf/llava-1.5-13b-hf",                 # figure 42
    #                  "llava-hf/llava-onevision-qwen2-7b-ov-hf"    # figure 43
    # ]
    # k_list = [2,3,5,10,12,15,20,30,40,50,60,100]
    # alpha_lists = [
    #     [5,10,15,20,30,50,100,150,200,300],
    #     [1,2,3,4,5,10,15],
    #     [1,2,3,10,15,20,50,100]
    # ]

    model_id_list = ["Qwen/Qwen2-VL-7B-Instruct"                 # figure 40
    ]
    k_list = [10]
    alpha_lists = [
        [3],[3],[3]
    ]

    print("Loading estimation dataset...")
    est_dataset = cma_loading_ue5_dataset("est")

    print("Loading evaluation dataset...")
    eval_dataset = cma_loading_ue5_dataset("eval")

    patching_results = {}
    for model_id in model_id_list:
        config = load_config()
        tier = config['pipeline']['tier']
        model, processor = load_vlm(model_id, tier)    
        num_layers = get_num_hidden_layers(model)
        _, num_heads = _resolve_text_model_dims(model)
        mediation_scores_list = run_mediation_analysis(model_id)

        patching_results[model_id] = {}
        for stage in range(3):
            mediation_scores = mediation_scores_list[stage]

            patching_results[model_id][stage] = {}
            for k in k_list:
                top_k_heads = get_top_k_heads(mediation_scores, k)
                print(f"Calculating binding embeddings (stage={stage+1}, k={k})...")
                left_binding_embs, right_binding_embs = cma_binding_embeddings(
                    model=model, 
                    processor=processor, 
                    num_heads=num_heads, 
                    top_k_heads=top_k_heads, 
                    stage=stage+1,    # 0 base to 1 base
                    est_dataset=est_dataset
                )

                print(f"Patching embeddings (stage={stage+1}, k={k})...")
                left_patching_results, right_patching_results = get_patching_results(
                    model=model, 
                    processor=processor, 
                    num_layers=num_layers, 
                    num_heads=num_heads, 
                    top_k_heads=top_k_heads, 
                    left_binding_embs=left_binding_embs, 
                    right_binding_embs=right_binding_embs, 
                    stage=stage+1,    # 0 base to 1 base
                    alpha_list=alpha_lists[stage],
                    eval_dataset=eval_dataset
                )
                
                patching_results[model_id][stage][k] = {"left": left_patching_results, "right": right_patching_results}

    print("final patching_results: ", patching_results)

if __name__ == "__main__":
    main()