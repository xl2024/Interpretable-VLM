import os
import glob
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path

from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import load_config, _resolve_text_model_dims, get_text_prompt, get_num_hidden_layers
from src.plots.cma_1d import run_mediation_analysis
from src.mech_interp.cma import cma_head_patching_by_logits, get_head_embeddings, get_top_k_heads


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

def cma_binding_embeddings(model, processor, num_heads, top_k_heads, est_dataset):
    text_prompts = {"left_target": [], "right_target": []}
    image_list = []
    for image_data in est_dataset:
        image_list.append(image_data["image"])
        left_prompt = f"In this image there is a {image_data["right_color"]} {image_data["right_animal"]} and a"
        text_prompts["left_target"].append(get_text_prompt(model, left_prompt, image_data["image"], processor))
        right_prompt = f"In this image there is a {image_data["left_color"]} {image_data["left_animal"]} and a"
        text_prompts["right_target"].append(get_text_prompt(model, right_prompt, image_data["image"], processor))

    left_binding_embs = get_head_embeddings(
        model=model, 
        processor=processor, 
        num_heads=num_heads, 
        prompt_list=text_prompts["left_target"], 
        image_list=image_list, 
        top_k_heads=top_k_heads
    )

    right_binding_embs = get_head_embeddings(
        model=model, 
        processor=processor, 
        num_heads=num_heads, 
        prompt_list=text_prompts["right_target"], 
        image_list=image_list, 
        top_k_heads=top_k_heads
    )

    return left_binding_embs, right_binding_embs

def get_patching_results(model, processor, num_layers, num_heads, top_k_heads, left_binding_embs, right_binding_embs, eval_dataset):
    left_patching_results = []
    right_patching_results = []
    for image_data in eval_dataset:
        left_prompt = f"In this image there is a {image_data["right_color"]} {image_data["right_animal"]} and a"
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
            alpha=3,
            d_o_head_cache=left_binding_embs
        )
        left_patching_results.append([image_data["right_color"], predicted_word])
        
        right_prompt = f"In this image there is a {image_data["left_color"]} {image_data["left_animal"]} and a"
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
            alpha=3,
            d_o_head_cache=right_binding_embs
        )
        right_patching_results.append([image_data["left_color"], predicted_word])
        
    return left_patching_results, right_patching_results

def main():
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    
    stage = 2
    mediation_scores = run_mediation_analysis(model_id)
    mediation_scores = mediation_scores[stage-1]
    k = 1
    top_k_heads = get_top_k_heads(mediation_scores, k)

    config = load_config()
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)    
    num_layers = get_num_hidden_layers(model)
    _, num_heads = _resolve_text_model_dims(model)

    est_dataset = cma_loading_ue5_dataset("est")
    left_binding_embs, right_binding_embs = cma_binding_embeddings(
        model=model, 
        processor=processor, 
        num_heads=num_heads, 
        top_k_heads=top_k_heads, 
        est_dataset=est_dataset
    )
 
    eval_dataset = cma_loading_ue5_dataset("eval")
    left_patching_results, right_patching_results = get_patching_results(
        model=model, 
        processor=processor, 
        num_layers=num_layers, 
        num_heads=num_heads, 
        top_k_heads=top_k_heads, 
        left_binding_embs=left_binding_embs, 
        right_binding_embs=right_binding_embs, 
        eval_dataset=eval_dataset
    )

    print("left_patching_results:", left_patching_results)
    print("right_patching_results", right_patching_results)

