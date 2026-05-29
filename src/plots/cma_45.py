import os
import glob
import json
import torch
from PIL import Image
import torch
import random
import numpy as np
from typing import List, Dict, Tuple, Any


from src.model.loader import load_vlm
from src.utils.tools import load_config, get_text_prompt, get_num_hidden_layers, _resolve_text_model_dims, setup_dataset_from_zip, predict, get_token_position
from src.mech_interp.cma import cma_head_patching_by_generator, get_head_embeddings, get_top_k_heads
from src.plots.cma_1d import run_mediation_analysis


# def get_coco_objects(model, processor, coco_val_dir, cache_file, max_images=None):
def get_coco_objects(model, processor, coco_val_dir, cache_file, max_images=100):
    """
    Loads or generates the O_0 and O_1 objects for the COCO dataset, 
    filtering out cases where the model describes the same object twice.
    """
    # 1. Load from cache if it exists
    if os.path.exists(cache_file):
        print(f"Found existing cache at {cache_file}. Loading...")
        with open(cache_file, 'r') as f:
            object_mapping = json.load(f)
        print(f"Loaded {len(object_mapping)} valid image-objects pairs.")
        return object_mapping

    # 2. If no cache, generate from scratch
    print(f"No cache found. Generating object pairs from {coco_val_dir}...")
    
    search_pattern = os.path.join(coco_val_dir, "*.jpg")
    all_image_paths = glob.glob(search_pattern)
    
    if max_images:
        all_image_paths = all_image_paths[:max_images]
        
    object_mapping = {}
    prompt = "In this image there is 1. a"
    system_format = "OBJECT1 2. a OBJECT2, replacing OBJECT1 and OBJECT2 with the first and second object in the image, respectively. Do not repeat the prompt words, just append with the requested format"
    
    for idx, img_path in enumerate(all_image_paths):
        filename = os.path.basename(img_path)
        # print("filename:", filename)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Skipping corrupted image {filename}: {e}")
            continue

        prompt_text = get_text_prompt(model, prompt, img, processor, system_format)
        
        
        # We need to let it generate enough tokens to spit out two objects
        # e.g., " cat 2. a dog" -> approx 10 tokens
        raw_output = predict(model, processor, img, prompt_text, 25, True)
        # print("raw_output:", raw_output)
        
        # 3. Parse the response to extract O_0 and O_1
        # Example raw_output: "dog 2. a cat"
        # We split by '2. a' to isolate the nouns
        parts = raw_output.split("2. a")

        if len(parts) == 2 and len(parts[0].strip()) > 0 and len(parts[1].strip()) > 0:
            o_0 = parts[0].split(',')[0].split('.')[0].strip()
            o_1 = parts[1].split(',')[0].split('.')[0].strip()
            
            # Remove punctuation (commas, periods)
            _o_0 = ''.join(c for c in o_0.lower() if c.isalnum())    # alphanumeric A-Z, a-z, 0-9
            _o_1 = ''.join(c for c in o_1.lower() if c.isalnum())
            
            # 4. Filter out duplicates
            if _o_0 and _o_1 and (_o_0 != _o_1):
                # print(f"Got O_0: {o_0} O_1: {o_1}")
                object_mapping[filename] = {"O_0": o_0, "O_1": o_1}
            
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(all_image_paths)} images...")

    # 5. Save the generated mapping to disk
    print(f"Saving {len(object_mapping)} valid pairs to {cache_file}...")
    with open(cache_file, 'w') as f:
        json.dump(object_mapping, f, indent=4)
        
    return object_mapping

def run_cma_coco_unit(
    model: Any, 
    processor: Any, 
    coco_val_dir: str,
    source_filenames: List[str], 
    target_filenames: List[str], 
    top_k_heads: List[Tuple[int, int]],
    num_heads: int,
    num_layers: int,
    object_mapping: Dict[str, Dict[str, str]]
):
    # Reconstruct the full file paths
    source_paths = [os.path.join(coco_val_dir, f) for f in source_filenames]
    target_paths = [os.path.join(coco_val_dir, f) for f in target_filenames]
    
    # Load the physical images using PIL (converting to RGB for safety)
    print("Loading Source Images into memory...")
    source_images = [Image.open(p).convert('RGB') for p in source_paths]
    
    print("Loading Target Images into memory...")
    target_images = [Image.open(p).convert('RGB') for p in target_paths]

    print(f"Starting COCO Experiment with {len(source_images)} source images and {len(target_images)} target images.")
    print("Estimating average position ID (IDO_0) from the Source Set...")
    
    source_prompt = "In this image there is 1. a"
    system_format = "OBJECT1 2. a OBJECT2, replacing OBJECT1 and OBJECT2 with the first and second object in the image, respectively. Do not repeat the prompt words, just append with the requested format"
    source_prompt_texts = [
        get_text_prompt(model, source_prompt, img, processor, system_format) for img in source_images
    ]
    
    estimated_id_embeddings = get_head_embeddings(
        model=model,
        processor=processor,
        num_heads=num_heads,
        prompt_list=source_prompt_texts,
        image_list=source_images,
        top_k_heads=top_k_heads
    )
    
    print("Performing the Intervention on the Target Set...")
    
    unit_results = []
    for i in range(len(target_images)):
        img = target_images[i]
        o_0 = object_mapping[target_filenames[i]]['O_0']
        o_1 = object_mapping[target_filenames[i]]['O_1']
        
        intervention_prompt = f"In this image there is 1. a {o_0} 2. a"
        intervention_system_format = "OBJECT, replacing OBJECT with the second object in the image. Do not repeat the prompt words, just append with the requested format"
        intervention_prompt_text = get_text_prompt(model, intervention_prompt, img, processor, intervention_system_format)
        token_pos = get_token_position(processor, intervention_prompt_text, img, intervention_prompt_text[-1], False)
        
        predicted_words = cma_head_patching_by_generator(
            model=model,
            processor=processor,
            num_layers=num_layers,
            num_heads=num_heads,
            prompt_c1=intervention_prompt_text,
            image_c1=img,
            d_t_head_cache=estimated_id_embeddings,
            top_k_heads=top_k_heads,
            token_pos=[token_pos,token_pos],
            max_new_tokens = 10
        )
        
        predicted = predicted_words[1].split(',')[0].split('.')[0].strip()
        # print(f"target_filename: {target_filenames[i]}")
        # print(f"o_0: {o_0} o_1: {o_1} predicted_word: {predicted}")
        unit_results.append((target_filenames[i], o_0, o_1, predicted))
    
    return unit_results

def run_cma_coco(model_id, k_list, coco_val_dir, cache_dir, num_splits=3):
    model_name = model_id.replace('/', '_')
    cache_name = os.path.join(cache_dir, f"coco_results_{model_name}.json")
    coco_results = {}
    if os.path.exists(cache_name):
        print(f"Found existing cache at {cache_name}. Loading...")
        with open(cache_name, 'r') as f:
            coco_results = json.load(f)
        if list(coco_results.keys()) == [str(k) for k in k_list]:
            return coco_results
    
    config = load_config()
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)    
    num_layers = get_num_hidden_layers(model)
    _, num_heads = _resolve_text_model_dims(model)
    mediation_scores_list = run_mediation_analysis(model_id)
    mediation_scores = mediation_scores_list[1]

    mapping_cache_name = os.path.join(cache_dir, f"coco_objects_{model_name}.json")
    object_mapping = get_coco_objects(model, processor, coco_val_dir, mapping_cache_name)

    for k in k_list:
        if str(k) in coco_results:
            continue

        top_k_heads = get_top_k_heads(mediation_scores, k)

        valid_filenames = list(object_mapping.keys())
        print(f"Total valid images available for splitting: {len(valid_filenames)}")
        
        coco_results_k = {}

        for split_idx in range(num_splits):
            random.seed(42 + split_idx)
            shuffled_files = valid_filenames.copy()
            random.shuffle(shuffled_files)
            midpoint = len(shuffled_files) // 2
            source_filenames = shuffled_files[:midpoint]
            target_filenames = shuffled_files[midpoint:]

            unit_results = run_cma_coco_unit(
                model=model,
                processor=processor,
                coco_val_dir=coco_val_dir,
                source_filenames=source_filenames,
                target_filenames=target_filenames,
                top_k_heads=top_k_heads,
                num_heads=num_heads,
                num_layers=num_layers,
                object_mapping=object_mapping
            )
            
            coco_results_k[str(split_idx)] = unit_results

        coco_results[str(k)] = coco_results_k

        print(f"Saving coco_results (k={k}) to {cache_name}...")
        with open(cache_name, 'w') as f:
            json.dump(coco_results, f, indent=4)

    return coco_results

def get_coco_stats(coco_results, num_splits=3):
    split_accuracies = {}

    for k, coco_results_k in coco_results.items():
        split_accuracies_k = []

        for split_idx in range(num_splits):
            unit_results = coco_results_k[str(split_idx)]
            successful_repeats = 0
            for fn, o_0, o_1, pred in unit_results:
                clean_prediction = pred.strip().lower()
                clean_o_0 = o_0.lower()
                
                if clean_prediction == clean_o_0:
                    successful_repeats += 1
                    
            mean_accuracy = successful_repeats / len(unit_results)
            print(f"Intervention Complete! Total Accuracy: {mean_accuracy * 100:.2f}%")
        
            split_accuracies_k.append(mean_accuracy)
        
        # 5. Calculate Final Statistics (Mean and Error)
        mean_acc = np.mean(split_accuracies_k)
        std_err = np.std(split_accuracies_k)    # for the error bar
        split_accuracies[str(k)] = {"mean_acc": mean_acc, "std_err": std_err}
        
        print(f"Accuracies across {num_splits} splits (k={k}): {[f'{acc*100:.2f}%' for acc in split_accuracies_k]}")
        print(f"Mean Accuracy (k={k}): {mean_acc*100:.2f}% ± {std_err*100:.2f}%")
    
    return split_accuracies

def main():
    data_url = "http://images.cocodataset.org/zips/val2017.zip"
    target_dir = "./dataset/coco"
    dataset_name = "COCO Val2017 (~778MB)"
    coco_directory = setup_dataset_from_zip(dataset_name, data_url, target_dir)
    if coco_directory is None:
        print("No coco_directory.")
        return
    
    # model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    # model_id = "Qwen/Qwen2.5-VL-7B-Instruct"    # 658/1000 -> 329 source + 329 target
    # model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    model_id = "llava-hf/llava-1.5-7b-hf"    # 843/1000 -> 421 source + 422 target
    # model_id = "llava-hf/llava-1.5-13b-hf"
    k_list = [50,100,200]
    cache_dir = "src/data/cma/coco"
    coco_results = run_cma_coco(model_id, k_list, coco_directory, cache_dir)
    coco_stats = get_coco_stats(coco_results)


if __name__ == "__main__":
    main()