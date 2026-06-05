# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

import gc
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import random
from typing import Tuple, Any

from src.model.loader import load_vlm
from src.utils.tools import load_config, _resolve_text_model_dims, get_text_prompt, get_num_hidden_layers, get_model_id, to_kv_heads
from src.plots.cma_1d import run_mediation_analysis
from src.mech_interp.cma import cma_head_patching_by_logits, get_head_embeddings, get_top_k_heads
from src.data.synthetic_generator import generate_custom_image


def get_token_pos_for_object(
        model_id: str,
        inputs: dict,
        processor: Any,
        coord: Tuple[int, int],
        image_size: Tuple[int, int] = (336, 336), 
        cols: int = 2, 
        rows: int = 1
):
    model_id_lower = model_id.lower()
    if "qwen" in model_id_lower:
        # image_grid_thw shape is [1, 3] -> [Time, Height, Width]
        grid_t, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
        grid_h, grid_w = grid_h // 2, grid_w // 2
        vision_start_token = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        # Qwen image tokens start exactly ONE token after <|vision_start|>
        sequence_offset = (inputs["input_ids"][0] == vision_start_token).nonzero(as_tuple=True)[0].item() + 1
        # vision_end_token = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        # sequence_end = (inputs["input_ids"][0] == vision_end_token).nonzero(as_tuple=True)[0].item()
        # print(f"model_id: {model_id}, h: {grid_h}, w: {grid_w}, hxw: {grid_h*grid_w} start: {sequence_offset-1}, end: {sequence_end}, end-start: {sequence_end-sequence_offset+1}")
    elif "llava" in model_id_lower:
        grid_h, grid_w = 24, 24
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        sequence_offset = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0][0].item()
        # sequence_end = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0][-1].item()
        # print(f"model_id: {model_id}, h: {grid_h}, w: {grid_w}, hxw: {grid_h*grid_w} start: {sequence_offset}, end: {sequence_end}, end-start: {sequence_end-sequence_offset}")
    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    width, height = image_size
    row_idx, col_idx = coord
    
    # 1. Recreate the exact bounding box
    cell_width = width / cols
    cell_height = height / rows
    box_size = min(cell_width, cell_height) * 0.6
    half_size = int(box_size / 2)
    
    cx = int((col_idx + 0.5) * cell_width)
    cy = int((row_idx + 0.5) * cell_height)
    
    x0, y0 = cx - half_size, cy - half_size
    x1, y1 = cx + half_size, cy + half_size
    # bbox = [x0, y0, x1, y1]
    
    # col_min = max(0, int(x0 * grid_w / width))
    # col_max = min(grid_w - 1, int(x1 * grid_w / width))
    # row_min = max(0, int(y0 * grid_h / height))
    # row_max = min(grid_h - 1, int(y1 * grid_h / height))
    
    # simplify to left/right objects only and perform slightly better
    col_min = grid_w * col_idx // 2
    col_max = grid_w * (col_idx + 1) // 2 - 1
    row_min = 0
    row_max = grid_h - 1

    # 3. Flatten the 2D grid box into 1D sequence indices
    local_image_indices = []
    for r in range(row_min, row_max + 1):
        for c in range(col_min, col_max + 1):
            local_image_indices.append(r * grid_w + c)
            
    # 4. Add the LLM text offset
    global_indices = [sequence_offset + i for i in local_image_indices]
    
    return global_indices

def generate_dataset():
    color_set = ["red", "blue", "green"]
    shape_set = ["circle", "square", "triangle", "cross", "star", "heart", "sun", "umbrella", "plane"]
    # shape_set = ["circle", "square", "triangle"]
    color_list = []
    shape_list = []
    image_list = []
    for c1 in color_set:
        for c2 in color_set:
            if c1 == c2:
                continue
            for s1 in shape_set:
                for s2 in shape_set:
                    if s1 == s2:
                        continue
                    colors = [c1,c2]
                    shapes = [s1,s2]
                    color_list.append(colors)
                    shape_list.append(shapes)
                    image_list.append(generate_custom_image(shapes=shapes,colors=colors,coords=[(0,0),(0,1)]))
    
    return image_list, color_list, shape_list

def cma_position_keys(model, processor, num_heads, top_k_heads, image_list, shape_list):
    position_keys = {}
    for pos, key in enumerate(["left_target", "right_target"]):
        text_prompts = []
        token_pos_list = []

        for i in range(len(image_list)):
            prompt = f"In this image what is the color of the {shape_list[i][pos]}. Answer with the correct color only. Answer:"
            text_prompt = get_text_prompt(model, prompt, image_list[i], processor, use_system_prompt=False)
            text_prompts.append(text_prompt)

            inputs = processor(text=text_prompt, images=image_list[i], return_tensors="pt")
            token_pos_list.append(get_token_pos_for_object(get_model_id(model), inputs, processor, (0,pos)))
            
        # print("token_pos_list:", token_pos_list)
        position_keys[key] = get_head_embeddings(
            model=model, 
            processor=processor, 
            num_heads=num_heads, 
            prompt_list=text_prompts,
            image_list=image_list, 
            top_k_heads=top_k_heads,
            token_pos_list=token_pos_list,
            stage=4
        )

    return position_keys["left_target"], position_keys["right_target"]

def get_patching_results(model, processor, num_layers, num_heads, top_k_heads, left_binding_embs, right_binding_embs, image_list, shape_list, color_list):
    def print_results(patching_results, pos):
        matchings = sum(1 for pairs in patching_results if len(set(pairs)) == 1)
        print(f"patching_acc (position={pos}): {matchings}/{len(patching_results)}, patching_results: {patching_results}")
        # print(f"patching_acc (position={pos}): {matchings}/{len(patching_results)}")

    all_patching_results = {}
    
    for pos, key in enumerate(["left", "right"]):
        all_patching_results[key] = []

        for i in range(len(image_list)):
            prompt = f"In this image what is the color of the {shape_list[i][pos]}. Answer with the correct color only. Answer:"    # [Note: LLaVa would predict </s> (end of seq) without "Answer:" in prompt.]
            text_prompt = get_text_prompt(model, prompt, image_list[i], processor, use_system_prompt=False)
            inputs = processor(text=text_prompt, images=image_list[i], return_tensors="pt")
            token_pos = get_token_pos_for_object(get_model_id(model), inputs, processor, (0,pos))
            d_t = right_binding_embs if key == "left" else left_binding_embs
            d_o = left_binding_embs if key == "left" else right_binding_embs

            predicted_word = cma_head_patching_by_logits(
                model=model,
                processor=processor,
                num_layers=num_layers,
                num_heads=num_heads,
                prompt_c1=text_prompt,
                image_c1=image_list[i],
                d_t_head_cache=d_t,
                top_k_heads=top_k_heads,
                token_pos=token_pos,
                stage=4,
                alpha=2,
                d_o_head_cache=d_o
            )
            # print(f"i={i}, target={color_list[i][1-pos]}, other={color_list[i][pos]}, pred={predicted_word}")
            all_patching_results[key].append([color_list[i][1-pos], predicted_word.lower()])
        

        print_results(all_patching_results[key], key)

    return all_patching_results["left"], all_patching_results["right"]


def main():
    model_id_list = [
        "Qwen/Qwen2-VL-7B-Instruct",    # left: 61/212, right: 3/212    split: 432 total, 220 in est, 212 in eval
        "Qwen/Qwen2.5-VL-3B-Instruct",    # left: 135/212, right: 86/212
        "Qwen/Qwen2.5-VL-7B-Instruct",    # left: 70/212, right: 41/212
        #  "Qwen/Qwen2.5-VL-32B-Instruct",
        "llava-hf/llava-1.5-7b-hf",    # left: 16/212, right: 6/212
         "llava-hf/llava-1.5-13b-hf"
    ]

    image_list, color_list, shape_list = generate_dataset()
    print(f"Generated {len(image_list)} images.")
    
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

        patching_results[model_id] = {}
        for split in range(3):
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
            print(f"Split {split}: {len(image_dataset['est'])} in estimation set, {len(image_dataset['eval'])} in evaluation set.")

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
        
            patching_results[model_id][split] = {"left": left_patching_results, "right": right_patching_results}
            
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