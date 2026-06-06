import numpy as np
import gc
import torch
import pandas as pd
import json
from pathlib import Path
import math

from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import predict, get_num_hidden_layers, load_config, get_coord_from_index, is_equiv, _resolve_text_model_dims, get_permutations
from src.mech_interp.cma import cma_head_patching_by_generator, get_head_embeddings, get_top_k_heads
from src.plots.rsa_1c import get_dynamic_token_indices
from src.plots.cma_1d import run_mediation_analysis

# Reproduces Table 1 in Appendix D
# For 2x2 grid images


def cma_entr_get_embeds(model, processor, num_heads, color_list, shape_list, num_trials, top_k_heads):
    corr_trials = 0
    prompt_lists = {}
    image_lists = {}
    for pos in range(len(color_list)):
        prompt_lists[get_coord_from_index(pos)] = []
        image_lists[get_coord_from_index(pos)] = []
    for last_object in range(len(color_list)):
        for last_pos in range(len(color_list)):
            for c_shuffle in get_permutations([i for i in range(len(color_list)) if i != last_object]):
                c_shuffle.append(last_object)
                p_shuffle = [i for i in range(len(color_list)) if i != last_pos]
                p_shuffle.append(last_pos)
                shapes, colors = [], []
                for j in range(len(color_list)):
                    shapes.append(shape_list[c_shuffle[j]])
                    colors.append(color_list[c_shuffle[j]])

                coords = [get_coord_from_index(p_shuffle[j]) for j in range(len(color_list))]

                img = generate_custom_image(cols=2, rows=2, shapes=shapes, colors=colors, coords=coords)
                
                obj_indices, text_prompt = get_dynamic_token_indices(
                    model, processor, colors=colors, shapes=shapes, coords=coords, image=img, last_color=False, do_shffule=False
                )

                pred = predict(model, processor, img, text_prompt, max_new_tokens=10, new_only=True).split('.')[0].split()

                equiv_shapes = [
                    ['airplane', 'plane'],
                    ['x', 'cross'],
                    ['rectangle', 'square'],
                    ['light bulb', 'sun']
                    # ['dot', 'sun'] happens but shouldn't be equiv
                ]
                pred_color = pred[0].strip().lower()
                pred_shape = ' '.join(pred[1:]).strip().lower()
                if len(pred) >= 2 and pred_color == obj_indices[-1]['color'] and is_equiv(pred_shape, obj_indices[-1]['shape'], equiv_shapes):
                    corr_trials += 1
                    image_lists[obj_indices[-1]["coords"]].append(img)
                    prompt_lists[obj_indices[-1]["coords"]].append(text_prompt)
                else:
                    print(f"pred={pred}, target_color={obj_indices[-1]['color']}, target_shape={obj_indices[-1]['shape']}")
                
    high_entr_embeds = {}
    for pos in range(len(color_list)):
        coord = get_coord_from_index(pos)
        high_entr_embeds[coord] = get_head_embeddings(
            model=model, 
            processor=processor, 
            num_heads=num_heads, 
            prompt_list=prompt_lists[coord], 
            image_list=image_lists[coord], 
            top_k_heads=top_k_heads
        )
    return corr_trials, high_entr_embeds

def cma_entr_intervs(model, processor, num_layers, num_heads, color_list, shape_list, num_trials, top_k_heads, embeds):
    corr_trials = 0
    corr_trials_interv = 0
    for last_object in range(len(color_list)):
        for last_pos in range(len(color_list)):
            for c_shuffle in get_permutations([i for i in range(len(color_list)) if i != last_object]):
                c_shuffle.append(last_object)
                p_shuffle = [i for i in range(len(color_list)) if i != last_pos]
                p_shuffle.append(last_pos)
                shapes, colors = [], []
                for j in range(len(color_list)):
                    shapes.append(shape_list[c_shuffle[j]])
                    colors.append(color_list[c_shuffle[j]])

                coords = [get_coord_from_index(p_shuffle[j]) for j in range(len(color_list))]

                img = generate_custom_image(cols=2, rows=2, shapes=shapes, colors=colors, coords=coords)
                
                obj_indices, text_prompt = get_dynamic_token_indices(
                    model, processor, colors=colors, shapes=shapes, coords=coords, image=img, last_color=False, do_shffule=False
                )

                pred = predict(model, processor, img, text_prompt, max_new_tokens=10, new_only=True).split('.')[0].split()

                equiv_shapes = [
                    ['airplane', 'plane'],
                    ['x', 'cross'],
                    ['rectangle', 'square'],
                    ['light bulb', 'sun']
                    # ['dot', 'sun'] happens but shouldn't be equiv
                ]
                pred_color = pred[0].strip().lower()
                pred_shape = ' '.join(pred[1:]).strip().lower()
                if len(pred) >= 2 and pred_color == obj_indices[-1]['color'] and is_equiv(pred_shape, obj_indices[-1]['shape'], equiv_shapes):
                    corr_trials += 1
                else:
                    print(f"pred={pred}, target_color={obj_indices[-1]['color']}, target_shape={obj_indices[-1]['shape']}")
                
                predicted_word = cma_head_patching_by_generator(
                    model=model,
                    processor=processor,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    prompt_c1=text_prompt,
                    image_c1=img,
                    d_t_head_cache=embeds[obj_indices[-1]["coords"]],
                    top_k_heads=top_k_heads,
                    max_new_tokens=5
                )
                pred_interv = predicted_word[1].split('.')[0].split()
                pred_color_interv = pred_interv[0].strip().lower()
                pred_shape_interv = ' '.join(pred_interv[1:]).strip().lower()
                if len(pred_interv) >= 2 and pred_color_interv == obj_indices[-1]['color'] and is_equiv(pred_shape_interv, obj_indices[-1]['shape'], equiv_shapes):
                    corr_trials_interv += 1
                else:
                    print(f"pred_interv={pred_interv}, target_color={obj_indices[-1]['color']}, target_shape={obj_indices[-1]['shape']}")

    return corr_trials, corr_trials_interv

def cma_entr_by_model(model_id, num_trials, top_k):
    print("=== Starting Table 1 Reproduction ===")
    config = load_config()
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)
    num_layers = get_num_hidden_layers(model)
    _, num_heads = _resolve_text_model_dims(model)
    mediation_scores_list = run_mediation_analysis(model_id)
    mediation_scores = mediation_scores_list[1]
    top_k_heads = get_top_k_heads(mediation_scores, top_k)
    
    colors_list = [
        ['red', 'blue', 'green', 'purple'],
        ['green', 'blue', 'blue', 'green']
    ]
    shapes_list = [
        ['circle', 'square', 'triangle', 'heart'],
        ['triangle', 'square', 'triangle', 'square']
    ]
    
    print("Conducting high entropy trials...")
    corr_high, embeds = cma_entr_get_embeds(
        model, processor, num_heads, colors_list[0], shapes_list[0], num_trials, top_k_heads
    )    # high entr

    print("Conducting low entropy trials and interventions...")
    corr_low, corr_low_interv = cma_entr_intervs(
        model, processor, num_layers, num_heads, colors_list[1], shapes_list[1], num_trials, top_k_heads, embeds
    )    # low entr

    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    total_trials = math.factorial(len(colors_list[0])-1) * len(colors_list[0]) ** 2
    return corr_high, corr_low, corr_low_interv, total_trials

def cma_save_table(accs, save_path):
    processed_data = []
    for model_id, scores in accs.items():
        high_acc, low_no_int, low_with_int, total_trials = scores
        high_acc, low_no_int, low_with_int = high_acc/total_trials, low_no_int/total_trials, low_with_int/total_trials
        improvement = low_with_int - low_no_int
        processed_data.append([model_id, high_acc, low_no_int, low_with_int, improvement])

    columns = pd.MultiIndex.from_tuples([
        ("Model", ""),
        ("High Entropy", ""),
        ("Low Entropy", "No Intervention"),
        ("Low Entropy", "With Intervention"),
        ("Improvement", "")
    ])

    df = pd.DataFrame(processed_data, columns=columns)

    for col in df.columns:
        if col[0] != "Model":
            # escape the % sign with a backslash for LaTeX compatibility (\%)
            df[col] = df[col].apply(lambda x: f"{x*100:.2f}\\%")

    df_csv = df.copy()
    df_csv.columns = [
        "Model", 
        "High Entropy", 
        "Low Entropy (No Intervention)", 
        "Low Entropy (With Intervention)", 
        "Improvement"
    ]
    df_csv.to_csv(f"{save_path}.csv", index=False)
    print(f"CSV table saved to '{save_path}.csv'")

    df_csv.to_markdown(f"{save_path}.md", index=False)
    print(f"Markdown table saved to '{save_path}.md'")

def main():
    model_id_list = [
        ("llava-hf/llava-1.5-7b-hf", "LLaVA 1.5 7B"),
        # ("llava-hf/llava-1.5-13b-hf", "LLaVA 1.5 13B")
        ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen 2.5-VL 3B"),
        ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen 2.5-VL 7B"),
        ("Qwen/Qwen2-VL-7B-Instruct", "Qwen 2-VL 7B")
    ]
    # [Note: num_trials is not used for this experiment as once an object is fixed at a position, 
    # there are 3!=6 different combinations in total for all other 3 objects and 3 positions.]
    num_trials = 100
    # k_list = [2,5,10,20,50,100,200]    # sweeping with num_trials = 1
    k_list = [50]
    for top_k in k_list:
        accs = {}
        for model_id, model_label in model_id_list:
            model_name = model_id.replace('/', '_')
            filename = f"src/data/cma/entr_v2/{model_name}.json"
            file_path = Path(filename)
            if file_path.exists():
                print(f"Found {filename}! Loading results for entropy intervention...")
                with open(filename, 'r') as f:
                    accs[model_label] = json.load(f)
                continue

            accs[model_label] = cma_entr_by_model(model_id, num_trials, top_k)
            with open(filename, 'w') as f:
                json.dump(accs[model_label], f, indent=4)

        # save_path = f"outputs/cma/entr_v2/cma_tbl_1_k_{top_k}"    # for sweeping results
        save_path = f"outputs/cma/entr_v2/cma_tbl_1"
        cma_save_table(accs, save_path)


if __name__ == "__main__":
    main()