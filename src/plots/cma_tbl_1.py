import numpy as np
import gc
import torch
import pandas as pd

from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import predict, get_num_hidden_layers, load_config, get_coord_from_index, is_equiv, _resolve_text_model_dims
from src.mech_interp.cma import cma_head_patching_by_generator, get_head_embeddings, get_top_k_heads
from src.plots.rsa_1c import get_dynamic_token_indices
from src.plots.cma_1d import run_mediation_analysis

# Reproduces Table 1 in Appendix D


def cma_entr_trials(model, processor, num_layers, num_heads, color_list, shape_list, num_trials, get_embeds, interv, top_k_heads=None, embeds=None):
    corr_trials = 0
    if get_embeds:
        prompt_list = []
        image_list = []
    if interv:
        corr_trials_interv = 0
    for i in range(num_trials):
        shuffle = np.random.permutation(len(color_list))
        shapes, colors = [], []
        for j in range(len(color_list)):
            shapes.append(shape_list[shuffle[j]])
            colors.append(color_list[shuffle[j]])

        coords = [get_coord_from_index(j) for j in range(len(color_list))]

        img = generate_custom_image(cols=3, rows=3, shapes=shapes, colors=colors, coords=coords)
        
        obj_indices, text_prompt = get_dynamic_token_indices(
            model, processor, colors=colors, shapes=shapes, coords=coords, image=img, last_color=False
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
            if get_embeds:
                image_list.append(img)
                prompt_list.append(text_prompt)
        else:
            print(f"pred={pred}, target_color={obj_indices[-1]['color']}, target_shape={obj_indices[-1]['shape']}")
        
        if interv:
            predicted_word = cma_head_patching_by_generator(
                model=model,
                processor=processor,
                num_layers=num_layers,
                num_heads=num_heads,
                prompt_c1=text_prompt,
                image_c1=img,
                d_t_head_cache=embeds,
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

    if get_embeds:
        high_entr_embeds = get_head_embeddings(
            model=model, 
            processor=processor, 
            num_heads=num_heads, 
            prompt_list=prompt_list, 
            image_list=image_list, 
            top_k_heads=top_k_heads
        )
        return corr_trials, high_entr_embeds
    
    if interv:
        return corr_trials, corr_trials_interv
    
    return corr_trials

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
        ['red', 'yellow', 'gray', 'blue', 'pink', 'green', 'black', 'purple', 'orange'],
        ['red', 'green', 'blue', 'red', 'blue', 'red', 'blue', 'green', 'green']
    ]
    shapes_list = [
        ['circle', 'star', 'plane', 'square', 'umbrella', 'triangle', 'sun', 'heart', 'cross'],
        ['circle', 'circle', 'square', 'square', 'triangle', 'triangle', 'circle', 'triangle', 'square']
    ]

    print("Conducting high entropy trials...")
    corr_high, embeds = cma_entr_trials(
        model, processor, num_layers, num_heads, colors_list[0], shapes_list[0], num_trials, 
        get_embeds=True, interv=False, top_k_heads=top_k_heads
    )    # high entr

    print("Conducting low entropy trials and interventions...")
    corr_low, corr_low_interv = cma_entr_trials(
        model, processor, num_layers, num_heads, colors_list[1], shapes_list[1], num_trials, 
        get_embeds=False, interv=True, top_k_heads=top_k_heads, embeds=embeds
    )    # low entr

    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    return corr_high/num_trials, corr_low/num_trials, corr_low_interv/num_trials

def cma_save_table(accs, save_path):
    processed_data = []
    for model_id, scores in accs.items():
        high_acc, low_no_int, low_with_int = scores
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
        ("Qwen/Qwen2-VL-7B-Instruct", "Qwen 2-VL 7B"),
        ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen 2.5-VL 3B"),
        ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen 2.5-VL 7B"),
        ("llava-hf/llava-1.5-7b-hf", "LLaVA 1.5 7B")
        # ("llava-hf/llava-1.5-13b-hf", "LLaVA 1.5 13B")
    ]
    num_trials = 10
    k_list = [1,2,5,10,20,50,100,150,200]
    for top_k in k_list:
        # top_k = 10
        accs = {}
        for model_id, model_label in model_id_list:
            accs[model_label] = cma_entr_by_model(model_id, num_trials, top_k)

        save_path = f"outputs/cma/entr/cma_tbl_1_k_{top_k}"
        cma_save_table(accs, save_path)


if __name__ == "__main__":
    main()