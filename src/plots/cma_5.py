import gc
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

from src.model.loader import load_vlm
from src.utils.tools import load_config, _resolve_text_model_dims, get_text_prompt, get_num_hidden_layers, predict, get_coord_from_index, is_equiv
from src.plots.cma_1d import run_mediation_analysis
from src.mech_interp.cma import cma_head_patching_by_generator, get_head_embeddings, get_top_k_heads
from src.data.synthetic_generator import generate_custom_image

# Reproduces Figure 5


def process_intervention_data(results):
    """
    Parses the nested results dictionary and calculates the 'Before' and 'After'
    accuracy percentages for each direction.
    """
    directions = ["above", "below", "left", "right"]
    aggregated = {}
    
    for model_id, pos_data in results.items():
        # Initialize counts for this model
        stats = {d: {"before": 0, "after": 0, "total": 0} for d in directions}
        
        # Handle whether pos_data was saved as a list or a dict
        pos_iterable = pos_data.values() if isinstance(pos_data, dict) else pos_data
        
        for dir_data in pos_iterable:
            for d in directions:
                for target, before_pred, after_pred in dir_data[d]:
                    if target == before_pred.lower():
                        stats[d]["before"] += 1
                    if target == after_pred.lower():
                        stats[d]["after"] += 1
                    stats[d]["total"] += 1
        
        # Convert counts to percentages (0 to 100)
        agg_accs = {d: {"before": 0.0, "after": 0.0} for d in directions}
        for d in directions:
            total = stats[d]["total"]
            agg_accs[d]["before"] = (stats[d]["before"] / total) * 100.0
            agg_accs[d]["after"] = (stats[d]["after"] / total) * 100.0
                
        aggregated[model_id] = agg_accs
        
    return aggregated

def plot_intervention_grid(aggregated_data, save_path):
    """
    Plots the aggregated data in a 2x3 grid, placing the legend in the top right.
    """
    directions = ["above", "below", "left", "right"]
    models = list(aggregated_data.keys())
    
    # 1. Create a 3x3 grid
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(11, 9))
    axes = axes.flatten() # Flatten for easy indexing
    
    plot_indices = [0, 1, 3, 4, 5, 6, 7]    # index 2 for legend
    
    width = 0.35
    x = np.arange(len(directions))
    after_color = '#808080'
    before_color = '#333333'
    
    # 2. Plot the data for 5 models
    for idx, model_id in enumerate(models):
        ax = axes[plot_indices[idx]]
        
        before_accs = [aggregated_data[model_id][d]["before"] for d in directions]
        after_accs = [aggregated_data[model_id][d]["after"] for d in directions]
        
        # 'After' is plotted first (shifted left, colored gray)
        ax.bar(x - width/2, after_accs, width, color=after_color, label='After')
        # 'Before' is plotted second (shifted right, colored dark gray)
        ax.bar(x + width/2, before_accs, width, color=before_color, label='Before')
        
        # Formatting
        ax.set_title(model_id, fontsize=13)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_ylim(0, 100)
        
        ax.set_xticks(x)
        # ax.set_xticklabels(directions, rotation=30, ha='right', rotation_mode='anchor', fontsize=11)
        ax.set_xticklabels(directions, fontsize=12)
    
    # 3. Hijack the top-right subplot (axes[2]) to draw the shared legend
    ax_legend = axes[2]
    ax_legend.axis('off')
    
    # Create manual legend patches to guarantee colors match
    after_patch = mpatches.Patch(color=after_color, label='After')
    before_patch = mpatches.Patch(color=before_color, label='Before')
    
    ax_legend.legend(
        handles=[after_patch, before_patch], 
        title='Intervention',
        loc='center',
        fontsize=16, 
        title_fontsize=16,
        frameon=True
    )

    axes[-1].axis('off')
    
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) 
    print(f"Bar Chart Saved at: {save_path}")

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

    rel_ref = {}
    for rel in relations:
        if rel == "above":
            index = get_index_from_coord(coords, pos+3)
        elif rel == "below":
            index = get_index_from_coord(coords, pos-3)
        elif rel == "left":
            index = get_index_from_coord(coords, pos+1)
        else:    # "right"
            index = get_index_from_coord(coords, pos-1)
        
        rel_ref[rel] = f"{colors[index]} {shapes[index]}"
    
    return rel_ref

def is_equiv_color(color, target):
    equiv_colors = [
        ["orange", "yellow"]
    ]
    return is_equiv(color, target, equiv_colors)
    
def get_intervention_results(model, processor, num_layers, num_heads, top_k_heads, ids_in_desc, color_list, shape_list):
    all_patching_results = {}
    
    before_correct = {"above": 0, "below": 0, "left": 0, "right": 0}
    after_correct = {"above": 0, "below": 0, "left": 0, "right": 0}
    all_count = {"above": 0, "below": 0, "left": 0, "right": 0}

    for pos in range(9):
        all_patching_results[pos] = {"above": [], "below": [], "left": [], "right": []}

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

            rel_ref = get_rel_ref(colors, shapes, coords, pos)
            for RELATION, REF in rel_ref.items():
                prompt = f"In this image, what is the color of the object that is directly {RELATION} of {REF}. Answer with the relevant color only. Answer:"    # adding "Answer:" for LLaVa 1.5 models
                image = generate_custom_image(cols=3, rows=3, shapes=shapes, colors=colors, coords=coords)
                text_prompt = get_text_prompt(model, prompt, image, processor, use_system_prompt=False)
                d_t_head_cache = ids_in_desc[pos]
                d_o_head_cache = {}
                for l,h in d_t_head_cache.keys():
                    d_o_head_cache[l,h] = torch.zeros_like(d_t_head_cache[l,h])

                prediction = predict(model, processor, image, text_prompt, new_only=True)
                pred_before = prediction

                predicted_word = cma_head_patching_by_generator(
                    model=model,
                    processor=processor,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    prompt_c1=text_prompt,
                    image_c1=image,
                    d_t_head_cache=d_t_head_cache,
                    top_k_heads=top_k_heads,
                    alpha=2,
                    d_o_head_cache=d_o_head_cache,
                    max_new_tokens=5
                )
                predicted_word = predicted_word[1]

                all_patching_results[pos][RELATION].append([color_list[obj], pred_before, predicted_word])

                all_count[RELATION] += 1
                if is_equiv_color(pred_before.lower(), color_list[obj]):
                    before_correct[RELATION] += 1
                else:
                    print(f"pos={pos}, obj={obj}, RELATION={RELATION}, target={color_list[obj]}, before={pred_before}")

                if is_equiv_color(predicted_word.lower(), color_list[obj]):
                    after_correct[RELATION] += 1
                else:
                    print(f"pos={pos}, obj={obj}, RELATION={RELATION}, target={color_list[obj]}, after={predicted_word}")

    for rel in {"above", "below", "left", "right"}:
        print(f"{rel}: Before: {before_correct[rel]}/{all_count[rel]}. After: {after_correct[rel]}/{all_count[rel]}")

    return all_patching_results


def main():
    model_id_list = [
        ("llava-hf/llava-1.5-13b-hf", "Llava-1.5-13B"),
        ("llava-hf/llava-1.5-7b-hf", "Llava-1.5-7B"),
        ("Qwen/Qwen2-VL-7B-Instruct", "Qwen-2-VL"),
        ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen-2.5-VL-3B"),
        ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen-2.5-VL-7B"),
        ("Qwen/Qwen2.5-VL-32B-Instruct", "Qwen-2.5-VL-32B"),
        ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "Llava-Onevision")
    ]

    shape_list=['circle', 'star', 'plane', 'square', 'umbrella', 'triangle', 'sun', 'heart', 'cross']
    color_list=['red', 'yellow', 'gray', 'blue', 'pink', 'green', 'black', 'purple', 'orange']
    
    patching_results = {}
    for model_id, model_label in model_id_list:
        model_name = model_id.replace('/', '_')
        filename = f"src/data/cma/reuse/{model_name}.json"
        file_path = Path(filename)
        if file_path.exists():
            print(f"Found {filename}! Loading results for spatial reasoning intervention...")
            with open(filename, 'r') as f:
                patching_results[model_label] = json.load(f)
            continue

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
        
        patching_results[model_label] = all_patching_results
        
        with open(filename, 'w') as f:
            # indent=4 formats it nicely to read it in a text editor
            json.dump(patching_results[model_label], f, indent=4)
        print(f"Spatial reasoning intervention results successfully saved in {filename}.")

        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()

    save_path = "outputs/cma/cma_fig_5.png"
    aggregated = process_intervention_data(patching_results)
    plot_intervention_grid(aggregated, save_path)


if __name__ == "__main__":
    main()