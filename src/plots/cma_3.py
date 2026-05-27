import os
import glob
from PIL import Image
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

from src.model.loader import load_vlm
from src.utils.tools import load_config, _resolve_text_model_dims, get_text_prompt, get_num_hidden_layers, get_token_position
from src.plots.cma_1d import run_mediation_analysis
from src.mech_interp.cma import cma_head_patching_by_logits, get_head_embeddings, get_top_k_heads

# Reproduces Figure 3 and 37-43


def plot_cma_sweeping_results(model_results, save_path=None):
    """
    Parses the model_results dictionary and generates a 1x3 subplot figure
    """
    def _aggregate_for_accuracy(stage_data, k, alpha, pairs):
        """Helper function to calculate matches and safely add them to the dictionary."""
        k = int(k)
        alpha = float(alpha)
        
        if k not in stage_data:
            stage_data[k] = {}
        if alpha not in stage_data[k]:
            stage_data[k][alpha] = {"correct": 0, "total": 0}
            
        correct = sum(1 for gt, pred in pairs if gt == pred)
        
        stage_data[k][alpha]["correct"] += correct
        stage_data[k][alpha]["total"] += len(pairs)

    # parsed[stage][k][alpha] = {"correct": X, "total": Y}    
    parsed = {1: {}, 2: {}, 3: {}}

    for stage, stage_dict in model_results.items():            
        for k_val, k_dict in stage_dict.items():
            for direction, alpha_data in k_dict.items():
                for alpha_val, pairs in alpha_data.items():
                    _aggregate_for_accuracy(parsed[eval(stage)], k_val, alpha_val, pairs)

    stage_titles = {
        1: 'Id Retrieval Heads', 
        2: 'Id Selection Heads', 
        3: 'Feature Retrieval Heads'
    }
    
    # Find all unique 'K' values across all stages to build a consistent colormap
    all_ks = set()
    for stage_data in parsed.values():
        all_ks.update(stage_data.keys())
    sorted_ks = sorted(list(all_ks))
    
    # Build the colormap (viridis_r maps low K to yellow, and high K to dark purple)
    if len(sorted_ks) > 1:
        colors = {k: cm.viridis_r(i / (len(sorted_ks) - 1)) for i, k in enumerate(sorted_ks)}
    else:
        colors = {sorted_ks[0]: cm.viridis_r(0)} # Fallback if only 1 K exists

    # Setup the 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)    # sharey -> share_y
    fig.subplots_adjust(wspace=0.05, left=0.15) # Shrink whitespace, leave room for legend
    
    for stage in range(1,4):
        ax = axes[stage-1]
        stage_data = parsed.get(stage, {})
        
        for k in sorted_ks:
            if k in stage_data:
                alphas = []
                accs = []
                for alpha in sorted(stage_data[k].keys(), key=lambda x: float(x)):
                    # Log scales cannot plot 0
                    if float(alpha) > 0:
                        acc = stage_data[k][alpha]["correct"] / stage_data[k][alpha]["total"]
                        alphas.append(float(alpha))
                        accs.append(acc)
                
                ax.plot(alphas, accs, marker='o', markersize=5, color=colors[k], label=f'Top-K={k}')
        
        ax.set_title(stage_titles[stage], fontsize=12)
        ax.set_xscale('log')
        ax.set_xlabel('Magnitude of Intervention', fontsize=11)
        ax.grid(True, linewidth=0.7, alpha=0.8)
        
        # Add Legend and Y-label only to the leftmost plot
        if stage == 1:
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_ylim(-0.05, 1.05)
            
            # Extract legend handles, deduplicate them, and place outside the plot
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, 
                        loc='center right', bbox_to_anchor=(-0.15, 0.5), 
                        frameon=True, fontsize=10)
                        
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph successfully saved to {save_path}")
    # plt.show()

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
        # if random.random() > 0.1:
        #     continue

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
    left_token_pos_list = [] if stage == 1 else None
    right_token_pos_list = [] if stage == 1 else None

    for image_data in est_dataset:
        image_list.append(image_data["image"])
        left_prompt = f"In this image there is a {image_data['right_color']} {image_data['right_animal']} and a"
        left_text_prompt = get_text_prompt(model, left_prompt, image_data["image"], processor)
        text_prompts["left_target"].append(left_text_prompt)
        
        right_prompt = f"In this image there is a {image_data['left_color']} {image_data['left_animal']} and a"
        right_text_prompt = get_text_prompt(model, right_prompt, image_data["image"], processor)
        text_prompts["right_target"].append(right_text_prompt)

        if stage == 1:
            left_token_pos_1 = get_token_position(processor, left_text_prompt, image_data['image'], image_data['right_color'], False)
            left_token_pos_2 = get_token_position(processor, left_text_prompt, image_data['image'], image_data['right_animal'], False)
            left_token_pos_list.append([left_token_pos_1, left_token_pos_2])
            right_token_pos_1 = get_token_position(processor, right_text_prompt, image_data['image'], image_data['left_color'], False)
            right_token_pos_2 = get_token_position(processor, right_text_prompt, image_data['image'], image_data['left_animal'], False)
            right_token_pos_list.append([right_token_pos_1, right_token_pos_2])

    left_binding_embs = get_head_embeddings(
        model=model, 
        processor=processor, 
        num_heads=num_heads, 
        prompt_list=text_prompts["left_target"], 
        image_list=image_list, 
        top_k_heads=top_k_heads,
        token_pos_list=left_token_pos_list,
        stage=stage
    )

    right_binding_embs = get_head_embeddings(
        model=model, 
        processor=processor, 
        num_heads=num_heads, 
        prompt_list=text_prompts["right_target"], 
        image_list=image_list, 
        top_k_heads=top_k_heads,
        token_pos_list=right_token_pos_list,
        stage=stage
    )

    return left_binding_embs, right_binding_embs

def get_patching_results(model, processor, num_layers, num_heads, top_k_heads, left_binding_embs, right_binding_embs, stage, alpha_list, eval_dataset):
    def print_results(alpha, pos, patching_results):
        matchings = sum(1 for pairs in patching_results if len(set(pairs)) == 1)
        # print(f"patching_acc (alpha={alpha}, position={pos}): {matchings}/{len(patching_results)}, patching_results: {patching_results}")
        print(f"patching_acc (alpha={alpha}, position={pos}): {matchings}/{len(patching_results)}")

    left_patching_results = {}
    right_patching_results = {}
    for alpha in alpha_list:
        left_patching_results[str(alpha)] = []
        right_patching_results[str(alpha)] = []
        for image_data in eval_dataset:
            left_prompt = f"In this image there is a {image_data['right_color']} {image_data['right_animal']} and a"
            left_prompt_text = get_text_prompt(model, left_prompt, image_data["image"], processor)
            if stage == 1:
                left_token_pos_1 = get_token_position(processor, left_prompt_text, image_data['image'], image_data['right_color'], False)
                left_token_pos_2 = get_token_position(processor, left_prompt_text, image_data['image'], image_data['right_animal'], False)
                left_token_pos = [left_token_pos_1, left_token_pos_2]
            else:
                left_token_pos = [-1]
            left_d_t = right_binding_embs
            left_d_o = left_binding_embs
            predicted_word = cma_head_patching_by_logits(
                model=model,
                processor=processor,
                num_layers=num_layers,
                num_heads=num_heads,
                prompt_c1=left_prompt_text,
                image_c1=image_data["image"],
                d_t_head_cache=left_d_t,
                top_k_heads=top_k_heads,
                token_pos=left_token_pos,
                stage=stage,
                alpha=alpha,
                d_o_head_cache=left_d_o
            )
            left_patching_results[str(alpha)].append([image_data["right_color"], predicted_word])
        
            right_prompt = f"In this image there is a {image_data['left_color']} {image_data['left_animal']} and a"
            right_prompt_text = get_text_prompt(model, right_prompt, image_data["image"], processor)
            if stage == 1:
                right_token_pos_1 = get_token_position(processor, right_prompt_text, image_data['image'], image_data['left_color'], False)
                right_token_pos_2 = get_token_position(processor, right_prompt_text, image_data['image'], image_data['left_animal'], False)
                right_token_pos = [right_token_pos_1, right_token_pos_2]
            else:
                right_token_pos = [-1]
            right_d_t = left_binding_embs
            right_d_o = right_binding_embs
            predicted_word = cma_head_patching_by_logits(
                model=model,
                processor=processor,
                num_layers=num_layers,
                num_heads=num_heads,
                prompt_c1=right_prompt_text,
                image_c1=image_data["image"],
                d_t_head_cache=right_d_t,
                top_k_heads=top_k_heads,
                token_pos=right_token_pos,
                stage=stage,
                alpha=alpha,
                d_o_head_cache=right_d_o
            )
            right_patching_results[str(alpha)].append([image_data["left_color"], predicted_word])

        print_results(alpha, "left", left_patching_results[str(alpha)])
        print_results(alpha, "right", right_patching_results[str(alpha)])

    return left_patching_results, right_patching_results

def main():
    model_id_list = [("Qwen/Qwen2-VL-7B-Instruct", 40),                 # figure 40
                     ("Qwen/Qwen2.5-VL-3B-Instruct", 37),               # figure 37
                     ("Qwen/Qwen2.5-VL-7B-Instruct", 38),               # figure 38
                    #  ("Qwen/Qwen2.5-VL-32B-Instruct", 39),              # figure 39
                     ("llava-hf/llava-1.5-7b-hf", 41),                  # figure 41
                    #  ("llava-hf/llava-1.5-13b-hf", 42),                 # figure 42
                     ("llava-hf/llava-onevision-qwen2-7b-ov-hf", 43)    # figure 43
    ]
    k_list = [2,3,5,10,12,15,20,30,40,50,60,100,200]
    alpha_lists = [
        [1,5,10,15,20,30,50,100,150,200,300],
        [1,2,3,4,5,10,15,20,30,50,100],
        [1,2,3,10,15,20,50,100]
    ]

    print("Loading estimation dataset...")
    est_dataset = cma_loading_ue5_dataset("est")

    print("Loading evaluation dataset...")
    eval_dataset = cma_loading_ue5_dataset("eval")

    patching_results = {}
    for model_id, fig_num in model_id_list:
        model_name = model_id.replace('/', '_')
        filename = f"src/data/cma/sweeping/{model_name}.json"
        imgname = f"outputs/cma/sweeping/cma_fig_{fig_num}_{model_name}.png"
        file_path = Path(filename)
        if file_path.exists():
            print(f"Found {filename}! Loading sweeping results for hyperparameters...")
            with open(filename, 'r') as f:
                patching_results[model_id] = json.load(f)
            # plot_cma_sweeping_results(patching_results[model_id], imgname)
            continue

        config = load_config()
        tier = config['pipeline']['tier']
        model, processor = load_vlm(model_id, tier)    
        num_layers = get_num_hidden_layers(model)
        _, num_heads = _resolve_text_model_dims(model)
        mediation_scores_list = run_mediation_analysis(model_id)

        patching_results[model_id] = {}
        for stage in range(1, 4):
            # [Note: In stage 3 (feature retrival), patching the output of attn heads would let the model to predict the feature information in the patching embeddings, 
            # while patching the query embeddings asks the moddel about the feature of the position ID gotten from stage 2 and stored in the patchings.]
            mediation_scores = mediation_scores_list[stage-1]

            patching_results[model_id][str(stage)] = {}
            for k in k_list:
                top_k_heads = get_top_k_heads(mediation_scores, k)
                print(f"Calculating binding embeddings (stage={stage}, k={k})...")
                left_binding_embs, right_binding_embs = cma_binding_embeddings(
                    model=model, 
                    processor=processor, 
                    num_heads=num_heads, 
                    top_k_heads=top_k_heads, 
                    stage=stage,
                    est_dataset=est_dataset
                )

                print(f"Patching embeddings (stage={stage}, k={k})...")
                left_patching_results, right_patching_results = get_patching_results(
                    model=model, 
                    processor=processor, 
                    num_layers=num_layers, 
                    num_heads=num_heads, 
                    top_k_heads=top_k_heads, 
                    left_binding_embs=left_binding_embs, 
                    right_binding_embs=right_binding_embs, 
                    stage=stage,
                    alpha_list=alpha_lists[stage-1],
                    eval_dataset=eval_dataset
                )
                
                patching_results[model_id][str(stage)][str(k)] = {"left": left_patching_results, "right": right_patching_results}
        
        with open(filename, 'w') as f:
            # indent=4 formats it nicely to read it in a text editor
            json.dump(patching_results[model_id], f, indent=4)
        print(f"Sweeping results successfully saved in {filename}.")

        plot_cma_sweeping_results(patching_results[model_id], imgname)

    # print("final patching_results: ", patching_results)


if __name__ == "__main__":
    main()