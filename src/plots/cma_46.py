import numpy as np
import random
import matplotlib.pyplot as plt
import os
import re

from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import predict, get_num_hidden_layers, load_config, get_coord_from_index, _resolve_text_model_dims, get_text_prompt
from src.mech_interp.cma import cma_head_ablate_and_generate, get_top_k_heads
from src.plots.cma_1d import run_mediation_analysis

# Reproduces Figure 46 in Appendix C


def get_counting_dataset(num_imgs, save_path):
    dataset = []
    for i in range(num_imgs):
        colors_list = ['red', 'blue', 'green', 'yellow']
        shapes_list = ['triangle', 'square', 'circle', 'cross']
        feat_coords = [get_coord_from_index(j,n_cols=4) for j in range(16)]
        pos_coords = [get_coord_from_index(j) for j in range(9)]
        n_object = random.randint(3, 8)
        sampled_feat_coords = random.sample(feat_coords, n_object)
        coords = random.sample(pos_coords, n_object)
        colors = [colors_list[j[0]] for j in sampled_feat_coords]
        shapes = [shapes_list[j[1]] for j in sampled_feat_coords]
        save_name = os.path.join(save_path, f"{i}.png")
        img = generate_custom_image(cols=3, rows=3, shapes=shapes, colors=colors, coords=coords, save_path=save_name)
        dataset.append({"image": img, "index": i, "count": n_object})

    return dataset

def cma_counting_trials(model, processor, num_heads, dataset, top_k_heads):
    corr_trials = 0
    prompt = (
        "You are given an image containing multiple colored objects. "
        "Your task is to carefully observe the image and identify all the unique colored objects present.\n"
        "Enumerate all the unique colored objects you find in the image, providing a numbered list for clarity.\n"
        "After listing the objects, provide the total count of these unique colored objects.\n"
        "Format the total count by writing 'Answer:' followed by the number. "
        "It is crucial to adhere to this format: 'Answer: TOTAL_COUNT'."
    )
    for image_data in dataset:
        img, gt = image_data["image"], image_data["count"]
        text_prompt = get_text_prompt(model, prompt, img, processor, use_system_prompt=False)
        predicted_words = cma_head_ablate_and_generate(
            model=model,
            processor=processor,
            num_heads=num_heads,
            prompt_text=text_prompt,
            image=img,
            top_k_heads=top_k_heads,
            max_new_tokens=300
        )
        pred = predicted_words.split('Answer:')
        if len(pred) >= 2 and re.sub(r'\D', '',  pred[-1]) == str(gt):
            corr_trials += 1
        else:
            print(f"predicted_words={predicted_words}, \nindex={image_data['index']}, pred={pred}, target_count={gt}")
            # pass
        
    return corr_trials / len(dataset)

def cma_counting_by_model(model_id, k_list, dataset):
    print("=== Starting Figure 46 Reproduction ===")
    config = load_config()
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)
    num_layers = get_num_hidden_layers(model)
    _, num_heads = _resolve_text_model_dims(model)
    mediation_scores_list = run_mediation_analysis(model_id)
    mediation_scores = np.zeros((num_layers, num_heads))
    for l,h in [(a, b) for a in range(num_layers) for b in range(num_heads)]:
        mediation_scores[l,h] = max(mediation_scores_list[0][l,h], mediation_scores_list[1][l,h], mediation_scores_list[2][l,h])

    print("Max top-500 heads:")
    max_top_500_heads = get_top_k_heads(mediation_scores, 500)
    for i in range(500):
        print(f"{i+1},{mediation_scores[max_top_500_heads[i]]}", end=' ')
        if (i+1) % 10 == 0:
            print('\n')
    print("Bottom top-500 heads:")
    bottom_top_500_heads = get_top_k_heads(mediation_scores, 500, max_k=False)
    for i in range(500):
        print(f"{i+1},{mediation_scores[bottom_top_500_heads[i]]}", end=' ')
        if (i+1) % 10 == 0:
            print('\n')

    accs = {"Max": {}, "Bottom": {}}
    for cat in accs.keys():
        for k in k_list:
            print(f"{len(dataset)} trials for ablation of {cat} Top-{k} heads...")
            if cat == "Max":
                top_k_heads = get_top_k_heads(mediation_scores, k)
            else:
                top_k_heads = get_top_k_heads(mediation_scores, k, max_k=False)
            print(f"{cat} top-{k}:")
            accs[cat][k] = cma_counting_trials(model, processor, num_heads, dataset, top_k_heads)
            print(f"acc={accs[cat][k]}")

    return accs

def plot_counting_trials(accs, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor('white')
    ax.grid(True, color='lightgray', linestyle='-', linewidth=1, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgray')

    ax.plot(accs['Max'].keys(), accs['Max'].values(), 'o-', color='blue', label='Max Top-k')
    ax.plot(accs['Bottom'].keys(), accs['Bottom'].values(), 'o-', color='red', label='Bottom Top-k')

    ax.set_xlabel('Top k Heads', fontsize=12)
    ax.set_ylabel('Counting Accuracy', fontsize=12)
    ax.legend(loc='upper right', frameon=True, edgecolor='lightgray')
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph successfully saved to {save_path}")

def main():
    # model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    model_name = model_id.replace('/', '_')
    # k_list = [0,10,20,50,100,150,200,250,300,400,500]
    k_list = [10,100,250,500]
    num_imgs = 100
    dataset_path = "dataset/figure_46"
    os.makedirs(dataset_path, exist_ok=True)
    dataset = get_counting_dataset(num_imgs, dataset_path)
    accs = cma_counting_by_model(model_id, k_list, dataset)

    fig_path = f"outputs/cma/count/cma_fig_46_{model_name}.png"
    plot_counting_trials(accs, fig_path)


if __name__ == "__main__":
    main()