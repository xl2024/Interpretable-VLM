import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import math
import os

from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import _resolve_layer_path, load_config, get_permutations, get_text_prompt, get_layer_path_template, get_coord_from_index, get_num_hidden_layers
from src.mech_interp.tracer import gc_collect

# Reproduces Figure 26 in the May version of the paper


def generate_dataset():
    color_list = ["red", "green", "purple", "blue"]
    shape_list = ["circle", "square", "heart", "triangle"]
    rel_coords = [get_coord_from_index(i, n_cols=2) for i in range(4)]
    glb_coords = [get_coord_from_index(i, n_cols=2) for i in range(4)]
    dataset = []
    permutations = get_permutations([i for i in range(4)])
    for p in permutations:
        colors = [color_list[p[i]] for i in range(4)]
        shapes = [shape_list[p[i]] for i in range(4)]
        for glb_c in glb_coords:
            coords = [(rel_coords[i][0]+glb_c[0], rel_coords[i][1]+glb_c[1]) for i in range(4)]
            image = generate_custom_image(cols=3, rows=3, shapes=shapes, colors=colors, coords=coords)
            dataset.append({"image": image, "colors": colors, "shapes": shapes, "rel_coords": rel_coords, "abs_coords": coords})
    
    return dataset

def collect_labels(model, processor, dataset):
    images = []
    text_prompts = []
    rel_pos_labels = []
    abs_pos_labels = []
    feat_pos_labels = []
    is_central_labels = []
    for image_data in dataset:
        for last in range(4):
            text = "In this image, there is a"
            for i in range(4):
                if i == last:
                    rel_pos_labels.append(image_data["rel_coords"][i])
                    abs_pos_labels.append(image_data["abs_coords"][i])
                    feat_pos_labels.append(image_data["shapes"][i])
                    is_central_labels.append(image_data["abs_coords"][i] == (1,1))
                else:
                    text += f" {image_data["colors"][i]} {image_data["shapes"][i]}, a"
            
            text = text[:-3] + " and a"
            text_prompt = get_text_prompt(model, text, image_data["image"], processor)
            images.append(image_data["image"])
            text_prompts.append(text_prompt)

    return images, text_prompts, rel_pos_labels, abs_pos_labels, feat_pos_labels, is_central_labels

def collect_pca_results(model, processor, num_layers, images, text_prompts, is_central_labels):
    layer_template = get_layer_path_template(model)
    pca_results = []
    pca_results_ctr = []
    for layer in range(num_layers):
        print(f"Processing layer {layer}...")
        states = []
        for i in range(len(images)):
            inputs = processor(text=text_prompts[i], images=images[i], return_tensors="pt").to(model.device)
            with torch.no_grad():
                with model.trace() as tracer:
                    with tracer.invoke(**inputs):
                        layer_module = _resolve_layer_path(model, layer_template.format(layer))
                        states.append(layer_module.output[0][-1, :].save())
                        
                gc_collect()
            
            states[-1] = states[-1].cpu().to(torch.float32).numpy()
        
        pca = PCA(n_components=2)
        proj = pca.fit_transform(states)
        pca_results.append(proj)

        states_ctr = []
        for i in range(states):
            if is_central_labels[i]:
                states_ctr.append(states[i])
        pca_ctr = PCA(n_components=2)
        proj_ctr = pca_ctr.fit_transform(states_ctr)
        pca_results_ctr.append(proj_ctr)

    return pca_results, pca_results_ctr

def plot_pca_grid(pca_results, labels, num_layers, title, save_path):
    """
    Generates a grid of PCA subplots (5 layers per row) with a shared legend.
    """ 
    unique_labels = sorted(list(set(labels)))
    cmap = plt.get_cmap("tab10")    # color map
    color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}

    ncols = 5
    nrows = math.ceil(num_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    # 4. Process and Plot Each Layer
    for layer in range(num_layers):
        ax = axes[layer]    
        pca_result = pca_results[layer]

        # Scatter plot colored by label
        for label in unique_labels:
            # Find indices for this specific label
            idx = [i for i, l in enumerate(labels) if l == label]
            ax.scatter(pca_result[idx, 0], pca_result[idx, 1], 
                       color=color_map[label], alpha=0.7, s=20)
            
        ax.set_title(f"Layer {layer}", fontsize=12)
        ax.set_xticks([]) # Remove ticks for clean PCA look
        ax.set_yticks([])

    # 5. Hide any unused subplots (if num_layers isn't divisible by 5)
    for i in range(num_layers, len(axes)):
        axes[i].axis('off')

    # 6. Create a Shared Legend at the very top of the figure
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label], markersize=10, label=str(label)) 
        for label in unique_labels
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02 + (0.05/nrows)), 
               ncol=len(unique_labels), fontsize=12, frameon=False)

    plt.suptitle(title, y=1.05 + (0.05/nrows), fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plots Saved at: {save_path}")

def generate_pca_figures(pca_results, pca_results_ctr, rel_pos, abs_pos, feat_pos, num_layers, save_path):
    """
    Executes the 5 specific PCA analyses
    """
    print("Generating PCA plots...")

    # Figure 26a: All objects by Relative Position
    plot_pca_grid(pca_results, rel_pos, num_layers, 
                  title="All Objects: Relative Grid Position", 
                  save_path=os.path.join(save_path, "pca_fig_26a.png"))

    # Figure 26b: All objects by Absolute Position
    plot_pca_grid(pca_results, abs_pos, num_layers, 
                  title="All Objects: Absolute Grid Position", 
                  save_path=os.path.join(save_path, "pca_fig_26b.png"))

    # Figure 26c: All objects by Semantic Features
    plot_pca_grid(pca_results, feat_pos, num_layers, 
                  title="All Objects: Semantic Features", 
                  save_path=os.path.join(save_path, "pca_fig_26c.png"))

    # Figure 26d: Central objects ONLY by Relative Position
    plot_pca_grid(pca_results_ctr, rel_pos, num_layers,
                  title="Central Object Only: Relative Grid Position", 
                  save_path=os.path.join(save_path, "pca_fig_26d.png"))

    # Figure 26e: Central objects ONLY by Semantic Identity
    plot_pca_grid(pca_results_ctr, feat_pos, num_layers,
                  title="Central Object Only: Semantic Identity", 
                  save_path=os.path.join(save_path, "pca_fig_26e.png"))

def main():
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    config = load_config()
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)
    # num_layers = get_num_hidden_layers(model)
    num_layers = get_num_hidden_layers(model) // 5
    dataset = generate_dataset()
    images, text_prompts, rel_pos_labels, abs_pos_labels, feat_pos_labels, is_central_labels = collect_labels(
        model, processor, dataset
    )
    pca_results, pca_results_ctr = collect_pca_results(model, processor, num_layers, images, text_prompts, is_central_labels)
    save_path = "outputs/pca"
    generate_pca_figures(pca_results, pca_results_ctr, rel_pos_labels, abs_pos_labels, feat_pos_labels, num_layers, save_path)


if __name__ == "__main__":
    main()