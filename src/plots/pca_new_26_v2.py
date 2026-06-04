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
# This script requires more Memory


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

def collect_hidden_states(model, processor, num_layers, dataset):
    layer_template = get_layer_path_template(model)
    states = {}
    for layer in range(num_layers):
        states[layer] = []
    rel_pos_labels = []
    abs_pos_labels = []
    feat_pos_labels = []
    is_central_labels = []
    count = 0
    for image_data in dataset:
        for last in range(4):
            print("count:",count)
            count += 1
            text = "In this image, there is a"
            for i in range(4):
                if i == last:
                    rel_pos_labels.append(image_data["rel_coords"][i])
                    abs_pos_labels.append(image_data["abs_coords"][i])
                    feat_pos_labels.append(image_data["shapes"][i])
                    is_central_labels.append(image_data["abs_coords"][i] == (1,1))
                else:
                    text += f" {image_data['colors'][i]} {image_data['shapes'][i]}, a"
            
            text = text[:-3] + " and a"
            text_prompt = get_text_prompt(model, text, image_data["image"], processor) 

            inputs = processor(text=text_prompt, images=image_data["image"], return_tensors="pt").to(model.device)

            with torch.no_grad():
                with model.trace() as tracer:
                    with tracer.invoke(**inputs):
                        for layer in range(num_layers):
                            layer_module = _resolve_layer_path(model, layer_template.format(layer))
                            if layer_module.output[0].ndim == 2:
                                states[layer].append(layer_module.output[0][-1, :].save())
                            elif layer_module.output[0].ndim == 3:
                                states[layer].append(layer_module.output[0][0, -1, :].save())
                            else:
                                raise AttributeError(f"layer_module.output[0].ndim={layer_module.output[0].ndim}")
                        
                gc_collect()
            
            for layer in range(num_layers):
                states[layer][-1] = states[layer][-1].cpu().to(torch.float32).numpy()
                
    return states, rel_pos_labels, abs_pos_labels, feat_pos_labels, is_central_labels

def plot_pca_grid(states_dict, labels, num_layers, title, save_path, mask=None):
    """
    Generates a grid of PCA subplots (5 layers per row) with a shared legend.
    """
    # 1. Filter data if a mask (like "is_central") is provided
    if mask is not None:
        mask_arr = np.array(mask)
        filtered_labels = [label for m, label in zip(mask_arr, labels) if m]
    else:
        filtered_labels = labels
        
    # 2. Extract unique labels and assign consistent colors
    unique_labels = sorted(list(set(filtered_labels)))
    cmap = plt.get_cmap("tab10")    # color map
    color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}

    # 3. Setup Figure Layout (5 columns per row)
    ncols = 5
    nrows = math.ceil(num_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    # 4. Process and Plot Each Layer
    for layer in range(num_layers):
        ax = axes[layer]
        
        # Stack all captured states for this layer into a single matrix
        layer_states = np.vstack(states_dict[layer])
        
        if mask is not None:
            layer_states = layer_states[mask_arr]
            
        # Run PCA to reduce hidden dimension to 2
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(layer_states)
        
        # Scatter plot colored by label
        for label in unique_labels:
            # Find indices for this specific label
            idx = [i for i, l in enumerate(filtered_labels) if l == label]
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

def generate_pca_figures(states, rel_pos, abs_pos, feat_pos, is_central, num_layers, save_path):
    """
    Executes the 5 specific PCA analyses
    """
    print("Generating PCA plots...")

    # Figure 26a: All objects by Relative Position
    plot_pca_grid(states, rel_pos, num_layers, 
                  title="All Objects: Relative Grid Position", 
                  save_path=os.path.join(save_path, "pca_fig_26a.png"))

    # Figure 26b: All objects by Absolute Position
    plot_pca_grid(states, abs_pos, num_layers, 
                  title="All Objects: Absolute Grid Position", 
                  save_path=os.path.join(save_path, "pca_fig_26b.png"))

    # Figure 26c: All objects by Semantic Features
    plot_pca_grid(states, feat_pos, num_layers, 
                  title="All Objects: Semantic Features", 
                  save_path=os.path.join(save_path, "pca_fig_26c.png"))

    # Figure 26d: Central objects ONLY by Relative Position
    plot_pca_grid(states, rel_pos, num_layers,
                  title="Central Object Only: Relative Grid Position", 
                  save_path=os.path.join(save_path, "pca_fig_26d.png"),
                  mask=is_central)

    # Figure 26e: Central objects ONLY by Semantic Identity
    plot_pca_grid(states, feat_pos, num_layers,
                  title="Central Object Only: Semantic Identity", 
                  save_path=os.path.join(save_path, "pca_fig_26e.png"),
                  mask=is_central)

def main():
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    config = load_config()
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)
    num_layers = get_num_hidden_layers(model)
    # num_layers = 6
    dataset = generate_dataset()
    states, rel_pos_labels, abs_pos_labels, feat_pos_labels, is_central_labels = collect_hidden_states(
        model, processor, num_layers, dataset
    )
    save_path = "outputs/pca"
    generate_pca_figures(states, rel_pos_labels, abs_pos_labels, feat_pos_labels, is_central_labels, num_layers, save_path)


if __name__ == "__main__":
    main()