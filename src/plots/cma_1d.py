import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Tuple, Any
from pathlib import Path
import itertools

from src.model.loader import load_vlm
from src.utils.tools import load_config, _resolve_text_model_dims, get_model_id, get_num_hidden_layers, get_permutations
from src.mech_interp.cma import run_cma_for_ID_retrieval, run_cma_for_ID_selection, run_cma_for_feature_retrieval

# Reproduces Figure 1d and 20-25

# model_id = "Qwen/Qwen2-VL-7B-Instruct"                      # Figure 1d
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct"                    # Figure 20
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct"                    # Figure 21
# model_id = "Qwen/Qwen2.5-VL-32B-Instruct"                   # Figure 22
# model_id = "llava-hf/llava-1.5-7b-hf"                       # Figure 23
# model_id = "llava-hf/llava-1.5-13b-hf"                      # Figure 24
# model_id = "bczhou/tiny-llava-v1-hf"
# model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"        # Figure 25
# model_id = "HuggingFaceM4/idefics2-8b-chatty"
# model_id = "HuggingFaceM4/idefics2-8b"

def run_mediation_analysis(model_id: str) -> Tuple[List[List[Any]], List[List[Any]], List[List[Any]]]:
    """
    Executes Causal Mediation Analysis (Activation Patching) across all attention heads.
    Patches activations from a modified context (c2) into the clean context (c1) following Eq. (1).
    """
    model_name = model_id.replace('/', '_')
    filename = f"src/data/cma/scores/{model_name}.npz"
    file_path = Path(filename)
    if file_path.exists():
        print(f"Found {filename}! Loading cma scores...")
        loaded_data = np.load(filename)
        mediation_scores_1 = loaded_data['mediation_scores_1']
        mediation_scores_2 = loaded_data['mediation_scores_2']
        mediation_scores_3 = loaded_data['mediation_scores_3']
    else:
        config = load_config()
        tier = config['pipeline']['tier']
        model, processor = load_vlm(model_id, tier)    
        num_layers = get_num_hidden_layers(model)
        _, num_heads = _resolve_text_model_dims(model)

        print("Preparing Causal Mediation Analysis...")

        shapes_list = ["circle", "square", "triangle"]
        colors_list = ["blue", "red", "green"]
        mediation_scores_1 = None
        mediation_scores_2 = None
        mediation_scores_3 = None
        count = 0

        for shapes in itertools.permutations(shapes_list, 2):
            for all_colors in get_permutations(colors_list):
                colors = all_colors[0:2]
                new_color = all_colors[-1]
                count += 1

                mediation_scores_1 = run_cma_for_ID_retrieval(model, processor, num_layers, num_heads, shapes, colors, mediation_scores_1)
                                        
                mediation_scores_2 = run_cma_for_ID_selection(model, processor, num_layers, num_heads, shapes, colors, mediation_scores_2)

                mediation_scores_3 = run_cma_for_feature_retrieval(model, processor, num_layers, num_heads, shapes, colors, new_color, mediation_scores_3)

        for l in range(num_layers):
            for h in range(num_heads):
                mediation_scores_1[l,h] /= count
                mediation_scores_2[l,h] /= count
                mediation_scores_3[l,h] /= count

        print("cma finished")
    
        np.savez(filename, 
                 mediation_scores_1=mediation_scores_1, 
                 mediation_scores_2=mediation_scores_2,
                 mediation_scores_3=mediation_scores_3
                 )
        print(f"cma scores successfully saved in {filename}.")

    return mediation_scores_1, mediation_scores_2, mediation_scores_3

def plot_causal_mediation(
    mediation_scores: Tuple[List[List[Any]], List[List[Any]], List[List[Any]]],
    save_paths: List[str]
):
    num_layers, num_heads = mediation_scores[0].shape
    raw_blue, raw_red, raw_green = mediation_scores

    raw_blue = np.clip(raw_blue, 0, None)
    raw_red = np.clip(raw_red, 0, None)
    raw_green = np.clip(raw_green, 0, None)
    
    scores_blue = raw_blue / raw_blue.max()
    scores_red = raw_red / raw_red.max()
    scores_green = raw_green / raw_green.max()

    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12) 
    
    ax.set_xticks(list(range(0, num_heads, 5)))
    ax.set_xlim(-0.5, num_heads - 0.5) 
    ax.set_yticks(list(range(0, num_layers, 5)))
    ax.set_ylim(-0.5, num_layers - 0.5)

    ax.set_facecolor('black')
    ax.grid(False)

    for l in range(num_layers):
        for h in range(num_heads):
            color = (scores_red[l,h], scores_green[l,h], scores_blue[l,h])
            rect = Rectangle((h - 0.5, l - 0.5), 1, 1, 
                             facecolor=color, edgecolor='none')
            ax.add_patch(rect)

    colors = {'blue': '#0000FF', 'red': '#FF0000', 'green': '#00FF00'}
    handles = [plt.Line2D([0], [0], color=colors[c], marker='s', markersize=10, linestyle='') 
               for c in ['blue', 'red', 'green']]
    labels = ['ID Retrieval', 'ID Selection', 'Feature Retrieval']
    legend = ax.legend(handles, labels, frameon=True, facecolor='lightgray', edgecolor='darkgray', 
                       loc='lower right', title='Stages', title_fontsize=11, fontsize=10)
    legend.get_title().set_color('black')
    for text in legend.get_texts():
        text.set_color('black')

    plt.tight_layout()
    plt.savefig(save_paths[0], dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Graph successfully saved to {save_paths[0]}")

    configs = [
        (raw_blue, save_paths[1], "#257EEA"),  # Blue gradient
        (raw_red, save_paths[2], "#E55353"),   # Red gradient
        (raw_green, save_paths[3], "#6DB36D")  # Green gradient
    ]
    
    for data, path, target_color in configs:
        figsize = (5, 8) if num_layers == 64 else (6, 6)
        tick_fontsize = 4 if num_heads == 40 else 6
        fig, ax = plt.subplots(figsize=figsize)
        cmap = LinearSegmentedColormap.from_list("custom", ["black", target_color])
        
        # imshow natively handles 2D matrices and perfectly pairs with colorbars
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        
        ax.set_xlabel('Head Index', fontsize=14, fontweight='bold')
        ax.set_ylabel('Layer Index', fontsize=14, fontweight='bold')
        
        ax.set_xticks(np.arange(num_heads))
        ax.set_yticks(np.arange(num_layers))
        ax.set_xticklabels(np.arange(num_heads), fontweight='bold', fontsize=tick_fontsize)
        ax.set_yticklabels(np.arange(num_layers), fontweight='bold', fontsize=tick_fontsize)
        
        # Create minor ticks shifted by 0.5 to draw the faint cell borders (Grid)
        ax.set_xticks(np.arange(-0.5, num_heads, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_layers, 1), minor=True)
        ax.grid(which='minor', color='#333333', linestyle='-', linewidth=0.5)
        ax.tick_params(which='both', bottom=False, left=False) # Hide physical major and minor tick lines
        
        divider = make_axes_locatable(ax)
        # Heatmap Width + (Heatmap Width * 0.05) + Padding(0.1 inches) = 100% Total Space
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('CMA Score', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_edgecolor('none')
        
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Individual channel graph saved to {path}")

def main():
    print("=== Execution Suite: Live Mechanistic Head Interventions ===")

    model_id_list = [
        # ("Qwen/Qwen2-VL-7B-Instruct", "1d"), 
        # ("Qwen/Qwen2.5-VL-3B-Instruct", "20"),
        # ("Qwen/Qwen2.5-VL-7B-Instruct", "21"),
        # ("Qwen/Qwen2.5-VL-32B-Instruct", "22"),
        ("llava-hf/llava-1.5-7b-hf", "23")
        # ("llava-hf/llava-1.5-13b-hf", "24"),
        # ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "25"),
        # ("HuggingFaceM4/idefics2-8b-chatty", "x"),
        # ("HuggingFaceM4/idefics2-8b", "x")
    ]
    for model_id, fig_num in model_id_list:
        mediation_scores = run_mediation_analysis(model_id)    # data stored in f"src/data/cma/scores/{model_name}.npz"

        model_name = model_id.replace('/', '_')
        plot_causal_mediation(
            mediation_scores=mediation_scores,
            save_paths=[
                f"outputs/cma/scores/cma_fig_{fig_num}_{model_name}.png",
                f"outputs/cma/scores/cma_fig_{fig_num}a_{model_name}.png",
                f"outputs/cma/scores/cma_fig_{fig_num}b_{model_name}.png",
                f"outputs/cma/scores/cma_fig_{fig_num}c_{model_name}.png"
            ]
        )


if __name__ == "__main__":
    main()