import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Any
from pathlib import Path

from src.model.loader import load_vlm
from src.utils.tools import load_config, _resolve_text_model_dims, get_model_id
from src.plots.rsa_1c import get_num_hidden_layers
from src.mech_interp.cma import run_cma_for_ID_retrieval, run_cma_for_ID_selection, run_cma_for_feature_retrieval

# Reproduces Figure 1d and 20-25

# model_id = "Qwen/Qwen2-VL-7B-Instruct"                      # Figure 1d
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct"                    # Figure 20
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct"                    # Figure 21
model_id = "Qwen/Qwen2.5-VL-32B-Instruct"                   # Figure 22
# model_id = "llava-hf/llava-1.5-7b-hf"                       # Figure 23
# model_id = "llava-hf/llava-1.5-13b-hf"                      # Figure 24
# model_id = "bczhou/tiny-llava-v1-hf"
# model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"        # Figure 25
# model_id = "HuggingFaceM4/idefics2-8b-chatty"
# model_id = "HuggingFaceM4/idefics2-8b"

def run_mediation_analysis(
    model: Any,
    processor: Any,
    num_layers: int,
    num_heads: int
) -> Tuple[List[List[Any]], List[List[Any]], List[List[Any]]]:
    """
    Executes Causal Mediation Analysis (Activation Patching) across all attention heads.
    Patches activations from a modified context (c2) into the clean context (c1) following Eq. (1).
    """
    model_name = get_model_id(model).replace('/', '_')
    filename = f"src/data/cma/{model_name}.npz"
    file_path = Path(filename)
    if file_path.exists():
        print(f"Found {filename}! Loading cma scores...")
        loaded_data = np.load(filename)
        mediation_scores_1 = loaded_data['mediation_scores_1']
        mediation_scores_2 = loaded_data['mediation_scores_2']
        mediation_scores_3 = loaded_data['mediation_scores_3']
    else:
        print("Preparing Causal Mediation Analysis...")

        shapes = ["circle", "square"]
        colors = ["blue", "red"]

        mediation_scores_1 = run_cma_for_ID_retrieval(model, processor, num_layers, num_heads, shapes, colors)
                                
        mediation_scores_2 = run_cma_for_ID_selection(model, processor, num_layers, num_heads, shapes, colors)

        new_color = "green"
        mediation_scores_3 = run_cma_for_feature_retrieval(model, processor, num_layers, num_heads, shapes, colors, new_color)

        print("cma finished")
    
        np.savez(filename, 
                 mediation_scores_1=mediation_scores_1, 
                 mediation_scores_2=mediation_scores_2,
                 mediation_scores_3=mediation_scores_3
                 )
        print(f"cma scores saved in {filename} successfully.")

    return mediation_scores_1, mediation_scores_2, mediation_scores_3

def plot_causal_mediation(
    mediation_scores: Tuple[List[List[Any]], List[List[Any]], List[List[Any]]],
    num_layers: int,
    num_heads: int,
    save_path: str
):
    scores_blue, scores_red, scores_green = mediation_scores

    scores_blue = np.clip(scores_blue, 0, None)
    scores_red = np.clip(scores_red, 0, None)
    scores_green = np.clip(scores_green, 0, None)
    
    scores_blue /= scores_blue.max()
    scores_red /= scores_red.max()
    scores_green /= scores_green.max()

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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph successfully saved to {save_path}")
    plt.show()

def main():
    print("=== Execution Suite: Live Mechanistic Head Interventions ===")
    config = load_config()
    # model_id = "bczhou/tiny-llava-v1-hf"
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)    
    num_layers = get_num_hidden_layers(model)
    _, num_heads = _resolve_text_model_dims(model)

    mediation_scores = run_mediation_analysis(
        model=model,
        processor=processor,
        num_layers=num_layers,
        num_heads= num_heads
    )
    
    plot_causal_mediation(
        mediation_scores=mediation_scores,
        num_layers=num_layers,
        num_heads=num_heads,
        save_path="outputs/cma_figure_1d.png"
    )

if __name__ == "__main__":
    main()