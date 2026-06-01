from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.math_core.rsa import compute_rsa_scores
from src.mech_interp.tracer import rsa_tracer
from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import predict, get_num_hidden_layers, load_config, get_coord_from_index
from src.plots.rsa_1c import get_dynamic_token_indices

# Reproduces Figure 6 and 30-36

# model_id = "Qwen/Qwen2-VL-7B-Instruct"                      # Figure 6 and 30
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct"                    # Figure 31
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct"                    # Figure 32
# model_id = "Qwen/Qwen2.5-VL-32B-Instruct"                   # Figure 33
# model_id = "llava-hf/llava-1.5-7b-hf"                       # Figure 34
# model_id = "llava-hf/llava-1.5-13b-hf"                      # Figure 35
# model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"        # Figure 36

def plot_rsa_figures(
    rsa_results: Dict[str, Dict[str, float]],
    num_layers: int,
    save_path: str
):
    print("Generating RSA graph...")
    
    for pos in ['Prompt', 'Last']:
        fig, ax = plt.subplots(figsize=(6, 4))
        layers = list(range(num_layers))
        colors = {'Low': 'blue', 'High': 'red'}
        for entr in ['Low', 'High']:
            ax.plot(layers, rsa_results[entr][pos], label=f'{entr} Entropy', color=colors[entr])
    
        titles = {'Prompt': 'Prompt Tokens', 'Last': 'Last Token'}
        ax.set_title(f"{titles[pos]} - Position RSA", fontweight="bold", fontsize=12, loc="center", pad=12)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Style the axes
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(r'RSA Correlation ($r$)', fontsize=12) 
        
        # Add legend without the bounding box
        ax.legend(frameon=False, loc='upper left', fontsize=11)
        
        # Save to your outputs folder
        plt.tight_layout()
        save_name = f'{save_path}_{pos.lower()}.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Graph saved successfully to {save_name}")
        
        # Display the graph
        plt.show()

def get_rsa_scores(model, processor, color_list, shape_list):
    trials = []
    
    for i in range(100):
        shuffle = np.random.permutation(len(color_list))
        shapes, colors = [], []
        for j in range(len(color_list)):
            shapes.append(shape_list[shuffle[j]])
            colors.append(color_list[shuffle[j]])

        coords = [get_coord_from_index(j) for j in range(len(color_list))]

        img = generate_custom_image(
            cols=3, 
            rows=3, 
            shapes=shapes,
            colors=colors,
            coords=coords
        )
        
        obj_indices, text_prompt = get_dynamic_token_indices(
            model, processor, colors=colors, shapes=shapes, coords=coords, image=img
        )

        inputs = processor(text=text_prompt, images=img, return_tensors="pt")
        trials.append({
            'inputs': inputs,
            'trial': obj_indices
        })

    return trials

def rsa_entr_by_model(model_id, save_path):
    print("=== Starting Figure 6 RSA Reproduction ===")
    config = load_config()
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)
    num_layers = get_num_hidden_layers(model)
    
    colors_list = [
        ['red', 'yellow', 'gray', 'blue', 'pink', 'green', 'black', 'purple', 'orange'],
        ['red', 'green', 'blue', 'red', 'blue', 'red', 'blue', 'green', 'green']
    ]
    shapes_list = [
        ['circle', 'star', 'plane', 'square', 'umbrella', 'triangle', 'sun', 'heart', 'cross'],
        ['circle', 'circle', 'square', 'square', 'triangle', 'triangle', 'circle', 'triangle', 'square']
    ]

    rsa_results = {}
    for i, entr in enumerate(['High', 'Low']):
        trials = get_rsa_scores(colors_list[i], shapes_list[i])
        
        print(f"\nExecuting 3D RSA across {len(trials)} trials and {num_layers} layers...")
        hidden_states_by_trial = rsa_tracer(model, config, num_layers, trials)

        print("Calculating RSA for Prompt Tokens...")
        rsa_scores_prompt, rsa_scores_last_token = compute_rsa_scores(hidden_states_by_trial, trials, num_layers)
        rsa_results[entr] = {"Prompt": rsa_scores_prompt['pos'], "Last": rsa_scores_last_token['pos']}

    plot_rsa_figures(
        rsa_results=rsa_results,
        num_layers=num_layers,
        save_path=save_path
    )

def main():
    model_id_list = [
        ("Qwen/Qwen2-VL-7B-Instruct", "6_30")
        # ("Qwen/Qwen2.5-VL-3B-Instruct", "31"),
        # ("Qwen/Qwen2.5-VL-7B-Instruct", "32"),
        # ("Qwen/Qwen2.5-VL-32B-Instruct", "33"),
        # ("llava-hf/llava-1.5-7b-hf", "34"),
        # ("llava-hf/llava-1.5-13b-hf", "35"),
        # ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "36"),    # scale up
    ]
    for model_id, fig_num in model_id_list:
        model_name = model_id.replace('/', '_')
        save_path = f"outputs/rsa/entr/rsa_fig_{fig_num}_{model_name}"
        rsa_entr_by_model(model_id, save_path)


if __name__ == "__main__":
    main()
