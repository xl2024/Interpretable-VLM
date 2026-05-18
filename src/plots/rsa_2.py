from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.math_core.rsa import compute_rsa_scores_2
from src.mech_interp.tracer import rsa_tracer_2
from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import predict, get_num_hidden_layers, load_config, get_text_prompt, get_permutations


# model_id = "bczhou/tiny-llava-v1-hf"
model_id = "Qwen/Qwen2-VL-7B-Instruct"

def plot_rsa_figure(
    rsa_scores_prompt: Dict[str, List[float]],
    save_path: str
):
    """
    Reproduces RSA figure 2
    """
    print("Generating RSA graph...")
    
    # Set up the figure with similar proportions to the paper
    fig, ax = plt.subplots(figsize=(6, 4))
    
    num_layers = len(rsa_scores_prompt['abs_pos'])
    layers = list(range(num_layers))
    
    # Plot the three lines
    ax.plot(layers, rsa_scores_prompt['feat'], label='Feature', color='blue')
    ax.plot(layers, rsa_scores_prompt['abs_pos'], label='Absolute Position', color='red')
    ax.plot(layers, rsa_scores_prompt['rel_pos'], label='Relative Position', color='green')
    
    # Style the axes
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(r'RSA Correlation (Pearson r)', fontsize=12) 
    
    # Add legend without the bounding box
    ax.legend(frameon=False, loc='upper left', fontsize=11)
    
    # Save to your outputs folder
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved successfully to {save_path}")
    
    # Display the graph
    plt.show()

def run_rsa_pipeline(
    model: Any,
    config: Dict[str, Any],
    num_layers: int,
    trials: List[Dict[str, Any]],
    save_path: str
) -> None:
    """
    computes the 3D correlation scores, and generates the paper's line graph.
    """
    # 1. Initialize storage for both token types
    hidden_states_by_trial = rsa_tracer_2(
        model,
        config,
        num_layers,
        trials
    )

    # 2. Math Execution
    print("Calculating RSA for Prompt Tokens...")
    rsa_scores_prompt = compute_rsa_scores_2(hidden_states_by_trial, trials, num_layers)
    
    # 3. Visualization
    plot_rsa_figure(
        rsa_scores_prompt=rsa_scores_prompt,
        save_path=save_path
    )
    
def get_dynamic_token_indices(processor: Any, colors: List[str], shapes: List[str], abs_coords: List[tuple[int, int]], rel_coords: List[tuple[int, int]], image: Image.Image, last_pos: int):
    """
    Dynamically calculates the exact sequence indices of the target objects
    by measuring token lengths, bypassing sub-word tokenization quirks.
    """
    prefix = "In this image, there is"
    for i in range(len(abs_coords)-1):
        if i != last_pos:
            prefix = f"{prefix} a {colors[i]} {shapes[i]},"
    prefix = f"{prefix} and a {colors[last_pos]}"

    text_prompt = get_text_prompt(model_id, prefix, image, processor)
    inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    input_ids = inputs["input_ids"][0].tolist()

    indices = {'abs_coords': abs_coords[last_pos], 
            'rel_coords': rel_coords[last_pos],
            'color': colors[last_pos],
            'shape': shapes[last_pos], 
            'index': len(input_ids)-1}
    return indices, text_prompt

def main():
    print("=== Starting Figure 2 RSA Reproduction ===")
    config = load_config()

    # 1. Load Model and Processor
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)

    num_layers = get_num_hidden_layers(model)

    # 2. Synthesize Trials for RSA
    # We generate a permutation matrix of shapes and colors to build the correlation variance.
    trials = []
    
    colors = ["red", "green", "purple", "blue"]
    shapes = ["circle", "square", "heart", "triangle"]
    abs_coords = [[(0, 0), (0, 1), (1, 0), (1, 1)],
                  [(0, 1), (1, 2), (1, 1), (0, 2)],
                  [(2, 0), (1, 1), (2, 1), (1, 0)],
                  [(1, 1), (1, 2), (2, 2), (2, 1)]]
    rel_coords = [[(0, 0), (0, 1), (1, 0), (1, 1)],
                  [(0, 0), (1, 1), (1, 0), (0, 1)],
                  [(1, 0), (0, 1), (1, 1), (0, 0)],
                  [(0, 0), (0, 1), (1, 1), (1, 0)]]
    # img_num = ['a', 'b', 'c', 'd']
    permutations = get_permutations([i for i in range(4)])
    all_permutations = []
    for p in permutations:
        for i in range(4):
            _colors = [colors[j] for j in p]
            _shapes = [shapes[j] for j in p]
            all_permutations.append({'colors': _colors, 'shapes': _shapes, 'abs_coords': abs_coords[i], 'rel_coords': rel_coords[i]})
    
    for p in all_permutations:
        img = generate_custom_image(
            cols=3, 
            rows=3, 
            shapes=shapes,
            colors=colors,
            coords=p['abs_coords'],
            # save_path=f'outputs/rsa_figure_2{img_num[p]}.png'
        )
        
        for last_pos in range(4):
            if p['abs_coords'][last_pos] != (1,1) and p['rel_coords'][last_pos] != (1,1):
                continue
            obj_indices, text_prompt = get_dynamic_token_indices(
                processor, colors=p['colors'], shapes=p['shapes'], abs_coords=p['abs_coords'], rel_coords=p['rel_coords'], image=img, last_pos=last_pos
            )

            # Process the inputs into PyTorch tensors
            inputs = processor(text=text_prompt, images=img, return_tensors="pt")
            # inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            trials.append({
                'inputs': inputs,
                'trial': obj_indices
            })

        # print(f"Prediction({len(trials)}): {predict(model, processor, img, text_prompt).strip()} (target: {obj_indices['shape']})")

    # 3. Execute Pipeline
    print(f"\nExecuting 3D RSA across {len(trials)} trials and {num_layers} layers...")
    run_rsa_pipeline(
        model=model,
        config=config,
        num_layers=num_layers,
        trials=trials,
        save_path="outputs/rsa_figure_2e.png"
    )

if __name__ == "__main__":
    main()
