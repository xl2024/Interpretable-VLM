from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.math_core.rsa import compute_rsa_scores
from src.mech_interp.tracer import rsa_tracer
from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import predict, get_num_hidden_layers, load_config, get_permutations, get_text_prompt


model_id = "Qwen/Qwen2-VL-7B-Instruct"
# model_id = "llava-hf/llava-1.5-7b-hf"
# model_id = "bczhou/tiny-llava-v1-hf"

def plot_rsa_figure_1c(
    rsa_scores_prompt: Dict[str, List[float]],
    rsa_scores_last_token: Dict[str, List[float]],
    save_path: str = "outputs/rsa_figure_1c.png"
):
    """
    Reproduces RSA figure 1c
    """
    print("Generating RSA graph...")
    
    # Set up the figure with similar proportions to the paper
    fig, ax = plt.subplots(figsize=(6, 4))
    
    num_layers = len(rsa_scores_prompt['pos'])
    layers = list(range(num_layers))
    
    # Plot the three lines
    ax.plot(layers, rsa_scores_prompt['pos'], label='Position (Prompt Tokens)', color='blue')
    ax.plot(layers, rsa_scores_last_token['pos'], label='Position (Last Token)', color='red')
    ax.plot(layers, rsa_scores_last_token['feat'], label='Feature (Last Token)', color='green')
    
    # Style the axes
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(r'Correlation ($r$)', fontsize=12) 
    
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
    save_path: str = "outputs/rsa_figure_1c.png"
) -> None:
    """
    computes the 3D correlation scores, and generates the paper's line graph.
    """
    # 1. Initialize storage for both token types
    hidden_states_by_trial = rsa_tracer(
        model,
        config,
        num_layers,
        trials
    )

    # 2. Math Execution
    print("Calculating RSA for Prompt Tokens...")
    rsa_scores_prompt, rsa_scores_last_token = compute_rsa_scores(hidden_states_by_trial, trials, num_layers)
    
    # 3. Visualization
    plot_rsa_figure_1c(
        rsa_scores_prompt=rsa_scores_prompt,
        rsa_scores_last_token=rsa_scores_last_token,
        save_path=save_path
    )
    
def get_dynamic_token_indices(processor: Any, colors: List[str], shapes: List[str], coords: List[tuple[int, int]], image: Image.Image):
    """
    Dynamically calculates the exact sequence indices of the target objects
    by measuring token lengths, bypassing sub-word tokenization quirks.
    """
    prefix = "In this image, there is"
    shuffle = np.random.permutation(len(coords))
    # last_object = {'color': 'red', 'shape': 'circle'}
    for i in range(len(coords)-1):
        prefix = f"{prefix} a {colors[shuffle[i]]} {shapes[shuffle[i]]},"
    prefix = f"{prefix} and a {colors[shuffle[-1]]}"

    text_prompt = get_text_prompt(model_id, prefix, image, processor)
    inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    input_ids = inputs["input_ids"][0].tolist()

    indices = []
    obj_idx = 0
    for token_index, token_id in enumerate(input_ids):
        token_str = processor.tokenizer.decode([token_id]).strip().lower()
        if ',' in token_str:
            indices.append({'coords': coords[shuffle[obj_idx]], 'color': colors[shuffle[obj_idx]], 'shape': shapes[shuffle[obj_idx]], 'index': token_index})
            obj_idx += 1
            print("Token index:", token_index, processor.tokenizer.decode([token_id-1]).strip().lower(), token_str)

    indices.append({'coords': coords[shuffle[-1]], 'color': colors[shuffle[-1]], 'shape': shapes[shuffle[-1]], 'index': len(input_ids)-1})
    return indices, text_prompt

def main():
    print("=== Starting Figure 1c RSA Reproduction ===")
    config = load_config()

    # 1. Load Model and Processor
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)

    num_layers = get_num_hidden_layers(model)

    # 2. Synthesize Trials for RSA
    # We generate a permutation matrix of shapes and colors to build the correlation variance.
    trials = []
    
    colors = ["red", "blue"]
    shapes = ["circle", "square"]
    
    objects = []
    for color in colors:
        for shape in shapes:
            objects.append({'color': color, 'shape': shape})
    print("all objects:", objects)
    
    permutations = get_permutations(objects)
    # To save memory in local mode, we will slice the first 10 permutations. 
    # Increase this for a smoother correlation curve.
    for p in permutations:
        o1, o2, o3, o4 = p

        shapes = [o1['shape'], o2['shape'], o3['shape'], o4['shape']]
        colors = [o1['color'], o2['color'], o3['color'], o4['color']]
        coords = [(0,0), (0,1), (1,0), (1,1)]

        img = generate_custom_image(
            cols=2, 
            rows=2, 
            shapes=shapes,
            colors=colors,
            coords=coords
        )
        
        obj_indices, text_prompt = get_dynamic_token_indices(
            processor, colors=colors, shapes=shapes, coords=coords, image=img
        )

        # Process the inputs into PyTorch tensors
        inputs = processor(text=text_prompt, images=img, return_tensors="pt")
        # inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        trials.append({
            'inputs': inputs,
            'trial': obj_indices
        })

        print(f"Prediction({len(trials)}): {predict(model, processor, img, text_prompt).strip()} (target: {obj_indices[-1]['shape']})")

    # 3. Execute Pipeline
    print(f"\nExecuting 3D RSA across {len(trials)} trials and {num_layers} layers...")
    run_rsa_pipeline(
        model=model,
        config=config,
        num_layers=num_layers,
        trials=trials,
        save_path="outputs/rsa_figure_1c.png"
    )

if __name__ == "__main__":
    main()
