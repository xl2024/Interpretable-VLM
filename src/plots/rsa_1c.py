from typing import Dict, List, Any
import matplotlib.pyplot as plt
import yaml
import itertools

from src.math_core.rsa import compute_rsa_scores
from src.mech_interp.tracer import rsa_tracer
from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_spatial_binding_image

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
    metadata_list: List[List[Dict[str, Any]]],
    save_path: str = "outputs/rsa_figure_1c.png"
) -> None:
    """
    computes the 3D correlation scores, and generates the paper's line graph.
    """
    # 1. Initialize storage for both token types
    hidden_states_prompt_by_trial, hidden_states_last_by_trial = rsa_tracer(
        model,
        config,
        num_layers,
        metadata_list,
        trials
    )

    # 2. Math Execution
    print("Calculating RSA for Prompt Tokens...")
    rsa_scores_prompt = compute_rsa_scores(hidden_states_prompt_by_trial, metadata_list, num_layers)
    
    print("Calculating RSA for the Last Token...")
    rsa_scores_last_token = compute_rsa_scores(hidden_states_last_by_trial, metadata_list, num_layers)
    
    # 3. Visualization
    plot_rsa_figure_1c(
        rsa_scores_prompt=rsa_scores_prompt,
        rsa_scores_last_token=rsa_scores_last_token,
        save_path=save_path
    )


def load_config(config_path: str = "configs/local.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def get_dynamic_token_indices(processor: Any, left_color: str, left_shape: str, right_color: str):
    """
    Dynamically calculates the exact sequence indices of the target objects
    by measuring token lengths, bypassing sub-word tokenization quirks.
    """
    # 1. Build the string up to the first target (Left Shape)
    prefix_left = f"<image>\nIn this image, there is a {left_color} {left_shape},"
    
    # 2. Build the full string up to the second target (Right Color)
    full_prompt = f"{prefix_left} and a {right_color}"
    
    # Encode both strings.
    tokens_left = processor.tokenizer.encode(prefix_left)
    
    # The target index is simply the length of the sequence minus 1 (to be 0-indexed)
    idx_left = len(tokens_left) - 1
    
    return [idx_left], full_prompt

def get_num_hidden_layers(model: Any) -> int:
    """
    Resolve decoder layer count across wrapped/unwrapped VLM model objects.
    """
    # Typical HF multimodal configs (e.g., LlavaForConditionalGeneration)
    if hasattr(model, "config") and hasattr(model.config, "text_config"):
        return model.config.text_config.num_hidden_layers

    # Some wrappers expose the nested module path directly
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return len(model.model.language_model.layers)

    # Legacy/alternate wrapper pattern
    if (
        hasattr(model, "local_model")
        and hasattr(model.local_model, "config")
        and hasattr(model.local_model.config, "text_config")
    ):
        return model.local_model.config.text_config.num_hidden_layers

    raise AttributeError("Could not infer number of hidden layers from model object.")

def main():
    print("=== Starting Figure 1c RSA Reproduction ===")
    config = load_config()

    # 1. Load Model and Processor
    model_id = config['model']['huggingface_id']
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)

    num_layers = get_num_hidden_layers(model)

    # 2. Synthesize Trials for RSA
    # We generate a permutation matrix of shapes and colors to build the correlation variance.
    trials = []
    metadata_list = []
    
    colors = ["red", "blue"]
    shapes = ["circle", "square"]
    
    # Create combinations for the left and right objects
    all_permutations = list(itertools.product(colors, shapes[1:], colors, shapes[1:]))
    permutations = [p for p in all_permutations if not (p[0] == p[2] and p[1] == p[3])]
    
    # To save memory in local mode, we will slice the first 10 permutations. 
    # Increase this for a smoother correlation curve.
    for p in permutations[:]:
        left_color, left_shape, right_color, right_shape = p

        obj_indices, text_prompt = get_dynamic_token_indices(
            processor, left_color, left_shape, right_color
        )

        img = generate_spatial_binding_image(
            left_shape=left_shape, left_color=left_color,
            right_shape=right_shape, right_color=right_color
        )
        
        # Process the inputs into PyTorch tensors
        inputs = processor(text=text_prompt, images=img, return_tensors="pt")
        inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Map the proportional spatial coordinates from synthetic_generator.py (Assuming 336x336)
        left_coord = (int(336 * 0.3), 336 // 2)
        right_coord = (int(336 * 0.7), 336 // 2)
        
        # Object 0 = Left Object, Object 1 = Right Object
        trial_meta = [
            {"coord": left_coord, "color": left_color, "shape": left_shape},
            {"coord": right_coord, "color": right_color, "shape": right_shape}
        ]
        metadata_list.append(trial_meta)
        
        trials.append({
            'inputs': inputs,
            'object_token_indices': obj_indices
        })

    # 3. Execute Pipeline
    print(f"\nExecuting 3D RSA across {len(trials)} trials and {num_layers} layers...")
    run_rsa_pipeline(
        model=model,
        config=config,
        num_layers=num_layers,
        trials=trials,
        metadata_list=metadata_list,
        save_path="outputs/rsa_figure_1c.png"
    )

if __name__ == "__main__":
    main()
