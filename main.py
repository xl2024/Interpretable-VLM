import yaml
import torch
import einops
from nnsight import LanguageModel
from transformers import AutoProcessor

# Import custom modules
from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.mech_interp.tracer import extract_hidden_states
from src.utils.einops_utils import get_vision_sequence_indices, slice_spatial_states
from src.math_core.spatial_pca import compute_spatial_pca
from src.mech_interp.causal_patch import calculate_shift_vector, run_causal_swap

def load_config(config_path: str = "configs/local.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    print("=== Starting Mechanistic Interpretability Pipeline ===")
    
    # 1. Load Configuration
    config = load_config()
    model_id = config['model']['huggingface_id']
    prompt = config['dataset']['base_prompt']
    
    # 2. Generate the exact synthetic test from the paper
    print("\n[1/5] Generating synthetic image...")
    image = generate_custom_image(
        shapes=["circle", "square"],
        colors=["blue", "red"],
        coords=[(0,0), (0,1)]
    )
    
    # 3. Load Model and Processor
    print(f"\n[2/5] Loading processor and model ({model_id})...")
    model, processor = load_vlm(model_id = model_id, tier = 'local')
    
    # 4. Extract Hidden States (Standard Forward Pass)
    print("\n[3/5] Extracting hidden states...")
    hidden_states_dict = extract_hidden_states(model, processor, config, image, prompt)
    
    # will use the middle layer (e.g., Layer 12)
    layer_to_patch = config['mechanistic_interp']['trace_layers'][2] 
    target_layer_states = hidden_states_dict[layer_to_patch]
    
    # 5. Math Core: Slice and PCA
    print("\n[4/5] Executing Spatial PCA...")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    vision_token_id = processor.tokenizer.convert_tokens_to_ids(config['model']['vision_token_anchor'])
    start_idx, end_idx = get_vision_sequence_indices(inputs['input_ids'], vision_token_id)
    spatial_states = slice_spatial_states(target_layer_states, start_idx, end_idx)
    
    components = compute_spatial_pca(spatial_states, n_components=2)
    
    # Snap 576 back into a 24x24 spatial grid
    grid = einops.rearrange(spatial_states, '(h w) c -> h w c', h=24, w=24)

    # Slice columns
    left_cluster_coord = grid[:, :12, :].mean(dim=(0, 1))   # All rows, first 12 columns
    right_cluster_coord = grid[:, 12:, :].mean(dim=(0, 1))  # All rows, last 12 columns
    
    # 6. The Causal Swap
    print("\n[5/5] Executing Causal Intervention...")
    # The word "red" is the very last token in our prompt, so its sequence index is -1
    target_token_idx = -1 
    
    # The model wants to read the "red" token and look at the right object (the red square).
    # calculate the vector to push its spatial ID from the right object to the left object.
    shift_vector = calculate_shift_vector(
        source_coord=right_cluster_coord, 
        target_coord=left_cluster_coord, 
        components=components
    )
    
    # Force the model to output the incorrect shape by swapping its spatial attention
    generated_text = run_causal_swap(
        model=model,
        processor=processor,
        config=config,
        image=image,
        text_prompt=prompt,
        target_token_idx=target_token_idx,
        shift_vector=shift_vector,
        layer_to_patch=layer_to_patch
    )
    
    print("\n=== Experiment Complete ===")
    print(f"Target Token (Truth): '{config['dataset']['target_token'].strip()}'")
    print(f"Patched Token (Result): '{generated_text.strip()}'")

if __name__ == "__main__":
    main()