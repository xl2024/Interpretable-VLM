import torch
from nnsight import LanguageModel
from typing import Dict, Any, List, Tuple

from src.utils.tools import _resolve_layer_path, _build_object_ids, _resolve_text_model_dims, get_layer_path_template, gc_collect


def rsa_tracer(
    model: LanguageModel,
    config: Dict[str, Any],
    num_layers: int,
    trials: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    extracts hidden states via nnsight.
    """
    # 1. Initialize storage for both token types
    hidden_states_by_trial = []
    trial_object_ids, token_object_ids = _build_object_ids(trials)

    hidden_size, num_heads = _resolve_text_model_dims(model)
    head_dim = hidden_size // num_heads
    print(f'hidden_size: {hidden_size}, num_heads: {num_heads}, head_dim: {head_dim}')

    print(f"Extracting hidden states across {len(trials)} trials...")
    with torch.no_grad():
        for trial_idx, trial_data in enumerate(trials):
            inputs = trial_data['inputs']
            # List of indices where the model reads the objects (e.g., [14, 22])
            obj_indices = token_object_ids[trial_idx] 
            object_ids = trial_object_ids[trial_idx]
            
            prompt_states = {}
            
            with model.trace() as tracer:
                with tracer.invoke(**inputs):
                    for layer_idx in range(num_layers):
                        layer_path = get_layer_path_template(model).format(layer_idx)
                        layer_module = _resolve_layer_path(model, layer_path)
                        
                        # The hidden state tensor for this layer
                        hs = layer_module.post_attention_layernorm.output[0]

                        prompt_states[layer_idx] = {}

                        for i in range(len(obj_indices)):
                            object_id, token_index = obj_indices[i]
                            prompt_states[layer_idx][i] = hs[token_index, :].save()
                    
            # Append the resolved dictionaries to main lists
            hidden_states_by_trial.append(prompt_states)

            gc_collect()
        
    print("Extraction complete!")
    return hidden_states_by_trial
    
def rsa_tracer_2(
    model: LanguageModel,
    config: Dict[str, Any],
    num_layers: int,
    trials: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    extracts hidden states via nnsight.
    """
    # 1. Initialize storage for both token types
    hidden_states_by_trial = []
    hidden_size, num_heads = _resolve_text_model_dims(model)
    head_dim = hidden_size // num_heads
    print(f'hidden_size: {hidden_size}, num_heads: {num_heads}, head_dim: {head_dim}')

    print(f"Extracting hidden states across {len(trials)} trials...")
    with torch.no_grad():
        for trial_idx, trial_data in enumerate(trials):
            inputs = trial_data['inputs']
            prompt_states = {}
            with model.trace() as tracer:
                with tracer.invoke(**inputs):
                    for layer_idx in range(num_layers):
                        layer_path = get_layer_path_template(model).format(layer_idx)
                        layer_module = _resolve_layer_path(model, layer_path)
                        
                        # The hidden state tensor for this layer
                        hs = layer_module.post_attention_layernorm.output[0]
                        prompt_states[layer_idx] = hs[-1, :].save()
                    
            # Append the resolved dictionaries to main lists
            hidden_states_by_trial.append(prompt_states)

            gc_collect()
        
    print("Extraction complete!")
    return hidden_states_by_trial
    