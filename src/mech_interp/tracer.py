import torch
from nnsight import LanguageModel
from typing import Dict, Any, List, Tuple
import gc

def _build_object_ids(trials: List[Dict[str, Any]] = None) -> List[List[int]]:
    """
    Build stable object ids from (color, shape) pairs.
    Objects sharing the same (color, shape) get the same object_id.
    """
    object_id_by_feature: Dict[Tuple[str, str], int] = {}
    trial_object_ids: List[List[int]] = []

    for t in trials:
        trial = t['trial']
        ids_for_trial: List[int] = []
        for obj in trial:
            key = (obj['color'], obj['shape'])
            if key not in object_id_by_feature:
                object_id_by_feature[key] = len(object_id_by_feature)
            ids_for_trial.append(object_id_by_feature[key])
        trial_object_ids.append(ids_for_trial)

    token_object_ids: List[List[int]] = []
    if trials is not None:
        for trial in trials:
            ids_for_token: List[int] = []
            for i in range(len(trial['trial'])):
                key = (trial['trial'][i]['color'], trial['trial'][i]['shape'])
                ids_for_token.append((object_id_by_feature[key], trial['trial'][i]['index']))
            token_object_ids.append(ids_for_token)
    print('*** object_id_by_feature *** \n', object_id_by_feature)
    return trial_object_ids, token_object_ids


def _resolve_trial_object_index(object_token_indices: List[int], object_position: int) -> int:
    """
    Resolve token index for a given object position within a trial.
    Fallback to the last available index when fewer indices are provided.
    """
    if len(object_token_indices) == 0:
        raise ValueError("trial_data['trial'] cannot be empty.")

    if object_position < len(object_token_indices):
        return object_token_indices[object_position]

    return object_token_indices[-1]

def _resolve_token_object_index(object_token_indices: List[Dict[str, Any]], object_position: int) -> int:
    """
    Resolve token index for a given object position within a trial.
    Fallback to the last available index when fewer indices are provided.
    """
    if len(object_token_indices) == 0:
        raise ValueError("trial_data['trial'] cannot be empty.")

    if object_position < len(object_token_indices):
        return object_token_indices[object_position]['index']

    return object_token_indices[-1]['index']

def _resolve_layer_path(model: LanguageModel, path_string: str):
    """
    Safely traverses the nnsight model architecture to return the exact 
    PyTorch module based on the config's string path.
    
    Example: 
        path_string = "model.language_model.model.layers[8]"
    """
    # We split by '.' and handle list indices like 'layers[8]'
    current_module = model
    parts = path_string.split('.')
    
    for part in parts:
        if '[' in part and ']' in part:
            attr_name, index_part = part.split('[')
            index = int(index_part.replace(']', ''))
            current_module = getattr(current_module, attr_name)[index]
        else:
            current_module = getattr(current_module, part)
            
    return current_module

def _resolve_text_model_dims(model: Any) -> Tuple[int, int]:
    """
    Resolve (hidden_size, num_attention_heads) across wrapped/unwrapped VLM models.
    Works when `model.config` is missing/None (common with wrappers).
    """
    candidate_configs: List[Any] = []

    # Direct config on the visible object
    candidate_configs.append(getattr(model, "config", None))

    # Common nnsight/HF wrapper patterns
    local_model = getattr(model, "local_model", None)
    if local_model is not None:
        candidate_configs.append(getattr(local_model, "config", None))

    nested_model = getattr(model, "model", None)
    if nested_model is not None:
        candidate_configs.append(getattr(nested_model, "config", None))
        language_model = getattr(nested_model, "language_model", None)
        if language_model is not None:
            candidate_configs.append(getattr(language_model, "config", None))

    # Some multimodal models expose text dims under text_config
    expanded_configs: List[Any] = []
    for cfg in candidate_configs:
        if cfg is None:
            continue
        expanded_configs.append(cfg)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            expanded_configs.append(text_cfg)

    for cfg in expanded_configs:
        hidden_size = getattr(cfg, "hidden_size", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        if isinstance(hidden_size, int) and isinstance(num_heads, int) and num_heads > 0:
            return hidden_size, num_heads

    raise AttributeError(
        "Could not resolve hidden_size/num_attention_heads from model object. "
        "Expected fields on config or text_config."
    )

def extract_hidden_states(
    model: LanguageModel, 
    processor: Any, 
    config: Dict[str, Any], 
    image: Any, 
    text_prompt: str
) -> Dict[int, torch.Tensor]:
    """
    Runs a clean forward pass to extract and save intermediate hidden states
    for the layers specified in the configuration.
    
    Args:
        model: The nnsight-wrapped Vision-Language Model.
        processor: The Hugging Face processor for text/image tokenization.
        config: The parsed YAML configuration dictionary.
        image: A PIL Image or dummy noise tensor.
        text_prompt: The prompt string (e.g., "<image>\nIn this image...")
        
    Returns:
        Dict mapping layer indices to their extracted hidden state tensors (on CPU).
    """
    print("Preparing inputs for tracing...")
    
    # 1. Process inputs into exactly what the specific VLM expects
    inputs = processor(
        text=text_prompt, 
        images=image, 
        return_tensors="pt"   # lists -> PyTorch Tensors
    )  # .to(model.device) # Move raw inputs to the active hardware device

    inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    trace_layers: List[int] = config['mechanistic_interp']['trace_layers']
    layer_template: str = config['model']['layer_path_template']
    
    extracted_states = {}
    
    # 2. Enter the nnsight Intervention Context
    print(f"Tracing forward pass and intercepting layers: {trace_layers}...")
    with model.trace() as tracer:
        with tracer.invoke(**inputs):        
            for layer_idx in trace_layers:
                # Build the exact string path (e.g., "model.language_model.model.layers[14]")
                layer_path = layer_template.format(layer_idx)
                
                # Grab the specific nnsight layer module
                layer_module = _resolve_layer_path(model, layer_path)
                
                # 3. The Extraction & CPU Offload
                # Transformer layers usually return a tuple: (hidden_states, attention_weights, etc.)
                # We strictly want index [0]. We save it, and immediately push to CPU to prevent OOM.
                extracted_states[layer_idx] = layer_module.output[0].save().cpu()
            
    print("Trace complete. Hidden states successfully offloaded to CPU.")
    
    # 4. nnsight unwraps the saved `.value` when the `with` block exits
    final_states = {
        layer_idx: proxy_tensor 
        for layer_idx, proxy_tensor in extracted_states.items()
    }
    
    return final_states

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
                        layer_path = config['model']['layer_path_template'].format(layer_idx)
                        layer_module = _resolve_layer_path(model, layer_path)
                        
                        # The hidden state tensor for this layer
                        hs = layer_module.post_attention_layernorm.output[0]

                        prompt_states[layer_idx] = {}

                        for i in range(len(obj_indices)):
                            object_id, token_index = obj_indices[i]
                            prompt_states[layer_idx][i] = hs[token_index, :].save()
                    
            # Append the resolved dictionaries to main lists
            hidden_states_by_trial.append(prompt_states)

            # Force clear the memory before the next trial begins
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    print("Extraction complete!")
    return hidden_states_by_trial
    