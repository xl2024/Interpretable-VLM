import torch
from nnsight import LanguageModel
from typing import Dict, Any, List

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
    ).to(model.device) # Move raw inputs to the active hardware device
    
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