import torch
from nnsight import LanguageModel
from typing import Dict, Any
from src.mech_interp.tracer import _resolve_layer_path

def calculate_shift_vector(
    source_coord: torch.Tensor, 
    target_coord: torch.Tensor, 
    components: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the 4096-dimensional vector required to move a token from the 
    source spatial location to the target spatial location.
    
    Args:
        source_coord: The [X, Y] PCA coordinate of the original object (e.g., Red token).
        target_coord: The [X, Y] PCA coordinate of the destination (e.g., Blue token).
        components: The [2, hidden_dim] PCA axes extracted previously.
        
    Returns:
        A 1D tensor of shape [hidden_dim] representing the exact geometric shift.
    """
    delta = target_coord - source_coord
    shift_vector = (delta[0] * components[0]) + (delta[1] * components[1])
    shift_vector = shift_vector.to('cuda')

    return shift_vector


def run_causal_swap(
    model: LanguageModel, 
    processor: Any, 
    config: Dict[str, Any], 
    image: Any, 
    text_prompt: str,
    target_token_idx: int,
    shift_vector: torch.Tensor,
    layer_to_patch: int
) -> str:
    """
    Executes the forward pass while mathematically rewriting the spatial identity
    of the target text token in real-time.
    
    Args:
        model: nnsight wrapped VLM.
        processor: Hugging Face processor.
        config: Parsed YAML config.
        image: The raw PIL image or tensor.
        text_prompt: The incomplete caption.
        target_token_idx: The sequence index of the word "red" in the prompt.
        shift_vector: The calculated 4096-D spatial shift.
        layer_to_patch: Which transformer layer to intercept.
        
    Returns:
        The generated string (the predicted next token).
    """
    print(f"Initiating causal swap at layer {layer_to_patch} on token index {target_token_idx}...")
    
    inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(model.device)
    layer_template: str = config['model']['layer_path_template']
    layer_path = layer_template.format(layer_to_patch)
    
    layer_module = _resolve_layer_path(model, layer_path)

    # Enter the intervention context
    with model.generate(max_new_tokens=2, pad_token_id=processor.tokenizer.eos_token_id) as tracer:
        # input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], pixel_values=inputs['pixel_values']
        with tracer.invoke(**inputs):
            # Intercept the forward pass exactly at the specified layer
            hidden_states = layer_module.output[0]
            
            # Surgically add the shift vector to strictly the target token's hidden state
            # Shape: [batch, sequence_length, hidden_dim]
            hidden_states[target_token_idx, :] = hidden_states[target_token_idx, :] + shift_vector
        
            output = tracer.result.save()
        
    # Extract the decoded text from the output of [batch_size, sequence_length]
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    # print(f"Causal swap complete. Model predicted: '{generated_text.strip()}'")
    
    return generated_text