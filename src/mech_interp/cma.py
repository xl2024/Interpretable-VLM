import torch
import einops
import numpy as np
from typing import Dict, List, Tuple, Any

# Internal project imports
from src.utils.tools import _resolve_layer_path, get_layer_path_template
from src.mech_interp.tracer import gc_collect

def cma_headwise(
    model: Any,
    processor: Any,
    num_layers: int,
    num_heads: int,
    prompt_c1: str,
    prompt_c2: str,
    image_c1: Any,
    image_c2: Any,
    token_pos: Tuple[int, int],
    a1_id: int,
    a1_star_id: int
) -> List[List[Any]]:
    """
    Executes Causal Mediation Analysis (Activation Patching) across all attention heads.
    Patches activations from a modified context (c2) into the clean context (c1) following Eq. (1).
    """
    # 1. Resolve architecture dimensions dynamically
    layer_template = get_layer_path_template(model)
    
    inputs_c1 = processor(text=prompt_c1, images=image_c1, return_tensors="pt").to(model.device)
    inputs_c2 = processor(text=prompt_c2, images=image_c2, return_tensors="pt").to(model.device)

    # 3. Cache c2 Counterfactual States
    c2_head_cache = {}
    with torch.no_grad():
        with model.trace() as tracer:
            with tracer.invoke(**inputs_c2):
                for l in range(num_layers):
                    layer_module = _resolve_layer_path(model, layer_template.format(l))
                    # Safely intercept full 3D tensor: [batch, seq_len, hidden_dim]
                    attn_out = layer_module.self_attn.o_proj.input[0]
                    c2_head_cache[l] = einops.rearrange(attn_out, 's (h d) -> s h d', h=num_heads).save()
        
        gc_collect()

    # 4. Trace Baseline Clean (c1) Execution
    with torch.no_grad():
        with model.trace() as tracer:
            with tracer.invoke(**inputs_c1):
                # Safely slice 3D logit tensor preserving batch dim
                clean_logits = model.lm_head.output[:, token_pos[0]:token_pos[1], :].save()

        gc_collect()

    # Calculate Baseline Clean Term: M(c1)[a1*] - M(c1)[a1]
    base_a1_logit = clean_logits[0, :, a1_id].mean().item()
    base_a1_star_logit = clean_logits[0, :, a1_star_id].mean().item()
    base_term = base_a1_star_logit - base_a1_logit
    print(f"Baseline Clean Term: {base_term:.4f}")

    # 5. Activation Patching Intervention Loop (c2 -> c1)
    mediation_scores = np.zeros((num_layers, num_heads))
    print(f"Executing intervention sweep across {num_layers} layers and {num_heads} heads per layer...")
    
    for l in range(num_layers):
        print(f"Processing layer {l+1}/{num_layers}...")
        for h in range(num_heads):
            patched_logits = None
            with torch.no_grad():
                with model.trace() as tracer:
                    with tracer.invoke(**inputs_c1):
                        target_layer = _resolve_layer_path(model, layer_template.format(l))
                        
                        # Intercept input to o_proj
                        hs_input = target_layer.self_attn.o_proj.input[0]
                        hs_heads = einops.rearrange(hs_input, 's (h d) -> s h d', h=num_heads)
                        
                        # True CMA Patch: Inject cached c2 head state into c1 stream
                        hs_heads[token_pos[0]:token_pos[1], h, :] = c2_head_cache[l][token_pos[0]:token_pos[1], h, :].to(model.device)
                        
                        # Repack dimensions safely
                        hs_input[:] = einops.rearrange(hs_heads, 's h d -> s (h d)')
                        
                        # Capture patched output logits safely
                        patched_logits = model.lm_head.output[:, token_pos[0]:token_pos[1], :].save()

                gc_collect()
            
            # Calculate Patched Term: M(c1*)[a1*] - M(c1*)[a1]
            p_a1_logit = patched_logits[0, :, a1_id].mean().item()
            p_a1_star_logit = patched_logits[0, :, a1_star_id].mean().item()
            patched_term = p_a1_star_logit - p_a1_logit
            
            # Equation (1): s = Patched Term - Baseline Term
            s = patched_term - base_term
            mediation_scores[l, h] = s

    return mediation_scores

def cma_head_patching(
    model: Any,
    processor: Any,
    num_layers: int,
    num_heads: int,
    prompt_c1: str,
    prompt_c2: str,
    image_c1: Any,
    image_c2: Any,
    top_k_heads: List[Tuple[int, int]]
) -> str:
    """
    Executes Causal Mediation Analysis (Activation Patching) across top k ID selection heads.
    """
    # 1. Resolve architecture dimensions dynamically
    layer_template = get_layer_path_template(model)
    
    inputs_c1 = processor(text=prompt_c1, images=image_c1, return_tensors="pt").to(model.device)
    inputs_c2 = processor(text=prompt_c2, images=image_c2, return_tensors="pt").to(model.device)

    # 3. Cache c2 Counterfactual States
    heads_by_layer = {}
    for l, h in top_k_heads:
        heads_by_layer.setdefault(l, []).append(h)
    c2_head_cache = {}
    with torch.no_grad():
        with model.trace() as tracer:
            with tracer.invoke(**inputs_c2):
                for l, heads_in_this_layer in heads_by_layer.items():
                    layer_module = _resolve_layer_path(model, layer_template.format(l))
                    print("layer_module:",l,layer_module)
                    # Safely intercept full 3D tensor: [batch, seq_len, hidden_dim]
                    attn_out = layer_module.self_attn.o_proj.input[0]
                    hs_heads = einops.rearrange(attn_out, 's (h d) -> s h d', h=num_heads)
                    for h in heads_in_this_layer:
                        c2_head_cache[l,h] = hs_heads[-1, h, :].save()

        gc_collect()

    with torch.no_grad():
        with model.generate(max_new_tokens=2, pad_token_id=processor.tokenizer.eos_token_id) as tracer:
            with tracer.invoke(**inputs_c1):
                for l, h in top_k_heads:
                    target_layer = _resolve_layer_path(model, layer_template.format(l))
                    
                    # Intercept input to o_proj
                    hs_input = target_layer.self_attn.o_proj.input[0]
                    hs_heads = einops.rearrange(hs_input, 's (h d) -> s h d', h=num_heads)
                    
                    # True CMA Patch: Inject cached c2 head state into c1 stream
                    hs_heads[-1, h, :] = c2_head_cache[l,h].to(model.device)
                    
                    # Repack dimensions safely
                    hs_input[:] = einops.rearrange(hs_heads, 's h d -> s (h d)')
        
                patched_output = tracer.output.save()

        gc_collect()

    predicted_text = processor.decode(patched_output[0], skip_special_tokens=True)
    predicted_word = predicted_text[len(prompt_c1):]
    print(f"The patched model said (text): {predicted_text} length {len(predicted_text)}")
    print(f"The patched model said (word): {predicted_word} length {len(predicted_word)}")


    input_length = inputs_c1["input_ids"].shape[1]
    new_tokens = patched_output[0][input_length:]

    predicted_word_2 = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"predicted_word_2: {predicted_word_2} length {len(predicted_word_2)}")

    return predicted_word