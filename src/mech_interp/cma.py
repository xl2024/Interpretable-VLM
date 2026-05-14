import torch
import einops
import numpy as np
from typing import Dict, List, Tuple, Any

# Internal project imports
from src.utils.tools import _resolve_layer_path
from src.mech_interp.tracer import gc_collect

def cma_headwise(
    model: Any,
    processor: Any,
    config: Dict[str, Any],
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
    layer_template = config['model']['layer_path_template']
    
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
                    c2_head_cache[l] = einops.rearrange(attn_out, 's (h d) -> s h d', h=num_heads).save().cpu()
        
        gc_collect()

    def _log_trace_context(tag: str, exc: Exception | None = None) -> None:
        print(f"[CMA] {tag} trace failed on device={model.device}")
        if exc is not None:
            print(f"[CMA] Exception: {type(exc).__name__}: {exc}")
        input_shapes = {
            k: tuple(v.shape)
            for k, v in inputs_c1.items()
            if torch.is_tensor(v)
        }
        print(f"[CMA] inputs_c1 tensor shapes: {input_shapes}")
        print(f"[CMA] token_pos={token_pos}, a1_id={a1_id}, a1_star_id={a1_star_id}")
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                print(f"[CMA] cuda mem allocated={allocated} reserved={reserved}")
            except Exception as mem_exc:
                print(f"[CMA] cuda mem query failed: {mem_exc}")

    # 4. Trace Baseline Clean (c1) Execution
    clean_logits = None
    try:
        with torch.no_grad():
            with model.trace() as tracer:
                with tracer.invoke(**inputs_c1):
                    # Safely slice 3D logit tensor preserving batch dim
                    clean_logits = model.lm_head.output[:, token_pos[0]:token_pos[1], :].save().cpu()

            gc_collect()
    except Exception as exc:
        _log_trace_context("baseline clean", exc)
        raise

    if clean_logits is None:
        _log_trace_context("baseline clean")
        return np.full((num_layers, num_heads), np.nan)

    # Calculate Baseline Clean Term: M(c1)[a1*] - M(c1)[a1]
    base_a1_logit = clean_logits[0, :, a1_id].mean().item()
    base_a1_star_logit = clean_logits[0, :, a1_star_id].mean().item()
    base_term = base_a1_star_logit - base_a1_logit
    print(f"Baseline Clean Term: {base_term:.4f}")

    # 5. Activation Patching Intervention Loop (c2 -> c1)
    mediation_scores = np.zeros((num_layers, num_heads))
    print(f"Executing intervention sweep across {num_layers} layers and {num_heads} heads per layer...")
    
    for l in range(num_layers):
        for h in range(num_heads):
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
                        patched_logits = model.lm_head.output[:, token_pos[0]:token_pos[1], :].save().cpu()

                gc_collect()
            
            # Calculate Patched Term: M(c1*)[a1*] - M(c1*)[a1]
            p_a1_logit = patched_logits[0, :, a1_id].mean().item()
            p_a1_star_logit = patched_logits[0, :, a1_star_id].mean().item()
            patched_term = p_a1_star_logit - p_a1_logit
            
            # Equation (1): s = Patched Term - Baseline Term
            s = patched_term - base_term
            mediation_scores[l, h] = s

    return mediation_scores