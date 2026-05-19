import torch
import einops
import numpy as np
from typing import Dict, List, Tuple, Any
import copy

from src.utils.tools import _resolve_layer_path, get_layer_path_template, get_text_prompt, predict
from src.mech_interp.tracer import gc_collect
from src.data.synthetic_generator import generate_custom_image


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
    a1_star_id: int,
    _mediation_scores: List[List[Any]] = None
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
    if _mediation_scores is None:
        mediation_scores = np.zeros((num_layers, num_heads))
    else:
        mediation_scores = _mediation_scores
    print(f"Executing intervention sweep across {num_layers} layers and {num_heads} heads per layer...")
    
    print(f"skipping the first {int(num_layers/2)} layers for speeding up...")
    for l in range(num_layers):
        if l < num_layers/2:
            continue

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
            mediation_scores[l, h] += s

    return mediation_scores

def run_cma_for_ID_retrieval(
    model: Any,
    processor: Any,
    num_layers: int,
    num_heads: int,
    shapes: List[str],
    colors: List[str],
    _mediation_scores: List[List[Any]] = None
) -> List[List[Any]]:
    """
    Executes Causal Mediation Analysis (Activation Patching) across all attention heads.
    Patches activations from a modified context (c2) into the clean context (c1) following Eq. (1).
    """
    # ID Retrieval Heads
    print("cma for ID Retrieval Heads...")

    num_objs = len(shapes)
    prompt = "In this image there is a"
    for i in range(num_objs-1):
        prompt += f" {colors[i]} {shapes[i]}, a"
    prompt = prompt[:-3] + " and a"
    
    num_cols, num_rows = 2, int((num_objs+1)/2)
    coords_c1 = []
    coords_c2 = []
    for i in range(num_rows):
        for j in range(2):
            coords_c1.append((i,j))
            coords_c2.append((i,j))
    coords_c2[-2:] = coords_c2[-1:-3:-1]
    
    image_c1 = generate_custom_image(
        cols=num_cols,
        rows=num_rows,
        shapes=shapes,
        colors=colors,
        coords=coords_c1
    )
    image_c2 = generate_custom_image(
        cols=num_cols,
        rows=num_rows,
        shapes=shapes,
        colors=colors,
        coords=coords_c2
    )

    text_prompt_c1 = get_text_prompt(model, prompt, image_c1, processor)
    text_prompt_c2 = get_text_prompt(model, prompt, image_c2, processor)

    print(f"Prediction: {predict(model, processor, image_c1, text_prompt_c1)} (target: {colors[-1]})")
    print(f"Prediction: {predict(model, processor, image_c2, text_prompt_c2)} (target: {colors[-1]})")

    token_inputs = processor(text=text_prompt_c1, images=image_c1, return_tensors="pt")
    input_ids = token_inputs["input_ids"][0].tolist()
    for index, token_id in enumerate(input_ids):
        token_str = processor.tokenizer.decode(token_id).strip().lower()
        if colors[-2] in token_str:
            token_pos_1 = index
        elif shapes[-2] in token_str:
            token_pos_2 = index
            break
    token_pos = (token_pos_1, token_pos_2+1)

    a1_tokens = processor.tokenizer.encode(colors[-1], add_special_tokens=False)
    a1_star_tokens = processor.tokenizer.encode(colors[-2], add_special_tokens=False)
    a1_id = a1_tokens[-1]
    a1_star_id = a1_star_tokens[-1]

    print(f"Target Token ID (a1): {a1_id} -> '{processor.tokenizer.decode([a1_id])}'")
    print(f"Contrast Token ID (a1*): {a1_star_id} -> '{processor.tokenizer.decode([a1_star_id])}'")

    mediation_scores_1 = cma_headwise(
        model=model,
        processor=processor,
        num_layers=num_layers,
        num_heads=num_heads,
        prompt_c1=text_prompt_c1,
        prompt_c2=text_prompt_c2,
        image_c1=image_c1,
        image_c2=image_c2,
        token_pos=token_pos,
        a1_id=a1_id,
        a1_star_id=a1_star_id,
        _mediation_scores=_mediation_scores
    )

    return mediation_scores_1

def run_cma_for_ID_selection(
    model: Any,
    processor: Any,
    num_layers: int,
    num_heads: int,
    shapes: List[str],
    colors: List[str],
    _mediation_scores: List[List[Any]] = None
) -> List[List[Any]]:
    """
    Executes Causal Mediation Analysis (Activation Patching) across all attention heads.
    Patches activations from a modified context (c2) into the clean context (c1) following Eq. (1).
    """
    # ID Selection Heads
    print("cma for ID Selection Heads...")
    
    num_objs = len(shapes)
    prompt = "In this image there is a"
    for i in range(num_objs-1):
        prompt += f" {colors[i]} {shapes[i]}, a"
    prompt = prompt[:-3] + " and a"
    
    num_cols, num_rows = 2, int((num_objs+1)/2)
    coords_c1 = []
    coords_c2 = []
    for i in range(num_rows):
        for j in range(2):
            coords_c1.append((i,j))
            coords_c2.append((i,j))
    coords_c2[-2:] = coords_c2[-1:-3:-1]

    image_c1 = generate_custom_image(
        cols=num_cols,
        rows=num_rows,
        shapes=shapes,
        colors=colors,
        coords=coords_c1
    )
    image_c2 = generate_custom_image(
        cols=num_cols,
        rows=num_rows,
        shapes=shapes,
        colors=colors,
        coords=coords_c2
    )

    text_prompt_c1 = get_text_prompt(model, prompt, image_c1, processor)
    text_prompt_c2 = get_text_prompt(model, prompt, image_c2, processor)

    print(f"Prediction: {predict(model, processor, image_c1, text_prompt_c1)} (target: {colors[-1]})")
    print(f"Prediction: {predict(model, processor, image_c2, text_prompt_c2)} (target: {colors[-1]})")

    a1_tokens = processor.tokenizer.encode(colors[-1], add_special_tokens=False)
    a1_star_tokens = processor.tokenizer.encode(colors[-2], add_special_tokens=False)
    a1_id = a1_tokens[-1]
    a1_star_id = a1_star_tokens[-1]

    print(f"Target Token ID (a1): {a1_id} -> '{processor.tokenizer.decode([a1_id])}'")
    print(f"Contrast Token ID (a1*): {a1_star_id} -> '{processor.tokenizer.decode([a1_star_id])}'")

    # ID Selection Heads
    print("cma for ID Selection Heads...")
    token_inputs = processor(text=text_prompt_c1, images=image_c1, return_tensors="pt")
    input_ids = token_inputs["input_ids"][0].tolist()
    token_pos = (len(input_ids)-1, len(input_ids))

    mediation_scores_2 = cma_headwise(
        model=model,
        processor=processor,
        num_layers=num_layers,
        num_heads=num_heads,
        prompt_c1=text_prompt_c1,
        prompt_c2=text_prompt_c2,
        image_c1=image_c1,
        image_c2=image_c2,
        token_pos=token_pos,
        a1_id=a1_id,
        a1_star_id=a1_star_id,
        _mediation_scores=_mediation_scores
    )

    return mediation_scores_2

def run_cma_for_feature_retrieval(
    model: Any,
    processor: Any,
    num_layers: int,
    num_heads: int,
    shapes: List[str],
    colors: List[str],
    new_color: str = 'green',
    _mediation_scores: List[List[Any]] = None
) -> List[List[Any]]:
    """
    Executes Causal Mediation Analysis (Activation Patching) across all attention heads.
    Patches activations from a modified context (c2) into the clean context (c1) following Eq. (1).
    """
    # Feature Retrieval Heads
    print("cma for Feature Retrieval Heads...")

    num_objs = len(shapes)
    prompt = "In this image there is a"
    for i in range(num_objs-1):
        prompt += f" {colors[i]} {shapes[i]}, a"
    prompt = prompt[:-3] + " and a"
    colors_c2 = copy.deepcopy(colors)
    colors_c2[-1] = new_color
    num_cols, num_rows = 2, int((num_objs+1)/2)
    coords = []
    for i in range(num_rows):
        for j in range(2):
            coords.append((i,j))

    image_c1 = generate_custom_image(
        cols=num_cols,
        rows=num_rows,
        shapes=shapes,
        colors=colors,
        coords=coords
    )
    image_c2 = generate_custom_image(
        cols=num_cols,
        rows=num_rows,
        shapes=shapes,
        colors=colors_c2,
        coords=coords
    )

    text_prompt_c1 = get_text_prompt(model, prompt, image_c1, processor)
    text_prompt_c2 = get_text_prompt(model, prompt, image_c2, processor)

    print(f"Prediction: {predict(model, processor, image_c1, text_prompt_c1)} (target: {colors[-1]})")
    print(f"Prediction: {predict(model, processor, image_c2, text_prompt_c2)} (target: {new_color})")

    a1_tokens = processor.tokenizer.encode(colors[-1], add_special_tokens=False)
    a1_id = a1_tokens[-1]
    a1_star_tokens = processor.tokenizer.encode(new_color, add_special_tokens=False)
    a1_star_id = a1_star_tokens[-1]

    print(f"Target Token ID (a1): {a1_id} -> '{processor.tokenizer.decode([a1_id])}'")
    print(f"Contrast Token ID (a1*): {a1_star_id} -> '{processor.tokenizer.decode([a1_star_id])}'")

    token_inputs = processor(text=text_prompt_c1, images=image_c1, return_tensors="pt")
    input_ids = token_inputs["input_ids"][0].tolist()
    token_pos = (len(input_ids)-1, len(input_ids))

    mediation_scores_3 = cma_headwise(
        model=model,
        processor=processor,
        num_layers=num_layers,
        num_heads=num_heads,
        prompt_c1=text_prompt_c1,
        prompt_c2=text_prompt_c2,
        image_c1=image_c1,
        image_c2=image_c2,
        token_pos=token_pos,
        a1_id=a1_id,
        a1_star_id=a1_star_id,
        _mediation_scores=_mediation_scores
    )

    return mediation_scores_3

def multi_runs_for_ID_selection(
    model: Any,
    processor: Any,
    num_layers: int,
    num_heads: int,
    runs: int
) -> List[List[Any]]:
    shapeset = ["circle", "square", "triangle", "cross", "star", "heart"]
    colorset = ['red', 'blue', 'green', 'yellow', 'purple']
    mediation_scores_2 = None
    for i in range(runs):
        shapes = np.random.choice(shapeset, size=2, replace=False)
        colors = np.random.choice(colorset, size=2, replace=False)
        print(f"scores_for_ID_selection runs {i+1}/{runs}")
        mediation_scores_2 = run_cma_for_ID_selection(
                                model=model,
                                processor=processor,
                                num_layers=num_layers,
                                num_heads=num_heads,
                                shapes=shapes,
                                colors=colors,
                                _mediation_scores = mediation_scores_2
                            )
        
    return mediation_scores_2

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
                for l, heads_in_this_layer in sorted(heads_by_layer.items()):
                    layer_module = _resolve_layer_path(model, layer_template.format(l))
                    # Safely intercept full 3D tensor: [batch, seq_len, hidden_dim]
                    attn_out = layer_module.self_attn.o_proj.input[0]
                    hs_heads = einops.rearrange(attn_out, 's (h d) -> s h d', h=num_heads)
                    for h in sorted(heads_in_this_layer):
                        c2_head_cache[l,h] = hs_heads[-1, h, :].save()

        gc_collect()

    
    patched_logits = None
    with torch.no_grad():
        with model.trace() as tracer:
            with tracer.invoke(**inputs_c1):
                for l, heads_in_this_layer in sorted(heads_by_layer.items()):
                    target_layer = _resolve_layer_path(model, layer_template.format(l))
                    
                    # Intercept input to o_proj
                    hs_input = target_layer.self_attn.o_proj.input[0]
                    hs_heads = einops.rearrange(hs_input, 's (h d) -> s h d', h=num_heads)
                    
                    for h in sorted(heads_in_this_layer):
                        # True CMA Patch: Inject cached c2 head state into c1 stream
                        # hs_heads[-1, h, :] = c2_head_cache[l, h].to(model.device)
                        c2_state = c2_head_cache[l, h].to(model.device)
                        c1_state = hs_heads[-1, h, :]
                        concept_vector = c2_state - c1_state
                        hs_heads[-1, h, :] = c1_state + (3.0 * concept_vector)

                    # Repack dimensions safely
                    hs_input[:] = einops.rearrange(hs_heads, 's h d -> s (h d)')

                # Capture patched output logits safely
                patched_logits = model.lm_head.output[:, -1, :].save()

        gc_collect()

    # 1. Grab the raw logits for the final token
    final_logits = patched_logits[0, :]

    # 2. Instantly find the index (Token ID) of the highest number
    predicted_token_id = final_logits.argmax(dim=-1).item()    # dim=-1 to work in vocabulary dim

    # 3. Decode that ID straight back into an English word
    predicted_word = processor.tokenizer.decode([predicted_token_id])

    print(f"The model predicted: '{predicted_word}'")

    # with torch.no_grad():
    #     with model.generate(max_new_tokens=2, pad_token_id=processor.tokenizer.eos_token_id) as tracer:
    #         with tracer.invoke(**inputs_c1):
    #             for l, h in sorted(top_k_heads):
    #                 target_layer = _resolve_layer_path(model, layer_template.format(l))
                    
    #                 # Intercept input to o_proj
    #                 hs_input = target_layer.self_attn.o_proj.input[0]
    #                 hs_heads = einops.rearrange(hs_input, 's (h d) -> s h d', h=num_heads)
                    
    #                 # True CMA Patch: Inject cached c2 head state into c1 stream
    #                 hs_heads[-1, h, :] = c2_head_cache[l,h].to(model.device)
                    
    #                 # Repack dimensions safely
    #                 hs_input[:] = einops.rearrange(hs_heads, 's h d -> s (h d)')
        
    #             patched_output = tracer.result.save()

    #     gc_collect()

    # predicted_text = processor.decode(patched_output[0], skip_special_tokens=True)
    # print(f"The patched model said: {predicted_text}")

    # input_length = inputs_c1["input_ids"].shape[1]
    # new_tokens = patched_output[0][input_length:]
    # predicted_word = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    # print(f"predicted_word: {predicted_word}")

    return predicted_word