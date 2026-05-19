import numpy as np
from typing import Dict, List, Tuple, Any


from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import load_config, _resolve_text_model_dims, get_text_prompt, predict
from src.plots.rsa_1c import get_num_hidden_layers
from src.mech_interp.cma import cma_headwise, cma_head_patching




absolute_score = 0
relative_score = 0

def scores_for_ID_selection(
    model: Any,
    processor: Any,
    num_layers: int,
    num_heads: int,
    runs: int
) -> List[List[Any]]:
    shapeset = ["circle", "square", "triangle", "cross", "star", "heart"]
    colorset = ['red', 'blue', 'green', 'yellow', 'purple']
    mediation_scores_2 = np.zeros((num_layers, num_heads))
    for i in range(runs):
        if i > 0: continue
        shapes = np.random.choice(shapeset, size=2, replace=False)
        colors = np.random.choice(colorset, size=2, replace=False)
        print(f"scores_for_ID_selection runs {i+1}/{runs}")
        mediation_scores_2 = run_mediation_analysis_for_ID_selection(
                                model=model,
                                processor=processor,
                                num_layers=num_layers,
                                num_heads=num_heads,
                                shapes=shapes,
                                colors=colors,
                                _mediation_scores = mediation_scores_2
                            )
        
    return mediation_scores_2

def run_mediation_analysis_for_ID_selection(
    model: Any,
    processor: Any,
    num_layers: int,
    num_heads: int,
    shapes: List[str],
    colors: List[str],
    _mediation_scores: List[List[Any]]
) -> List[List[Any]]:
    """
    Executes Causal Mediation Analysis (Activation Patching) across all attention heads.
    Patches activations from a modified context (c2) into the clean context (c1) following Eq. (1).
    """
    print("Preparing Causal Mediation Analysis...")

    # ID Retrieval Heads
    print("cma for ID Retrieval Heads...(skipped for head patching)")

    # prompt = f"In this image there is a {colors[0]} {shapes[0]} and a"
    prompt = "In this image there is a pink circle, a orange square, a purple heart and a"
    shapes = ["circle", "square", "heart", "triangle"]
    colors = ["pink", "orange", "purple", "blue"]

    image_c1 = generate_custom_image(
        shapes=shapes,
        colors=colors,
        coords=[(0,0), (0,1), (1,0), (1,1)]
    )
    image_c2 = generate_custom_image(
        shapes=shapes,
        colors=colors,
        coords=[(0,0), (0,1), (1,1), (1,0)]
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

    print("cma finished")

    return mediation_scores_2

def get_top_k_heads(mediation_scores: np.ndarray, k: int) -> List[Tuple[int, int]]:
    """
    Returns the (layer, head) coordinates for the top k highest mediation scores.
    """
    # 1. Flatten, sort ascending, reverse to descending, and grab top k
    top_k_flat_indices = np.argsort(mediation_scores.flatten())[::-1][:k]    # [::-1]=[-1::-1]=[start:stop:step]
    
    # 2. Convert flat 1D indices back into 2D (layer, head) coordinates
    layers, heads = np.unravel_index(top_k_flat_indices, mediation_scores.shape)
    layers = layers.tolist()    # np.int64 -> int
    heads = heads.tolist()

    print(f"Found top {k}/{mediation_scores.size} heads.")

    return list(zip(layers, heads))

def main():
    print("=== Execution Suite: Live Mechanistic Head Interventions ===")
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    # model_id = "bczhou/tiny-llava-v1-hf"
    config = load_config()
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)    
    num_layers = get_num_hidden_layers(model)
    _, num_heads = _resolve_text_model_dims(model)
    
    image_c1 = generate_custom_image(
        cols=3,
        rows=3,
        shapes=["circle", "square", "heart", "triangle"],
        colors=["pink", "orange", "purple", "blue"],
        coords=[(0,0), (0,1), (1,0), (1,1)],
        save_path='dataset/figure_29/1.png'
    )
    image_c2 = generate_custom_image(
        cols=3,
        rows=3,
        shapes=["circle", "triangle", "heart", "square"],
        colors=["pink", "blue", "purple", "orange"],
        coords=[(0,1), (0,2), (1,1), (1,2)],
        save_path='dataset/figure_29/2.png'
    )

    prompt_1 = "In this image there is a pink circle, a orange square, a purple heart and a"
    prompt_2 = "In this image there is a pink circle, a blue triangle, a"

    text_prompt_c1 = get_text_prompt(model, prompt_1, image_c1, processor)
    text_prompt_c2 = get_text_prompt(model, prompt_2, image_c2, processor)

    mediation_scores = scores_for_ID_selection(
        model=model,
        processor=processor,
        num_layers=num_layers,
        num_heads= num_heads,
        runs=10
    )
    top_k = int(0.1*num_layers*num_heads)
    top_k_heads = get_top_k_heads(mediation_scores, top_k)

    top_scores_list = [mediation_scores[l, h] for l, h in top_k_heads]
    print("top_scores_list:", top_scores_list)

    predicted_word = cma_head_patching(
        model=model,
        processor=processor,
        num_layers=num_layers,
        num_heads=num_heads,
        prompt_c1=text_prompt_c2,
        prompt_c2=text_prompt_c1,
        image_c1=image_c2,
        image_c2=image_c1,
        top_k_heads=top_k_heads
    )

if __name__ == "__main__":
    main()