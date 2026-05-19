import numpy as np
from typing import Dict, List, Tuple, Any


from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import load_config, _resolve_text_model_dims, get_text_prompt, predict
from src.plots.rsa_1c import get_num_hidden_layers
from src.mech_interp.cma import cma_head_patching, run_cma_for_ID_selection




absolute_score = 0
relative_score = 0

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

    mediation_scores = run_cma_for_ID_selection(
        model=model,
        processor=processor,
        num_layers=num_layers,
        num_heads= num_heads,
        shapes=["circle", "triangle"],
        colors=["pink", "blue"]
    )
    # top_k = int(0.1*num_layers*num_heads)
    for k in range(300):
        top_k_heads = get_top_k_heads(mediation_scores, k)

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

        print(f"k={k}: The model predicted: '{predicted_word}'")

if __name__ == "__main__":
    main()