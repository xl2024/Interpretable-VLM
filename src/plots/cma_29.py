import numpy as np
import copy
import random
from typing import Dict, List, Tuple, Any


from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import load_config, _resolve_text_model_dims, get_text_prompt, get_num_hidden_layers
from src.plots.cma_1d import run_mediation_analysis
from src.mech_interp.cma import cma_head_patching




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

def get_cma_test_cases():
    """
    generate pairs of figures a,b and save in dataset/figure_29 folder
    figure a always has a blue triangle at the common abs position
    figure b always has a purple heart at the common abs position
    and an orange square at the new rel position
    so that prompts and stats don't have to change
    """
    shapes = ["circle", "square", "heart", "triangle"]
    colors = ["pink", "orange", "purple", "blue"]
    coords_1_list = []
    coords_2_list = []
    base_coords = [(0,0),(0,1),(1,0),(1,1)]
    for s1 in base_coords:
        for s2 in base_coords:
            if s1 != s2:
                coords_1 = []
                coords_2 = []
                for i in range(4):
                    coords_1.append((base_coords[i][0]+s1[0], base_coords[i][1]+s1[1]))
                    coords_2.append((base_coords[i][0]+s2[0], base_coords[i][1]+s2[1]))
                common_coords = set(coords_1).intersection(set(coords_2))
                for common_abs_pos in list(common_coords):
                    new_rel_pos_1 = common_abs_pos[0] + s2[0] - s1[0]
                    new_rel_pos_2 = common_abs_pos[1] + s2[1] - s1[1]
                    new_rel_pos = (new_rel_pos_1, new_rel_pos_2)
                    other_pos_1 = list(set(coords_1) - set([common_abs_pos]))
                    other_pos_2 = list(set(coords_2) - set([common_abs_pos, new_rel_pos]))
                    # ["pink", "orange", "purple", "blue"]
                    coords_1_list.append([other_pos_1[0], other_pos_1[1], other_pos_1[2], common_abs_pos])
                    coords_2_list.append([other_pos_2[0], new_rel_pos, common_abs_pos, other_pos_2[1]])

    return shapes, colors, coords_1_list, coords_2_list

def main():
    print("=== Execution Suite: Live Mechanistic Head Interventions ===")
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    model_id = "bczhou/tiny-llava-v1-hf"
    config = load_config()
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)    
    num_layers = get_num_hidden_layers(model)
    _, num_heads = _resolve_text_model_dims(model)
    
    mediation_scores = run_mediation_analysis(model, processor, num_layers, num_heads)
    mediation_scores = mediation_scores[1]

    shapes, colors, coords_1_list, coords_2_list = get_cma_test_cases()

    prompt_1 = "In this image there is a pink circle, a orange square, a purple heart and a"
    prompt_2 = "In this image there is a pink circle, a blue triangle, a"

    # [Note: alpha=3, k=0 -> 'purple', k=1,...,21 -> 'orange', k>=22 -> 'blue']
    # top_k = int(0.1*num_layers*num_heads)
    for k in range(100):
        top_k_heads = get_top_k_heads(mediation_scores, k)
        predicted_words = {}
        for i in range(len(coords_1_list)):
            image_c1 = generate_custom_image(
                cols=3,
                rows=3,
                shapes=shapes,
                colors=colors,
                coords=coords_1_list[i],
                save_path=f'dataset/figure_29/{i+1}_a.png'
            )
            image_c2 = generate_custom_image(
                cols=3,
                rows=3,
                shapes=shapes,
                colors=colors,
                coords=coords_2_list[i],
                save_path=f'dataset/figure_29/{i+1}_b.png'
            )
            text_prompt_c1 = get_text_prompt(model, prompt_1, image_c1, processor)
            text_prompt_c2 = get_text_prompt(model, prompt_2, image_c2, processor)

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

            if predicted_word not in predicted_words:
                predicted_words[predicted_word] = 1
            else:
                predicted_words[predicted_word] += 1

        predicted_words = dict(sorted(predicted_words.items(), key=lambda item: item[1], reverse=True))
        print(f"k={k}: The model predicted: '{predicted_words}'")

if __name__ == "__main__":
    main()