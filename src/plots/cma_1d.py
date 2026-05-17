import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Any

# Internal project imports
from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import load_config, _resolve_text_model_dims, get_text_prompt, predict
from src.plots.rsa_1c import get_num_hidden_layers
from src.mech_interp.cma import cma_headwise

# model_id = "Qwen/Qwen2-VL-7B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
# model_id = "llava-hf/llava-1.5-7b-hf"
model_id = "llava-hf/llava-1.5-13b-hf"
# model_id = "bczhou/tiny-llava-v1-hf"

def run_mediation_analysis(
    model: Any,
    processor: Any,
    config: Dict[str, Any],
    num_layers: int,
    num_heads: int
) -> Tuple[List[List[Any]], List[List[Any]], List[List[Any]]]:
    """
    Executes Causal Mediation Analysis (Activation Patching) across all attention heads.
    Patches activations from a modified context (c2) into the clean context (c1) following Eq. (1).
    """
    print("Preparing Causal Mediation Analysis...")

    # ID Retrieval Heads
    print("cma for ID Retrieval Heads...")

    prompt = "In this image there is a blue circle and a"

    image_c1 = generate_custom_image(
        shapes=["circle", "square"],
        colors=["blue", "red"],
        coords=[(0,0), (0,1)]
    )
    image_c2 = generate_custom_image(
        shapes=["circle", "square"],
        colors=["blue", "red"],
        coords=[(0,1), (0,0)]
    )

    text_prompt_c1 = get_text_prompt(model_id, prompt, image_c1, processor)
    text_prompt_c2 = get_text_prompt(model_id, prompt, image_c2, processor)

    print(f"Prediction: {predict(model, processor, image_c1, text_prompt_c1)} (target: red)")
    print(f"Prediction: {predict(model, processor, image_c2, text_prompt_c2)} (target: red)")

    token_inputs = processor(text=text_prompt_c1, images=image_c1, return_tensors="pt")
    input_ids = token_inputs["input_ids"][0].tolist()
    for index, token_id in enumerate(input_ids):
        token_str = processor.tokenizer.decode(token_id).strip().lower()
        if 'blue' in token_str:
            token_pos_1 = index
        elif 'circle' in token_str:
            token_pos_2 = index
            break
    token_pos = (token_pos_1, token_pos_2+1)

    a1_tokens = processor.tokenizer.encode("red", add_special_tokens=False)
    a1_star_tokens = processor.tokenizer.encode("blue", add_special_tokens=False)
    a1_id = a1_tokens[-1]
    a1_star_id = a1_star_tokens[-1]

    print(f"Target Token ID (a1): {a1_id} -> '{processor.tokenizer.decode([a1_id])}'")
    print(f"Contrast Token ID (a1*): {a1_star_id} -> '{processor.tokenizer.decode([a1_star_id])}'")

    mediation_scores_1 = cma_headwise(
        model=model,
        processor=processor,
        config=config,
        num_layers=num_layers,
        num_heads=num_heads,
        prompt_c1=text_prompt_c1,
        prompt_c2=text_prompt_c2,
        image_c1=image_c1,
        image_c2=image_c2,
        token_pos=token_pos,
        a1_id=a1_id,
        a1_star_id=a1_star_id
    )

    # ID Selection Heads
    print("cma for ID Selection Heads...")
    token_inputs = processor(text=text_prompt_c1, images=image_c1, return_tensors="pt")
    input_ids = token_inputs["input_ids"][0].tolist()
    token_pos = (len(input_ids)-1, len(input_ids))

    mediation_scores_2 = cma_headwise(
        model=model,
        processor=processor,
        config=config,
        num_layers=num_layers,
        num_heads=num_heads,
        prompt_c1=text_prompt_c1,
        prompt_c2=text_prompt_c2,
        image_c1=image_c1,
        image_c2=image_c2,
        token_pos=token_pos,
        a1_id=a1_id,
        a1_star_id=a1_star_id
    )

    # Feature Retrieval Heads
    print("cma for Feature Retrieval Heads...")

    image_c1 = generate_custom_image(
        shapes=["circle", "square"],
        colors=["blue", "red"],
        coords=[(0,0), (0,1)]
    )
    image_c2 = generate_custom_image(
        shapes=["circle", "square"],
        colors=["blue", "green"],
        coords=[(0,0), (0,1)]
    )

    text_prompt_c2 = get_text_prompt(model_id, prompt, image_c2, processor)
    
    print(f"Prediction: {predict(model, processor, image_c2, text_prompt_c2)} (target: green)")

    a1_star_tokens = processor.tokenizer.encode("green", add_special_tokens=False)
    a1_star_id = a1_star_tokens[-1]

    mediation_scores_3 = cma_headwise(
        model=model,
        processor=processor,
        config=config,
        num_layers=num_layers,
        num_heads=num_heads,
        prompt_c1=text_prompt_c1,
        prompt_c2=text_prompt_c2,
        image_c1=image_c1,
        image_c2=image_c2,
        token_pos=token_pos,
        a1_id=a1_id,
        a1_star_id=a1_star_id
    )

    print("cma finished")

    return mediation_scores_1, mediation_scores_2, mediation_scores_3

def plot_causal_mediation(
    mediation_scores: Tuple[List[List[Any]], List[List[Any]], List[List[Any]]],
    num_layers: int,
    num_heads: int,
    save_path: str
):
    scores_blue, scores_red, scores_green = mediation_scores

    scores_blue = np.clip(scores_blue, 0, None)
    scores_red = np.clip(scores_red, 0, None)
    scores_green = np.clip(scores_green, 0, None)
    
    scores_blue /= scores_blue.max()
    scores_red /= scores_red.max()
    scores_green /= scores_green.max()

    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12) 
    
    ax.set_xticks(list(range(0, num_heads, 5)))
    ax.set_xlim(-0.5, num_heads - 0.5) 
    ax.set_yticks(list(range(0, num_layers, 5)))
    ax.set_ylim(-0.5, num_layers - 0.5)

    ax.set_facecolor('black')
    ax.grid(False)

    for l in range(num_layers):
        for h in range(num_heads):
            color = (scores_red[l,h], scores_green[l,h], scores_blue[l,h])
            rect = Rectangle((h - 0.5, l - 0.5), 1, 1, 
                             facecolor=color, edgecolor='none')
            ax.add_patch(rect)

    colors = {'blue': '#0000FF', 'red': '#FF0000', 'green': '#00FF00'}
    handles = [plt.Line2D([0], [0], color=colors[c], marker='s', markersize=10, linestyle='') 
               for c in ['blue', 'red', 'green']]
    labels = ['ID Retrieval', 'ID Selection', 'Feature Retrieval']
    legend = ax.legend(handles, labels, frameon=True, facecolor='lightgray', edgecolor='darkgray', 
                       loc='lower right', title='Stages', title_fontsize=11, fontsize=10)
    legend.get_title().set_color('black')
    for text in legend.get_texts():
        text.set_color('black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph successfully saved to {save_path}")
    plt.show()

def main():
    print("=== Execution Suite: Live Mechanistic Head Interventions ===")
    config = load_config()
    # model_id = "bczhou/tiny-llava-v1-hf"
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)    
    num_layers = get_num_hidden_layers(model)
    _, num_heads = _resolve_text_model_dims(model)

    mediation_scores = run_mediation_analysis(
        model=model,
        processor=processor,
        config=config,
        num_layers=num_layers,
        num_heads= num_heads
    )
    
    plot_causal_mediation(
        mediation_scores=mediation_scores,
        num_layers=num_layers,
        num_heads=num_heads,
        save_path="outputs/cma_figure_1d.png"
    )

if __name__ == "__main__":
    main()