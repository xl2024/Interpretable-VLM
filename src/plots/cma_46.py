import numpy as np
import random

from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import predict, get_num_hidden_layers, load_config, get_coord_from_index, is_equiv, _resolve_text_model_dims, get_text_prompt, get_token_position
from src.mech_interp.cma import cma_head_ablate_and_generate, get_top_k_heads
from src.plots.cma_1d import run_mediation_analysis

# Reproduces Figure 46 in Appendix C


def get_counting_dataset(num_imgs):
    dataset = []
    for i in range(num_imgs):
        colors_list = ['red', 'blue', 'green', 'yellow']
        shapes_list = ['triangle', 'square', 'circle', 'cross']
        feat_coords = [get_coord_from_index(j) for j in range(16)]
        pos_coords = [get_coord_from_index(j) for j in range(9)]
        N = random.randint(3, 8)
        sampled_feat_coords = random.sample(feat_coords, N)
        coords = random.sample(pos_coords, N)
        print("sampled_feat_coords:", sampled_feat_coords)
        colors = [colors_list[j[0]] for j in sampled_feat_coords]
        shapes = [shapes_list[j[1]] for j in sampled_feat_coords]
        save_path = f"dataset/figure_46/{i}.png"
        img = generate_custom_image(cols=3, rows=3, shapes=shapes, colors=colors, coords=coords, save_path=save_path)
        dataset.append({"image": img, "index": i, "count": len(set(colors))})

    return dataset

def cma_counting_trials(model, processor, num_heads, dataset, top_k_heads):
    corr_trials = 0
    prompt = (
        "You are given an image containing multiple colored objects. "
        "Your task is to carefully observe the image and identify all the unique colored objects present.\n"
        "Enumerate all the unique colored objects you find in the image, providing a numbered list for clarity.\n"
        "After listing the objects, provide the total count of these unique colored objects.\n"
        "Format the total count by writing 'Answer:' followed by the number. "
        "It is crucial to adhere to this format: 'Answer: TOTAL_COUNT'."
    )
    for image_data in dataset:
        img, gt = image_data["image"], image_data["count"]
        text_prompt = get_text_prompt(model, prompt, img, processor, use_system_prompt=False)
        predicted_words = cma_head_ablate_and_generate(
            model=model,
            processor=processor,
            num_heads=num_heads,
            prompt_text=text_prompt,
            image=img,
            top_k_heads=top_k_heads,
            max_new_tokens=100
        )
        pred = predicted_words.split('.')[0].split('Answer: ')

        print(f"index: {image_data["index"]}, gt: {gt}, pred: {pred}")
        if len(pred) == 2 and eval(pred[-1]) == gt:
            corr_trials += 1
        else:
            print(f"pred={pred}, target_count={gt}")
        
    
    return corr_trials / len(dataset)

def cma_counting_by_model(model_id, dataset):
    print("=== Starting Figure 46 Reproduction ===")
    config = load_config()
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)
    _, num_heads = _resolve_text_model_dims(model)
    mediation_scores_list = run_mediation_analysis(model_id)
    mediation_scores = {}
    for l,h in mediation_scores_list[0]:
        mediation_scores[l,h] = max(mediation_scores_list[0][l,h], mediation_scores_list[1][l,h], mediation_scores_list[2][l,h])
    # k_list = [0,10,20,50,100,150,200,250,300,400,500]
    k_list = [10,500]
    accs = {"Max": {}, "Bottom": {}}
    for cat in accs.keys():
        for k in k_list:
            print(f"{len(dataset)} trials for ablation of {cat} Top-{k} heads...")
            if cat == "Max":
                top_k_heads = get_top_k_heads(mediation_scores, k)
            else:
                top_k_heads = get_top_k_heads(mediation_scores, k, max_k=False)
            accs[cat][k] = cma_counting_trials(model, processor, num_heads, dataset, top_k_heads)

    return accs


def main():
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    # model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    model_name = model_id.replace('/', '_')
    num_imgs = 100
    dataset = get_counting_dataset(num_imgs)
    accs = cma_counting_by_model(model_id, dataset)
    print(f"accs: {accs}")

    save_path = f"outputs/cma/count/cma_fig_46_{model_name}.png"



if __name__ == "__main__":
    main()