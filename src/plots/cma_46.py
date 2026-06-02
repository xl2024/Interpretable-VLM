import numpy as np

from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import predict, get_num_hidden_layers, load_config, get_coord_from_index, is_equiv, _resolve_text_model_dims
from src.mech_interp.cma import cma_head_patching_by_generator, get_head_embeddings, get_top_k_heads
from src.plots.rsa_1c import get_dynamic_token_indices
from src.plots.cma_1d import run_mediation_analysis

# Reproduces Figure 46 in Appendix C


def cma_counting_trials(model, processor, num_layers, num_heads, color_list, shape_list, num_trials, top_k_heads):
    corr_trials = 0
    for i in range(num_trials):
        shuffle = np.random.permutation(len(color_list))
        shapes, colors = [], []
        for j in range(len(color_list)):
            shapes.append(shape_list[shuffle[j]])
            colors.append(color_list[shuffle[j]])

        coords = [get_coord_from_index(j) for j in range(len(color_list))]

        img = generate_custom_image(cols=3, rows=3, shapes=shapes, colors=colors, coords=coords)
        
        obj_indices, text_prompt = get_dynamic_token_indices(
            model, processor, colors=colors, shapes=shapes, coords=coords, image=img, last_color=False
        )

        pred = predict(model, processor, img, text_prompt, max_new_tokens=10, new_only=True).split('.')[0].split()

        equiv_shapes = [
            ['x', 'cross'],
            ['rectangle', 'square']
        ]
        pred_color = pred[0].strip().lower()
        pred_shape = ' '.join(pred[1:]).strip().lower()
        if len(pred) >= 2 and pred_color == obj_indices[-1]['color'] and is_equiv(pred_shape, obj_indices[-1]['shape'], equiv_shapes):
            corr_trials += 1
        else:
            print(f"pred={pred}, target_color={obj_indices[-1]['color']}, target_shape={obj_indices[-1]['shape']}")
        
    
    return corr_trials

def cma_counting_by_model(model_id, num_trials):
    print("=== Starting Table 1 Reproduction ===")
    config = load_config()
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)
    num_layers = get_num_hidden_layers(model)
    _, num_heads = _resolve_text_model_dims(model)
    mediation_scores_list = run_mediation_analysis(model_id)
    mediation_scores = mediation_scores_list[1]
    # k_list = [10,20,50,100,150,200,250,300,400,500]
    k_list = [10,500]
    accs = {"Max": {}, "Bottom": {}}
    for cat in accs.keys():
        for k in k_list:
            top_k_heads = get_top_k_heads(mediation_scores, k)
            
            colors_list = ['red', 'blue', 'green', 'yellow']
            shapes_list = ['triangle', 'square', 'circle', 'cross']

            print(f"{num_trials} trials for ablation of {cat} Top-{k} heads...")
            accs[cat][k] = cma_counting_trials(
                model, processor, num_layers, num_heads, colors_list[0], shapes_list[0], num_trials, 
                get_embeds=True, interv=False, top_k_heads=top_k_heads
            )

    return accs


def main():
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    # model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    model_name = model_id.replace('/', '_')
    num_trials = 100
    accs = cma_counting_by_model(model_id, num_trials)

    save_path = f"outputs/cma/count/cma_fig_46_{model_name}.png"



if __name__ == "__main__":
    main()