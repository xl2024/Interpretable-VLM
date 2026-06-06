import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import load_config, _resolve_text_model_dims, get_text_prompt, get_num_hidden_layers
from src.plots.cma_1d import run_mediation_analysis
from src.mech_interp.cma import cma_head_patching_by_logits, get_head_embeddings, get_top_k_heads

# Reproduces Figure 29


def aggregate_data(fig_29_results):
    aggregated = {}
    model_id_to_label = {
        "Qwen/Qwen2-VL-7B-Instruct": "Qwen 2\n7B",
        "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen 2.5\n3B",
        "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen 2.5\n7B",
        "Qwen/Qwen2.5-VL-32B-Instruct": "Qwen 2.5\n32B",
        "llava-hf/llava-1.5-7b-hf": "LLaVA-1.5\n7B",
        "llava-hf/llava-1.5-13b-hf": "LLaVA-1.5\n13B"
    }

    for model_id, model_label in model_id_to_label.items():
        if model_id not in fig_29_results:
            continue

        k_data = fig_29_results[model_id]
        best_k = None
        best_rel_mean = -1.0
        best_k_metrics = {}

        for k, repeats_data in k_data.item().items():
            if k == 0:
                continue

            rel_accs = []
            abs_accs = []

            for repeat, word_counts in repeats_data.items():
                rel_count = 0
                abs_count = 0
                
                for word, freq in word_counts.items():
                    w = str(word).lower().strip()
                    if w in ['orange', 'yellow']:
                        rel_count += freq
                    elif w in ['purple', 'pur']:
                        abs_count += freq

                # 20 counts total per repeat
                rel_accs.append(rel_count / 20.0)
                abs_accs.append(abs_count / 20.0)

            mean_rel = np.mean(rel_accs)
            
            if mean_rel > best_rel_mean:
                best_rel_mean = mean_rel
                
                # Calculate Standard Error of the Mean (SEM)
                se_rel = np.std(rel_accs, ddof=1) / np.sqrt(len(rel_accs)) if len(rel_accs) > 1 else 0
                
                mean_abs = np.mean(abs_accs)
                se_abs = np.std(abs_accs, ddof=1) / np.sqrt(len(abs_accs)) if len(abs_accs) > 1 else 0

                best_k = k
                best_k_metrics = {
                    'mean_rel': mean_rel,
                    'se_rel': se_rel,
                    'mean_abs': mean_abs,
                    'se_abs': se_abs
                }

        aggregated[model_label] = {
            'best_k': best_k,
            'metrics': best_k_metrics
        }
        
    return aggregated

def plot_position_patching(aggregated_data, save_path):
    models = list(aggregated_data.keys())
    
    x_labels = []
    means_abs, se_abs = [], []
    means_rel, se_rel = [], []
    
    for m in models:
        # Append k value to the model name for the x-axis tick
        k_val = aggregated_data[m]['best_k']
        x_labels.append(f"{m}\nk={k_val}")
        
        metrics = aggregated_data[m]['metrics']
        means_abs.append(metrics['mean_abs'])
        se_abs.append(metrics['se_abs'])
        means_rel.append(metrics['mean_rel'])
        se_rel.append(metrics['se_rel'])

    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot bars
    color_rel = '#666666'  # Dark Gray
    color_abs = '#cccccc'   # Light Gray
    edge_color = '#333333'
    ax.bar(x - width/2, means_abs, width, yerr=se_abs, 
           label='Absolute', color=color_abs, edgecolor=edge_color,
           capsize=3, error_kw={'elinewidth': 2.5, 'capthick': 2.5, 'ecolor': edge_color})
    
    ax.bar(x + width/2, means_rel, width, yerr=se_rel, 
           label='Relative', color=color_rel, edgecolor=edge_color,
           capsize=3, error_kw={'elinewidth': 2.5, 'capthick': 2.5, 'ecolor': edge_color})
    
    ax.set_ylabel('Proportion Correct', fontsize=11)
    ax.set_title('Position Patching Performance', fontsize=12, fontweight='bold', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.tick_params(bottom=False, left=False)
    # ax.tick_params(length=0)
    ax.set_ylim(0, 1.0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(frameon=True, edgecolor='lightgray', loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig) 
    print(f"Bar Chart Saved at: {save_path}")

def get_cma_test_cases():
    """
    generate pairs of figures a,b and save in dataset/figure_29 folder
    figure a always has a blue triangle at the common abs position
    figure b always has a purple heart at the common abs position
    and an orange square at the new rel position
    so that prompts and stats don't have to change
    2a and 2b are the same as the example in figure 13
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
                    other_pos_1 = [c for c in coords_1 if c!= common_abs_pos ]
                    other_pos_2 = [c for c in coords_2 if c not in [common_abs_pos, new_rel_pos]]
                    # ["pink", "orange", "purple", "blue"]
                    coords_1_list.append([other_pos_1[0], other_pos_1[1], other_pos_1[2], common_abs_pos])
                    coords_2_list.append([other_pos_2[0], new_rel_pos, common_abs_pos, other_pos_2[1]])

    return shapes, colors, coords_1_list, coords_2_list

def cma_test_by_model(model_id):
    mediation_scores = run_mediation_analysis(model_id)
    mediation_scores = mediation_scores[1]
    
    config = load_config()
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)    
    num_layers = get_num_hidden_layers(model)
    _, num_heads = _resolve_text_model_dims(model)

    shapes, colors, coords_1_list, coords_2_list = get_cma_test_cases()

    prompt_1 = [
        "In this image there is a pink circle, a orange square, a purple heart and a",
        "In this image there is a purple heart, a pink circle, a orange square and a",
        "In this image there is a orange square, a purple heart, a pink circle and a"
    ]
    prompt_2 = [
        "In this image there is a pink circle, a blue triangle, a",
        "In this image there is a blue triangle, a pink circle, a"
    ]
    # Qwen 2.5 VL 32B model always starts by In...
    system_format = "COLOR SHAPE, replacing COLOR and SHAPE with the color and shape of the missing object in the image. Do not repeat the prompt words, just append with the requested format"
    cma_by_model = {}
    # [Note: alpha=3, k=0 -> 'purple', k=1,...,21 -> 'orange', k>=22 -> 'blue']
    # top_k = int(0.1*num_layers*num_heads)
    for k in range(100):
        top_k_heads = get_top_k_heads(mediation_scores, k)
        cma_by_model[k] = {}
        for repeat in range(6):
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
                text_prompt_c1 = get_text_prompt(model, prompt_1[repeat % 3], image_c1, processor, format=system_format)
                text_prompt_c2 = get_text_prompt(model, prompt_2[repeat % 2], image_c2, processor, format=system_format)

                head_cache = get_head_embeddings(
                    model=model, 
                    processor=processor, 
                    num_heads=num_heads, 
                    prompt_list=[text_prompt_c1], 
                    image_list=[image_c1], 
                    top_k_heads=top_k_heads
                )
                
                predicted_word = cma_head_patching_by_logits(
                    model=model,
                    processor=processor,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    prompt_c1=text_prompt_c2,
                    image_c1=image_c2,
                    d_t_head_cache=head_cache,
                    top_k_heads=top_k_heads
                )

                if predicted_word not in predicted_words:
                    predicted_words[predicted_word] = 1
                else:
                    predicted_words[predicted_word] += 1

            predicted_words = dict(sorted(predicted_words.items(), key=lambda item: item[1], reverse=True))
            print(f"k={k}, repeat={repeat}: The model predicted: '{predicted_words}'")
            cma_by_model[k][repeat] = predicted_words

    return cma_by_model

def main():
    print("=== Execution Suite: Live Mechanistic Head Interventions ===")
    model_ids = ["Qwen/Qwen2-VL-7B-Instruct",    # rel: orange, abs: purple
                 "llava-hf/llava-1.5-13b-hf",
                 "Qwen/Qwen2.5-VL-7B-Instruct",    # rel: yellow/orange, abs: purple
                 "Qwen/Qwen2.5-VL-32B-Instruct",
                 "llava-hf/llava-1.5-7b-hf",    # rel: yellow, abs: pur
                 "Qwen/Qwen2.5-VL-3B-Instruct"    # rel: orange, abs: purple
                 ]
    # model_ids = ["llava-hf/llava-1.5-13b-hf"]

    filename = "src/data/cma/fig_29_results.npz"
    file_path = Path(filename)
    if file_path.exists():
        print(f"Found {filename}! Loading cma results for figure 29...")
        data = np.load(filename, allow_pickle=True)
        fig_29_results = dict(data)
        data.close()
    else:
        fig_29_results = {}

    for model_id in model_ids:
        if model_id not in fig_29_results:
            print(f"Generating results in figure 29 for {model_id}...")
            fig_29_results[model_id] = cma_test_by_model(model_id)
   
    np.savez(filename, **fig_29_results)
    print(f"fig_29_results Saved in {filename}.")

    fig_path = "outputs/cma/cma_fig_29.png"
    processed_data = aggregate_data(fig_29_results)    # if get a "no items()" error, just re-run this script
    plot_position_patching(processed_data, fig_path)


if __name__ == "__main__":
    main()