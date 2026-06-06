from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import gc
import torch
import json
from pathlib import Path

from src.math_core.rsa import compute_rsa_scores
from src.mech_interp.tracer import rsa_tracer
from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import predict, get_num_hidden_layers, load_config, get_coord_from_index, is_equiv
from src.plots.rsa_1c import get_dynamic_token_indices

# Reproduces Figure 6, 7 and 30-36

# model_id = "Qwen/Qwen2-VL-7B-Instruct"                      # Figure 6 and 30
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct"                    # Figure 31
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct"                    # Figure 32
# model_id = "Qwen/Qwen2.5-VL-32B-Instruct"                   # Figure 33
# model_id = "llava-hf/llava-1.5-7b-hf"                       # Figure 34
# model_id = "llava-hf/llava-1.5-13b-hf"                      # Figure 35
# model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"        # Figure 36

def plot_rsa_figures(
    rsa_results: Dict[str, Dict[str, float]],
    num_layers: int,
    save_path: str
):
    print("Generating RSA graph...")
    
    for pos in ['Prompt', 'Last']:
        fig, ax = plt.subplots(figsize=(6, 4))
        layers = list(range(num_layers))
        colors = {'Low': 'blue', 'High': 'red'}
        for entr in ['Low', 'High']:
            ax.plot(layers, rsa_results[entr][pos], label=f'{entr} Entropy', color=colors[entr])
    
        titles = {'Prompt': 'Prompt Tokens', 'Last': 'Last Token'}
        ax.set_title(f"{titles[pos]} - Position RSA", fontweight="bold", fontsize=12, loc="center", pad=12)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Style the axes
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(r'RSA Correlation ($r$)', fontsize=12) 
        
        # Add legend without the bounding box
        ax.legend(frameon=False, loc='upper left', fontsize=11)
        
        # Save to your outputs folder
        plt.tight_layout()
        save_name = f'{save_path}_{pos.lower()}.png'
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Graph saved successfully to {save_name}")
        
        # Display the graph
        plt.show()

def get_trial_data(model, processor, color_list, shape_list, num_trials):
    trials = []
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

        inputs = processor(text=text_prompt, images=img, return_tensors="pt")
        trials.append({'inputs': inputs, 'trial': obj_indices})

        pred = predict(model, processor, img, text_prompt, max_new_tokens=10, new_only=True).split('.')[0].split()

        equiv_shapes = [
            ['airplane', 'plane'],
            ['x', 'cross'],
            ['rectangle', 'square'],
            ['light bulb', 'sun with rays', 'sun']
            # ['dot', 'sun'] happens but shouldn't be equiv
        ]
        pred_color = pred[0].strip().lower()
        pred_shape = ' '.join(pred[1:]).strip().lower()
        if len(pred) >= 2 and pred_color == obj_indices[-1]['color'] and is_equiv(pred_shape, obj_indices[-1]['shape'], equiv_shapes):
            corr_trials += 1
        else:
            print(f"pred={pred}, target_color={obj_indices[-1]['color']}, target_shape={obj_indices[-1]['shape']}")
        
    return trials, corr_trials / num_trials

def rsa_entr_by_model(model_id, num_trials, repeat, save_path):
    print("=== Starting Figure 6 RSA Reproduction ===")
    config = load_config()
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)
    num_layers = get_num_hidden_layers(model)
    colors_list = [
        ['red', 'yellow', 'gray', 'blue', 'pink', 'green', 'black', 'purple', 'orange'],
        ['red', 'green', 'blue', 'red', 'blue', 'red', 'blue', 'green', 'green']
    ]
    shapes_list = [
        ['circle', 'star', 'plane', 'square', 'umbrella', 'triangle', 'sun', 'heart', 'cross'],
        ['circle', 'circle', 'square', 'square', 'triangle', 'triangle', 'circle', 'triangle', 'square']
    ]

    rsa_results = {}
    for j in range(repeat):
        rsa_results[j] = {}
        for i, entr in enumerate(['High', 'Low']):
            trials, acc = get_trial_data(model, processor, colors_list[i], shapes_list[i], num_trials)
            
            print(f"\nExecuting RSA across {len(trials)} trials and {num_layers} layers...")
            hidden_states_by_trial = rsa_tracer(model, config, num_layers, trials)

            print("Calculating RSA for Prompt Tokens...")
            rsa_scores_prompt, rsa_scores_last_token = compute_rsa_scores(hidden_states_by_trial, trials, num_layers)
            rsa_results[j][entr] = {"Prompt": rsa_scores_prompt['pos'], "Last": rsa_scores_last_token['pos'], "Acc": acc}
        
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    plot_rsa_figures(rsa_results=rsa_results[0], num_layers=num_layers, save_path=save_path)

    return rsa_results

def process_rsa_data(all_rsa_results):
    # Helper function: Calculates Mean and Standard Error of the Mean (SEM)
    def get_mean_err(arr):
        mean_val = np.mean(arr)
        # ddof: int, optional
        # Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements. By default ddof is zero.
        err_val = np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        return mean_val, err_val

    models = list(all_rsa_results.keys())
    
    data_acc = {'high_means': [], 'high_errs': [], 'low_means': [], 'low_errs': []}
    data_prompt = {'high_means': [], 'high_errs': [], 'low_means': [], 'low_errs': []}
    data_last = {'high_means': [], 'high_errs': [], 'low_means': [], 'low_errs': []}
    
    for model in models:
        repeats_data = all_rsa_results[model]
        
        acc_h, acc_l = [], []
        prompt_h, prompt_l = [], []
        last_h, last_l = [], []
        
        for rep_id, rep_data in repeats_data.items():
            acc_h.append(rep_data['High']['Acc'])
            acc_l.append(rep_data['Low']['Acc'])
            
            prompt_h.append(np.mean(rep_data['High']['Prompt']))
            prompt_l.append(np.mean(rep_data['Low']['Prompt']))
            
            last_h.append(np.mean(rep_data['High']['Last']))
            last_l.append(np.mean(rep_data['Low']['Last']))
            
        m, e = get_mean_err(acc_h)
        data_acc['high_means'].append(m)
        data_acc['high_errs'].append(e)

        m, e = get_mean_err(acc_l)
        data_acc['low_means'].append(m)
        data_acc['low_errs'].append(e)
        
        m, e = get_mean_err(prompt_h)
        data_prompt['high_means'].append(m)
        data_prompt['high_errs'].append(e)

        m, e = get_mean_err(prompt_l)
        data_prompt['low_means'].append(m)
        data_prompt['low_errs'].append(e)
        
        m, e = get_mean_err(last_h)
        data_last['high_means'].append(m)
        data_last['high_errs'].append(e)

        m, e = get_mean_err(last_l)
        data_last['low_means'].append(m)
        data_last['low_errs'].append(e)

    plot_configs = [
        {
            'title': 'Model Performance (3x3 Grid)',
            'ylabel': 'Accuracy',
            'models': models,
            **data_acc
        },
        {
            'title': 'Prompt Tokens - Position RSA (3x3)',
            'ylabel': 'RSA Correlation (r)',
            'models': models,
            **data_prompt
        },
        {
            'title': 'Last Token - Position RSA (3x3)',
            'ylabel': 'RSA Correlation (r)',
            'models': models,
            **data_last
        }
    ]
    return plot_configs

def plot_bar_chart(config, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    models = config['models']
    x = np.arange(len(models))
    width = 0.35  
    
    color_high = '#666666'  # Dark Gray
    color_low = '#cccccc'   # Light Gray
    edge_color = '#333333'
    
    # Plot High Entropy bars
    ax.bar(x - width/2, config['high_means'], width, yerr=config['high_errs'],
           label='High Entropy', color=color_high, edgecolor=edge_color,
           capsize=3, error_kw={'elinewidth': 1.5, 'ecolor': edge_color})
           
    # Plot Low Entropy bars
    ax.bar(x + width/2, config['low_means'], width, yerr=config['low_errs'],
           label='Low Entropy', color=color_low, edgecolor=edge_color,
           capsize=3, error_kw={'elinewidth': 1.5, 'ecolor': edge_color})
           
    # Apply text and styling
    ax.set_ylabel(config['ylabel'], fontsize=11)
    ax.set_title(config['title'], fontweight='bold', fontsize=12, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.spines['bottom'].set_visible(False)
    ax.axhline(0, color='black', linewidth=1)
    ax.tick_params(axis='x', bottom=False)
    ax.legend(loc='best')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_name = f"{save_path}.png"
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close(fig) 
    print(f"Bar Chart Saved at: {save_name}")

def main():
    model_id_list = [
        ("Qwen/Qwen2-VL-7B-Instruct", "Qwen 2\n7B", "6_30"),
        ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen 2.5\n3B", "31"),
        ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen 2.5\n7B", "32"),
        ("Qwen/Qwen2.5-VL-32B-Instruct", "Qwen 2.5\n32B", "33"),
        ("llava-hf/llava-1.5-7b-hf", "LLaVA 1.5\n7B", "34"),
        ("llava-hf/llava-1.5-13b-hf", "LLaVA 1.5\n13B", "35"),
        ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "LLaVA One\n7B", "36")
    ]
    num_trials = 100
    repeat = 5

    filename = f"src/data/cma/entr/fig_6_results.json"
    file_path = Path(filename)
    if file_path.exists():
        print(f"Found {filename}! Loading results for entropy results...")
        with open(filename, 'r') as f:
            all_rsa_results = json.load(f)
    else:    
        all_rsa_results = {}

    for model_id, model_label, fig_num in model_id_list:
        model_name = model_id.replace('/', '_')
        if model_label not in all_rsa_results:
            save_path = f"outputs/rsa/entr/rsa_fig_{fig_num}_{model_name}"
            all_rsa_results[model_label] = rsa_entr_by_model(model_id, num_trials, repeat, save_path)

    with open(filename, 'w') as f:
        json.dump(all_rsa_results, f, indent=4)
    print(f"Entropy results successfully saved in {filename}.")

    plot_configs = process_rsa_data(all_rsa_results)
    save_paths = [f"outputs/rsa/entr/rsa_fig_7_{x}" for x in ['a','b','c']]
    for config, save_path in zip(plot_configs, save_paths):
        plot_bar_chart(config, save_path)


if __name__ == "__main__":
    main()