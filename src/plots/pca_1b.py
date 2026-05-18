import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import random
from typing import Dict, Any, Tuple

from src.model.loader import load_vlm
from src.data.synthetic_generator import generate_custom_image
from src.utils.tools import _resolve_layer_path, load_config, get_permutations, get_text_prompt, get_layer_path_template
from src.mech_interp.tracer import gc_collect

model_id = "Qwen/Qwen2-VL-7B-Instruct"

def collect_hidden_states_for_pca(
    model: Any,
    processor: Any,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs real forward passes to collect final-token hidden states at Layer 19 and Layer 27.
    Iterates through variations of positions and features to build a dense dataset for PCA.
    """
    # [Note: There are 6 objects with 6 positions, so overall there are 6!=720 different combinations, 
    # and no need for 7200 trials as claimed in paper.]
    permutations = get_permutations([i for i in range(6)])
    num_samples = len(permutations)
    print(f"Collecting real hidden states across {num_samples} forward passes...")
    
    # 6 distinct positions and 6 features to match the paper's 6 legend categories
    positions = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)]
    features = [
        ("red", "circle"),
        ("green", "triangle"),
        ("blue", "square"),
        ("purple", "star"),
        ("yellow", "heart"),
        ("orange", "cross")
    ]

    layer_19_idx = 19
    layer_27_idx = 27
    layer_template = get_layer_path_template(model)

    states_19 = []
    states_27 = []
    pos_labels = []
    feat_labels = []

    for i in range(num_samples):
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{num_samples} samples...")

        colors, shapes = [], []
        text = "In this image, there is a"
        t_pos = random.randint(0, 5)
        pos_labels.append(t_pos)
        for j in range(len(permutations[i])):
            color, shape = features[permutations[i][j]]
            colors.append(color)
            shapes.append(shape)
            if j == t_pos:
                feat_labels.append(permutations[i][j])
            else:
                text += f" {color} {shape}, a"
        text = text[:-3] + " and a"
        # Generate the specific combination canvas
        image = generate_custom_image(cols=3, rows=3, shapes=shapes, colors=colors, coords=positions)
        
        text_prompt = get_text_prompt(model, text, image, processor)
        
        if i < 10:
            print('text_prompt ', i, text_prompt)

        inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            with model.trace() as tracer:
                with tracer.invoke(**inputs):
                    # Resolve modules for extraction
                    l19_module = _resolve_layer_path(model, layer_template.format(layer_19_idx))
                    l27_module = _resolve_layer_path(model, layer_template.format(layer_27_idx))
                    
                    # Intercept the full output tuple, grab hidden states [0], and slice the last token [-1, :]
                    hs_19 = l19_module.output[0][-1, :].save()
                    hs_27 = l27_module.output[0][-1, :].save()
                    print("hs_19 shape:", hs_19.shape)
            gc_collect()
        print("hs_19.cpu() shape: ", hs_19.cpu().shape)
        print("hs_19.cpu()[0] shape: ", hs_19.cpu()[0].shape)
        print("hs_19.cpu()[0].to(torch.float32) shape: ", hs_19.cpu()[0].to(torch.float32).shape)
        print("hs_19.cpu()[0].to(torch.float32).numpy() shape: ", hs_19.cpu()[0].to(torch.float32).numpy().shape)
        states_19.append(hs_19.cpu()[0].to(torch.float32).numpy())
        states_27.append(hs_27.cpu()[0].to(torch.float32).numpy())
        print("shapes: ",hs_19.cpu()[0].to(torch.float32).numpy(), hs_27.cpu()[0].to(torch.float32).numpy())
    return np.array(states_19), np.array(states_27), np.array(pos_labels), np.array(feat_labels)

def plot_pca_figure_1b(
    states_19: np.ndarray, 
    states_27: np.ndarray, 
    pos_labels: np.ndarray, 
    feat_labels: np.ndarray, 
    save_path: str
):
    """
    Computes PCA on the extracted hidden states and maps them onto the 
    exact 2x2 grid aesthetic of Figure 1b.
    """
    print("Computing Principal Components...")
    # Fit PCA separately for each layer's manifold
    pca_19 = PCA(n_components=2)
    proj_19 = pca_19.fit_transform(states_19)

    pca_27 = PCA(n_components=2)
    proj_27 = pca_27.fit_transform(states_27)

    print("Rendering Figure 1b...")
    fig, axs = plt.subplots(2, 2, figsize=(8, 7))
    
    # Paper exact color mapping hex codes
    pos_colors = ['#00FF00', '#00FFFF', '#808000', '#8A2BE2', '#FF0000', '#FF00FF'] # Lime, Cyan, Olive, SlateBlue, Red, Magenta
    feat_colors = ['#FF0000', '#008000', '#0000FF', '#800080', '#FFD700', '#FFA500'] # Red, Green, Blue, Purple, Gold(better contrast), Orange

    # Map labels to colors arrays
    c_pos = [pos_colors[lbl] for lbl in pos_labels]
    c_feat = [feat_colors[lbl] for lbl in feat_labels]

    # Plot settings for dense overlapping scatter (tiny size dots, slight alpha)
    scatter_kwargs = {'s': 3, 'alpha': 0.7, 'edgecolors': 'none'}

    # Top Row: Colored by Position
    axs[0, 0].scatter(proj_19[:, 0], proj_19[:, 1], c=c_pos, **scatter_kwargs)
    axs[0, 1].scatter(proj_27[:, 0], proj_27[:, 1], c=c_pos, **scatter_kwargs)

    # Bottom Row: Colored by Features
    axs[1, 0].scatter(proj_19[:, 0], proj_19[:, 1], c=c_feat, **scatter_kwargs)
    axs[1, 1].scatter(proj_27[:, 0], proj_27[:, 1], c=c_feat, **scatter_kwargs)

    # Aesthetic stripping (no axes, no ticks, matching the paper's floating clusters)
    for ax in axs.flat:
        ax.axis('off')

    # Add Column Titles (Layers)
    fig.text(0.3, 0.92, 'Layer 19', ha='center', fontsize=14)
    fig.text(0.7, 0.92, 'Layer 27', ha='center', fontsize=14)

    # Add Row Titles (Coloring logic)
    fig.text(0.08, 0.7, 'Position', va='center', rotation='vertical', fontsize=14)
    fig.text(0.08, 0.3, 'Features', va='center', rotation='vertical', fontsize=14)
    
    # --- Legends ---
    # Create proxy artists for the legends using invisible (color='w'/'none') lines with colored circular markers
    pos_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8) for c in pos_colors]
    feat_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8) for c in feat_colors]
    
    # Dummy labels for position (since they are just dots in the paper legend)
    pos_legend_labels = [''] * 6 
    feat_legend_labels = ['Red Circle', 'Green Triangle', 'Blue Square', 'Purple Star', 'Yellow Heart', 'Orange Cross']

    # We attach the legends to the right-side axes, positioned outside the plot area
    # Target Position Legend (2 columns)
    leg_pos = axs[0, 1].legend(pos_handles, pos_legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5), 
                               title='Target Position', title_fontsize=12, ncol=2, frameon=True, 
                               facecolor='#F5F5F5', edgecolor='silver', handletextpad=0.1, columnspacing=0.5)
    leg_pos.get_title().set_fontweight('bold')

    # Target Features Legend (1 column)
    leg_feat = axs[1, 1].legend(feat_handles, feat_legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5), 
                                title='Target Features', title_fontsize=12, frameon=True, 
                                facecolor='#F5F5F5', edgecolor='silver')
    leg_feat.get_title().set_fontweight('bold')

    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved PCA grid to {save_path}")
    plt.show()

def main():
    print("=== Figure 1b Reproduction: PCA ===")
    config = load_config()
    
    # model_id = "Qwen/Qwen2-VL-7B-Instruct"
    tier = config['pipeline']['tier']
    model, processor = load_vlm(model_id, tier)

    states_19, states_27, pos_labels, feat_labels = collect_hidden_states_for_pca(
        model=model,
        processor=processor,
        config=config
    )
    
    plot_pca_figure_1b(
        states_19=states_19,
        states_27=states_27,
        pos_labels=pos_labels,
        feat_labels=feat_labels,
        save_path="outputs/pca_figure_1b.png"
    )

if __name__ == "__main__":
    main()