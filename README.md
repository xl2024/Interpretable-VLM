This repository contains the code to reproduce all experimental results and figures from [arXiv:2506.15871](https://arxiv.org/pdf/2506.15871) using the `nnsight` library. A summary is available at [Blog](https://xiangpingliu.pages.dev/blog/reproduction-of-visual-symbolic-mechanisms-emergent-symbol-processing-in-vision-language-models/).

For a fast, interactive introduction to the codebase, please see [`demo.ipynb`](demo.ipynb).

---

## Getting Started

### Installation
Clone the repository and install the required dependencies before running any scripts:

```bash
git clone https://github.com/xl2024/Interpretable-VLM.git
cd Interpretable-VLM
pip install -r requirements.txt
```

### Running Experiments
All execution scripts for experiments and plotting are located in the `src/plots/` directory. You can run them as Python modules from the root directory.

**Example:**
```bash
python -m src.plots.pca_1b
```

---

## Project Structure

```text
.
├── configs/
│   └── config.yaml
├── dataset/                  # Test datasets
│   ├── figure_29/            # Image pairs for Figure 29 reproduction
│   └── figure_3/             # PUG-style dataset for Figure 3 reproduction
├── outputs/                  # Experiment results
│   ├── cma/
│   ├── pca/
│   ├── rsa/
│   └── v0/                   # Historic reference outputs
├── src/
│   ├── data/                 # Dataset generation and experiment data records
│   │   ├── cma/
│   │   ├── test_samples/
│   │   ├── generate_pug_dataset.py
│   │   └── synthetic_generator.py
│   ├── math_core/
│   │   └── rsa.py            # Common RSA computations
│   ├── mech_interp/
│   │   ├── cma.py            # Common Causal Mediation tracking & patching
│   │   └── tracer.py
│   ├── model/
│   │   └── loader.py
│   ├── plots/
│   │   ├── cma_*.py          # Scripts to reproduce and plot CMA results
│   │   ├── pca_*.py          # Scripts to reproduce and plot PCA results
│   │   └── rsa_*.py          # Scripts to reproduce and plot RSA results
│   └── utils/
│       ├── hardware.py
│       └── tools.py
├── demo.ipynb                # Quickstart Jupyter Notebook
```

---

## Reproduction Matrix

The following table details which script maps to which figure, alongside their datasets and output directories.

| Script | Reproduced Target | Dataset | Raw Data Filepath | Figure/Table Filepath | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `pca_1b.py` | Figure 1(b) | - | - | `outputs/pca/pca_fig_1b.png` | - |
| `rsa_1c.py` | Figures 1(c), 14–19 | - | - | `outputs/rsa/stage/` | - |
| `cma_1d.py` | Figures 1(d), 20–25 | - | `src/data/cma/scores/` | `outputs/cma/scores/` | - |
| `rsa_2.py` | Figures 2, 26–28 | - | - | `outputs/rsa/pos/` | - |
| `cma_3.py` | Figures 3(b), 37–43 | `dataset/figure_3/` ¹ | `src/data/cma/sweeping/` | `outputs/cma/sweeping/` | Does not repeat for standard errors (sweeping across all models is out of my budget). |
| `cma_4.py` | Figure 4 | - | `src/data/cma/color/` | `outputs/cma/cma_fig_4.png` | - |
| `cma_4_v2.py` | *Additional experiment* | - | `src/data/cma/color_v2/` | - | Patches on queries instead of keys and shows it is less effective. |
| `cma_5.py` | Figure 5 | - | `src/data/cma/reuse/` | `outputs/cma/cma_fig_5.png` | - |
| `rsa_6.py` | Figures 6, 7, 30–36 | - | `src/data/cma/entr/fig_6_results.json` | `outputs/rsa/entr/` | - |
| `rsa_6_v2.py`| *Additional experiment* | - | `src/data/cma/entr_v2/fig_6_results.json`| `outputs/rsa/entr_v2/` | Includes the last object's color in the prompt. |
| `cma_29.py` | Figure 29 | `dataset/figure_29/` | `src/data/cma/fig_29_results.npz` | `outputs/cma/cma_fig_29.png` | - |
| `cma_45.py` | Figure 45 | `dataset/coco/` ² | `src/data/cma/coco/` | `outputs/cma/cma_fig_45.png` | - |
| `cma_46.py` | Figure 46 | `dataset/figure_46/` ² | - | `outputs/cma/count/` | - |
| `cma_46_v2.py`| *Additional experiment* | `dataset/figure_46_v2/` ²| - | `outputs/cma/count_v2/` | Improved prompt for better base accuracy |
| `cma_tbl_1.py`| Table 1 | - | `src/data/cma/entr/` | `outputs/cma/entr/cma_tbl_1` | For images with a 3x3 grid and top_k=50. |
| `cma_tbl_1_v2.py`| Table 1 | - | `src/data/cma/entr_v2/` | `src/data/cma/entr_v2/cma_tbl_1` | For images with a 2x2 grid and top_k=50. |
| `pca_new_26.py`³| Figure 26 | - | - | `outputs/pca/` | Represents the **May** version of the paper (all other scripts track the January version for figure numbering). |

### Notes
> **¹** Generated using `src/data/generate_pug_dataset.py`. See [`dataset/figure_3/README.md`](dataset/figure_3/README.md) for generation details.  
> **²** Created from running script.  
> **³** `pca_new_26_v2.py` was used to generate results in the same output directory and `pca_new_26.py` requires less memory and more time.