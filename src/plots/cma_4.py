# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

import os
import glob
from PIL import Image
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

from src.model.loader import load_vlm
from src.utils.tools import load_config, _resolve_text_model_dims, get_text_prompt, get_num_hidden_layers, get_token_position
from src.plots.cma_1d import run_mediation_analysis
from src.mech_interp.cma import cma_head_patching_by_logits, get_head_embeddings, get_top_k_heads


