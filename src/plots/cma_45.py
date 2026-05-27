import torch
import random
from typing import List, Dict, Tuple, Any

from src.model.loader import load_vlm
from src.utils.tools import get_text_prompt, get_num_hidden_layers, _resolve_text_model_dims
from src.mech_interp.cma import cma_head_patching_by_logits, get_head_embeddings_and_generation

