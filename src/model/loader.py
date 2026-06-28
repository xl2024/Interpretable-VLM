import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration, Idefics2ForConditionalGeneration
from nnsight import LanguageModel
from src.utils.hardware import get_hardware_config
from src.utils.tools import set_num_key_value_heads

def load_vlm(model_id: str, tier: str):
    """
    Loads a Vision-Language Model and its processor, wrapped in nnsight for 
    causal interventions. Dynamically routes hardware constraints based on the tier.
    
    Args:
        model_id (str): Hugging Face model string (e.g., 'bczhou/TinyLLaVA-1.5B')
        tier (str): Execution tier ('local', 'colab', 'cloud')
        
    Returns:
        tuple: (nnsight_model, processor)
    """
    print(f"Initializing Load Sequence for: {model_id} on {tier}...")
    
    # 1. Fetch hardware constraints from router
    hw_config = get_hardware_config(tier)
    model_id_lower = model_id.lower()

    # 2. Construct dynamic loading arguments
    load_kwargs = {
        "dtype": hw_config["dtype"],
        "trust_remote_code": True, # Required for custom Qwen/LLaVA vision modules
    }

    # Handle device mapping and 4-bit quantization routing
    if hw_config.get("quantization_config") is not None:
        load_kwargs["quantization_config"] = hw_config["quantization_config"]
        # BitsAndBytes requires device_map="auto" to intelligently place 4-bit layers
        load_kwargs["device_map"] = "auto"
    else:
        # Standard unquantized loading to the specific tier device
        load_kwargs["device_map"] = hw_config["device"]

    # 3. Load the Multimodal Processor
    # This handles LLaVA's <image> tokens and Qwen's <box> bounding boxes
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True
    )

    # if "qwen" in model_id_lower and hw_config["dtype"] == torch.bfloat16:
    #     print("Applying Qwen-specific Flash Attention 2 optimizations...")
    #     try:
    #         load_kwargs["attn_implementation"] = "flash_attention_2"
    #     except Exception:
    #         pass # Fallback to standard attention if not installed

    if "llava" in model_id_lower:
        processor.patch_size = 14
    
    # 4. Load and wrap the model with nnsight
    # nnsight's LanguageModel class inherits the underlying HF architecture
    # but builds the computation graph required for spatial causal swaps.
    print(f"Loading and tracing model weights with {hw_config['dtype']}...")
    if "onevision" in model_id_lower:
        hf_model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    elif "idefics" in model_id_lower:
        hf_model = Idefics2ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    elif "qwen2.5" in model_id_lower:
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    elif "qwen" in model_id_lower:
        hf_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    elif "llava" in model_id_lower:
        hf_model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)

    model = LanguageModel(hf_model)
    print("Load sequence complete. Model is ready for intervention.")
    return model, processor

def ungroup_nnsight_vlm(model, hidden_size, num_heads, num_kv_heads):
    """
    Surgically ungroups an nnsight-wrapped Hugging Face VLM in-place, converting 
    shared Grouped Query Attention into isolated Multi-Head Attention.
    """
    if num_kv_heads == num_heads:
        return model
    
    num_groups = num_heads // num_kv_heads
    head_dim = hidden_size // num_heads

    # Access the raw PyTorch model underneath nnsight's wrapper
    raw_model = getattr(model, "_model", model)

    print(f"Ungrouping weights: Expanding {num_kv_heads} KV heads -> {num_heads} isolated KV heads...")
    
    def get_fp_weights(proj_layer):
        """Safely extracts weights as float16/bfloat16, unpacking 4-bit if necessary."""
        if hasattr(proj_layer.weight, "quant_state"):  # bitsandbytes 4-bit detection
            import bitsandbytes as bnb
            # Dequantize to the active compute dtype (usually bfloat16 or float16)
            return bnb.functional.dequantize_4bit(
                proj_layer.weight.data, 
                proj_layer.weight.quant_state
            ).to(raw_model.dtype)
        return proj_layer.weight.data.to(raw_model.dtype)
    
    for name, module in raw_model.named_modules():
        # Locate self-attention modules containing standard HF projection linear layers
        if hasattr(module, "k_proj") and hasattr(module, "v_proj"):
            
            # --- 1. Expand k_proj ---
            k_w = get_fp_weights(module.k_proj)
            # Reshape to (kv_heads, head_dim, hidden), repeat heads, flatten back
            new_k_w = k_w.view(num_kv_heads, head_dim, -1).repeat_interleave(num_groups, dim=0).view(num_heads * head_dim, -1)
            module.k_proj.weight = nn.Parameter(new_k_w)
            module.k_proj.out_features = num_heads * head_dim
            
            if module.k_proj.bias is not None:
                k_b = module.k_proj.bias.data
                new_k_b = k_b.view(num_kv_heads, head_dim).repeat_interleave(num_groups, dim=0).view(-1)
                module.k_proj.bias = nn.Parameter(new_k_b)

            # --- 2. Expand v_proj ---
            v_w = get_fp_weights(module.v_proj)
            new_v_w = v_w.view(num_kv_heads, head_dim, -1).repeat_interleave(num_groups, dim=0).view(num_heads * head_dim, -1)
            module.v_proj.weight = nn.Parameter(new_v_w)
            module.v_proj.out_features = num_heads * head_dim
            
            if module.v_proj.bias is not None:
                v_b = module.v_proj.bias.data
                new_v_b = v_b.view(num_kv_heads, head_dim).repeat_interleave(num_groups, dim=0).view(-1)
                module.v_proj.bias = nn.Parameter(new_v_b)

            # --- 3. Update internal attention routing flags ---
            if hasattr(module, "num_key_value_heads"):
                module.num_key_value_heads = num_heads
            if hasattr(module, "num_key_value_groups"):
                module.num_key_value_groups = 1

    # Update global config objects so standard SDPA / FlashAttention treats it as MHA
    set_num_key_value_heads(model, num_heads)

    print("Model successfully ungrouped. Ready for clean surgical Causal Mediation Analysis.")
    return model