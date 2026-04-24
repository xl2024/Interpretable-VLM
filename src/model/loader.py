import torch
import warnings
from transformers import AutoProcessor, AutoModelForImageTextToText
from nnsight import LanguageModel
from src.utils.hardware import get_hardware_config

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
    processor.patch_size = 14
    
    # 4. Load and wrap the model with nnsight
    # nnsight's LanguageModel class inherits the underlying HF architecture
    # but builds the computation graph required for spatial causal swaps.
    print(f"Loading and tracing model weights with {hw_config['dtype']}...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        **load_kwargs
    )
    model = LanguageModel(hf_model)
    print("Load sequence complete. Model is ready for intervention.")
    return model, processor