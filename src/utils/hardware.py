import torch
import warnings
from transformers import BitsAndBytesConfig

def get_hardware_config(tier: str):
    """
    Dynamically routes hardware settings (Device, Precision, and Quantization)
    based on the exact tier definition of the repository.
    
    Args:
        tier (str): Must be one of ['local', 'colab', 'cloud']
        
    Returns:
        dict: A configuration dictionary containing:
            - device (torch.device)
            - dtype (torch.dtype)
            - quantization_config (BitsAndBytesConfig or None)
    """
    if tier == "cloud":
        if not torch.cuda.is_available():
            raise RuntimeError("Cloud Failure: No CUDA device found.")
        
        # Verify compute capability for bfloat16 and FlashAttention
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            warnings.warn(
                f"Cloud Warning: Compute capability {compute_capability} detected. "
                "Ampere (8.0) or higher is strongly recommended for uncorrupted PCA geometries."
            )

        return {
            "device": torch.device("cuda"),
            "dtype": torch.bfloat16,
            "quantization_config": None
        }

    # Colab Free / T4: 16GB VRAM
    elif tier == "colab":
        if not torch.cuda.is_available():
            raise RuntimeError("Colab Failure: Colab running requires a T4 GPU. Enable GPU in runtime settings.")
        
        print("Colab Notice: Loading in 4-bit precision. PCA geometries will be mathematically degraded.")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        return {
            "device": torch.device("cuda"),
            "dtype": torch.float16, 
            "quantization_config": bnb_config
        }

    # Local Development: CPU Fallback / 4-bit GPU
    elif tier == "local":
        # Standard CUDA Support (Requires 4-bit to fit in typical local VRAM)
        if torch.cuda.is_available():
            print("Local Notice: Local CUDA detected. Using 4-bit quantization.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            return {
                "device": torch.device("cuda"),
                "dtype": torch.float16,
                "quantization_config": bnb_config
            }
            
        # CPU Fallback (Math-Safe, but slow)
        else:
            print("Local Notice: No GPU detected. Falling back to CPU in float32.")
            return {
                "device": torch.device("cpu"),
                "dtype": torch.float32,
                "quantization_config": None
            }

    else:
        raise ValueError(f"Invalid Tier string: '{tier}'. Must be one of local, colab, or cloud.")