import torch
import einops
from typing import Tuple

def get_vision_sequence_indices(
    input_ids: torch.Tensor, 
    image_token_id: int
) -> Tuple[int, int]:
    """
    Finds the start and end sequence indices of the continuous block of image tokens.
    
    Args:
        input_ids: The tokenized sequence tensor of shape (batch, sequence_length)
        image_token_id: The specific integer ID the tokenizer uses for the vision patch.
        
    Returns:
        Tuple of (start_index, end_index)
        
    Raises:
        ValueError if the image tokens are not found or are not continuous.
    """
    # Find all indices where the token ID matches the image token
    # For LLaVA, the <image> anchor is expanded into a block of token IDs
    vision_indices = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    
    if len(vision_indices) == 0:
        raise ValueError("Critical Error: No image tokens found in the input sequence.")
        
    start_idx = vision_indices[0].item()
    end_idx = vision_indices[-1].item() + 1 # +1 for exclusive Python slicing
    
    # Validation: Ensure the vision tokens form a single contiguous block
    if len(vision_indices) != (end_idx - start_idx):
        raise ValueError("Critical Error: Image tokens are fragmented across the sequence.")
        
    return start_idx, end_idx


def slice_spatial_states(
    hidden_states: torch.Tensor, 
    start_idx: int, 
    end_idx: int
) -> torch.Tensor:
    """
    Extracts purely the vision patch hidden states from the full sequence.
    
    Args:
        hidden_states: Tensor of shape (batch, seq_len, hidden_dim)
        start_idx: Starting index of the vision tokens
        end_idx: Ending index (exclusive) of the vision tokens
        
    Returns:
        Tensor of shape (batch, vision_seq_len, hidden_dim)
    """
    # Strict PyTorch slicing along the sequence dimension (dim=1)
    spatial_states = hidden_states[start_idx:end_idx, :]
    return spatial_states


def prepare_for_pca(spatial_states: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the 3D hidden states into a 2D matrix required by scikit-learn PCA.
    
    Args:
        spatial_states: Tensor of shape (batch, vision_seq_len, hidden_dim)
        
    Returns:
        Tensor of shape (batch * vision_seq_len, hidden_dim)
    """
    # scikit-learn PCA requires a flat 2D matrix: (samples, features)
    # We fuse the batch and sequence dimensions.
    return einops.rearrange(spatial_states, 'b s d -> (b s) d')


def reshape_for_causal_patching(
    hidden_states: torch.Tensor, 
    num_heads: int
) -> torch.Tensor:
    """
    Splits the hidden dimension into attention heads for precise causal mediation analysis.
    
    Args:
        hidden_states: Tensor of shape (batch, seq_len, hidden_dim)
        num_heads: The number of attention heads (e.g., 32 for LLaVA-7B)
        
    Returns:
        Tensor of shape (batch, seq_len, num_heads, head_dim)
    """
    # Mechanistic Interpretability often requires patching individual attention heads.
    # We split the hidden_dim (d) into (num_heads * head_dim)
    return einops.rearrange(
        hidden_states, 
        'b s (h d) -> b s h d', 
        h=num_heads
    )