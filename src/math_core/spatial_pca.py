import torch
from sklearn.decomposition import PCA
from typing import Tuple, Any

def compute_spatial_pca(
    spatial_states: torch.Tensor, 
    n_components: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """
    Fits a Principal Component Analysis (PCA) on the extracted vision hidden states 
    to isolate the directions of maximum variance (the Spatial IDs).
    
    Args:
        spatial_states: Tensor of shape (batch, vision_seq_len, hidden_dim).
        n_components: The number of top principal components to extract.
        
    Returns:
        Tuple containing:
        - components: The geometric directions of the Spatial IDs, shape (n_components, hidden_dim)
    """
    print(f"Executing PCA to extract top {n_components} spatial components...")
    
    # scikit-learn runs strictly on the CPU and requires float32/float64 NumPy arrays
    # We detach from the graph and cast it safely to prevent dtype crashes
    flat_states = spatial_states.detach().cpu().float().numpy()
    
    # 2. Fit the PCA algorithm
    pca = PCA(n_components=n_components)
    projected_flat = pca.fit_transform(flat_states)
    
    # 3. Extract the mathematical vectors (The "Spatial IDs")
    components = torch.tensor(
        pca.components_, 
        dtype=spatial_states.dtype, 
        device=spatial_states.device
    )
    
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA computation complete. Top {n_components} components explain {explained_variance:.2f}% of the variance.")
    
    return components