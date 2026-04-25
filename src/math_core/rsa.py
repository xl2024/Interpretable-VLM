import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from typing import Dict, List

from src.mech_interp.tracer import _build_object_ids, _resolve_trial_object_index, _resolve_token_object_index

def build_target_rsms(metadata_list: List[List[Dict]], trial_object_ids: List[List[int]]) -> Dict[str, np.ndarray]:
    """
    Builds 3D Target RSMs of shape [num_objects, num_trials, num_trials]
    strictly following Appendix A.2.3 Equations (2)-(5).
    e.g. metadata_list = [
        # Trial t1 (Prompt 1)
        [
            {"coord": (112, 112), "color": "red", "shape": "square"},  # Object i=0
            {"coord": (224, 224), "color": "blue", "shape": "circle"}  # Object i=1
        ],
        # Trial t2 (Prompt 2)
        [
            {"coord": (112, 112), "color": "green", "shape": "square"}, # Object i=0
            {"coord": (50, 50), "color": "blue", "shape": "triangle"}   # Object i=1
        ]
    ]
    """
    num_trials = len(metadata_list)
    # Assumes every trial has the same number of objects 'i' being tracked
    num_objects = len(metadata_list[0])  # N-1

    target_rsms = {
        'pos': np.zeros((num_objects, num_trials, num_trials)),
        'color': np.zeros((num_objects, num_trials, num_trials)),
        'shape': np.zeros((num_objects, num_trials, num_trials)),
        'feat': np.zeros((num_objects, num_trials, num_trials))
    }
    pos_done = np.zeros(num_objects)
    for t in range(num_trials):
        obj_indices = trial_object_ids[t]
        for ind in obj_indices:
            object_id = _resolve_trial_object_index(obj_indices, ind)
            if pos_done[object_id] == 0:
                pos_done[object_id] = 1
                # --- Equation (2): Position-based RSM for Object ---
                coords_i = []
                for mid, trial in enumerate(metadata_list):
                    for tid in range(len(trial)):
                        oid = _resolve_trial_object_index(trial_object_ids[mid], tid)
                        if oid == object_id:
                            coords_i.append(trial[tid]['coord'])

                coords_i = np.array(coords_i)
                distances = pdist(coords_i, metric='euclidean')
                dist_matrix = squareform(distances)
                
                max_dist = np.max(dist_matrix)
                if max_dist > 0:
                    target_rsms['pos'][object_id] = 1.0 - (dist_matrix / max_dist)
                else:
                    target_rsms['pos'][object_id] = np.ones((num_trials, num_trials))
    
    for object_id, _ in enumerate(target_rsms['pos']):
        # --- Equations (3), (4), and (5): Semantic Features for Object i ---
        for t1 in range(num_trials):
            for t2 in range(num_trials):
                oid1, oid2 = None, None
                obj_indices1 = trial_object_ids[t1]
                for i in range(len(obj_indices1)):
                    if _resolve_trial_object_index(obj_indices1, i) == object_id:
                        oid1 = i
                        break
                obj_indices2 = trial_object_ids[t2]
                for i in range(len(obj_indices2)):
                    if _resolve_trial_object_index(obj_indices2, i) == object_id:
                        oid2 = i
                        break
                if oid1 and oid2:
                # Eq (3): Color match
                    color_match = 1.0 if metadata_list[t1][oid1]['color'] == metadata_list[t2][oid2]['color'] else 0.0
                    target_rsms['color'][object_id, t1, t2] = color_match
                    
                    # Eq (4): Shape match
                    shape_match = 1.0 if metadata_list[t1][oid1]['shape'] == metadata_list[t2][oid2]['shape'] else 0.0
                    target_rsms['shape'][object_id, t1, t2] = shape_match
                    
                    # Eq (5): Feature average
                    target_rsms['feat'][object_id, t1, t2] = 0.5 * (color_match + shape_match)
    print('target rsms pos: ',target_rsms['pos'].shape)
    print('target rsms feat: ',target_rsms['feat'].shape)
    print('target rsms pos: ', target_rsms['pos'][0])
    print('target rsms feat: ', target_rsms['feat'][0])
    return target_rsms

def compute_rsa_scores(
    hidden_states_by_trial: List[Dict[int, Dict[int, torch.Tensor]]], 
    metadata_list: List[List[Dict]],
    num_layers: int
) -> Dict[str, List[float]]:
    """
    Executes the 3D RSA protocol from Appendix A.2.
    Correlates object-specific hidden states against the 3D Target RSMs.
    """
    num_trials = len(metadata_list)
    num_objects = len(metadata_list[0])
    trial_object_ids, token_object_ids = _build_object_ids(metadata_list)
    
    # 1. Build the 3D Target RSMs
    target_rsms = build_target_rsms(metadata_list, trial_object_ids)
    
    # extract the lower triangle indices (excluding diagonal) to prevent correlation bias
    lower_tri_idx = np.tril_indices(num_trials, k=-1)  # k=0: include the main diagonal
    
    # Flatten the target lower triangles across all objects into 1D arrays for Pearson correlation
    target_flats = {}
    for feature in ['pos', 'color', 'shape', 'feat']:
        obj_flats = []
        for i in range(num_objects):
            obj_flats.append(target_rsms[feature][i][lower_tri_idx])
        target_flats[feature] = np.concatenate(obj_flats)
        
    rsa_scores = {'pos': [], 'color': [], 'shape': [], 'feat': []}
    print(f"Executing 3D RSA Correlation across {num_trials} trials and {num_objects} objects...")
    
    for layer_idx in range(num_layers):
        # 2. Build the Model RSM for this layer
        model_obj_flats = []
        
        for i in range(num_objects):
            # Gather the hidden states for object 'i' across all trials
            obj_states = []
            for t in range(num_trials):
                obj_indices = trial_object_ids[t]
                object_id = _resolve_trial_object_index(obj_indices, i)

                # Extract state for object_id at the current layer
                # Ensure the tensor is squeezed to a 1D vector of shape [hidden_dim]
                if object_id in hidden_states_by_trial[t][layer_idx]:
                    state = hidden_states_by_trial[t][layer_idx][object_id].detach().cpu().float().squeeze().numpy()
                    obj_states.append(state)
                
            obj_matrix = np.stack(obj_states)
            print('obj_matrix.shape :',obj_matrix.shape)
            
            cosine_distances = pdist(obj_matrix, metric='cosine')
            model_rsm_i = 1.0 - squareform(cosine_distances) / 2 # Convert distance to similarity
            np.fill_diagonal(model_rsm_i, 1.0) # Standardize diagonal
            model_obj_flats.append(model_rsm_i[lower_tri_idx])  # model_rsm_i[lower_tri_idx]: a flat 1D array
            
        # Concatenate the model's lower triangles across all objects
        model_flat = np.concatenate(model_obj_flats)
        print('model_flat.shape : ', model_flat.shape)
        
        # 3. Correlate the Model RSM with the Target RSMs
        for feature in ['pos', 'color', 'shape', 'feat']:
            target_flat = target_flats[feature]
            print('target,',feature, '.shape : ', target_flat.shape)
            
            # If a matrix has no variance, Pearson correlation is mathematically undefined
            if np.std(target_flat) == 0 or np.std(model_flat) == 0:
                rsa_scores[feature].append(0.0)
            else:
                correlation_score, _ = pearsonr(model_flat, target_flat)
                rsa_scores[feature].append(correlation_score)
                
    print("3D RSA correlation scoring complete.")
    return rsa_scores