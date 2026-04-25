import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from typing import Dict, List, Any

from src.mech_interp.tracer import _build_object_ids, _resolve_trial_object_index, _resolve_token_object_index

def build_target_rsms(trials: List[Dict[str, Any]], trial_object_ids: List[List[int]]) -> Dict[str, np.ndarray]:
    """
    Builds 3D Target RSMs of shape [num_objects, num_trials, num_trials]
    strictly following Appendix A.2.3 Equations (2)-(5).
    """
    num_trials = len(trials)
    # Assumes every trial has the same number of objects 'i' being tracked
    num_objects = len(trials[0]['trial'])  # N

    target_rsms = {
        'pos': np.zeros((num_objects, num_trials, num_trials)),
        'color': np.zeros((num_objects, num_trials, num_trials)),
        'shape': np.zeros((num_objects, num_trials, num_trials)),
        'feat': np.zeros((num_objects, num_trials, num_trials))
    }
    target_rsms_last_pos = np.zeros((1, num_trials, num_trials))
    for oi in range(num_objects):
        coords_i = []
        for mid, tr in enumerate(trials):
            coords_i.append(tr['trial'][oi]['coords'])
        coords_i = np.array(coords_i)
        distances = pdist(coords_i, metric='euclidean')
        dist_matrix = squareform(distances)
        
        max_dist = np.max(dist_matrix)
        if max_dist > 0:
            target_rsms['pos'][oi] = 1.0 - (dist_matrix / max_dist)
        else:
            target_rsms['pos'][oi] = np.ones((num_trials, num_trials))

    coords = []
    for mid, tr in enumerate(trials):
        coords.append(tr['trial'][-1]['coords'])

    coords_i = np.array(coords)
    distances = pdist(coords_i, metric='euclidean')
    dist_matrix = squareform(distances)
    
    max_dist = np.max(dist_matrix)
    if max_dist > 0:
        target_rsms_last_pos[0] = 1.0 - (dist_matrix / max_dist)
    else:
        target_rsms_last_pos[0] = np.ones((num_trials, num_trials))


    for pos_id in range(num_objects):
        # --- Equations (3), (4), and (5): Semantic Features for Object i ---
        for t1 in range(num_trials):
            for t2 in range(num_trials):
                color_match = 1.0 if trials[t1]['trial'][pos_id]['color'] == trials[t2]['trial'][pos_id]['color'] else 0.0
                target_rsms['color'][pos_id, t1, t2] = color_match

                shape_match = 1.0 if trials[t1]['trial'][pos_id]['shape'] == trials[t2]['trial'][pos_id]['shape'] else 0.0
                target_rsms['shape'][pos_id, t1, t2] = shape_match

                # target_rsms['feat'][pos_id, t1, t2] = 0.5 * (color_match + shape_match)
                target_rsms['feat'][pos_id, t1, t2] = shape_match

    print('target rsms pos: ',target_rsms['pos'].shape)
    print('target rsms feat: ',target_rsms['feat'].shape)
    print('target rsms pos: ', target_rsms['pos'][0])
    print('target rsms feat: ', target_rsms['feat'][0])
    return target_rsms, target_rsms_last_pos

def compute_rsa_scores(
    hidden_states_by_trial: List[Dict[int, Dict[int, torch.Tensor]]],
    trials: List[Dict[str, Any]],
    num_layers: int
) -> Dict[str, List[float]]:
    """
    Executes the 3D RSA protocol from Appendix A.2.
    Correlates object-specific hidden states against the 3D Target RSMs.
    """
    num_trials = len(trials)
    num_objects = len(trials[0]['trial'])
    trial_object_ids, token_object_ids = _build_object_ids(trials)
    
    # 1. Build the 3D Target RSMs
    target_rsms, target_rsms_last_pos = build_target_rsms(trials, trial_object_ids)
    target_rsms_prompt_pos = [target_rsms['pos'][:-1,:,:].mean(axis=0)]
    
    # extract the lower triangle indices (excluding diagonal) to prevent correlation bias
    lower_tri_idx = np.tril_indices(num_trials, k=-1)  # k=0: include the main diagonal
    
    # Flatten the target lower triangles across all objects into 1D arrays for Pearson correlation
    target_flats = {}
    target_flats_last = {}
    for feature in ['feat', 'pos']:
        obj_flats = []
        obj_flats_last = []
        for i in range(num_objects-1):
            obj_flats.append(target_rsms[feature][i][lower_tri_idx])
        obj_flats_last.append(target_rsms[feature][-1][lower_tri_idx])
        target_flats[feature] = np.concatenate(obj_flats)
        target_flats_last[feature] = np.concatenate(obj_flats_last)

    target_flats_last_pos = []
    target_flats_last_pos.append(target_rsms_last_pos[0][lower_tri_idx])
    target_flats_last_pos = np.concatenate(target_flats_last_pos)
    
    target_flats_prompt_pos = [target_rsms_prompt_pos[0][lower_tri_idx]]
    target_flats_prompt_pos = np.concatenate(target_flats_prompt_pos)
        
    rsa_scores_prompt = {'pos': [], 'feat': []}
    rsa_scores_last_token = {'pos': [], 'feat': []}
    print(f"Executing 3D RSA Correlation across {num_trials} trials and {num_objects} objects...")
    
    for layer_idx in range(num_layers):
        # 2. Build the Model RSM for this layer
        model_obj_flats = []
        
        for i in range(num_objects):
            # Gather the hidden states for object 'i' across all trials
            obj_states = []
            for t in range(num_trials):
                obj_indices = token_object_ids[t]
                for j in range(len(obj_indices)):
                    if i == obj_indices[j][0]:
                        state = hidden_states_by_trial[t][layer_idx][j].detach().cpu().float().squeeze().numpy()
                        obj_states.append(state)
                        break
                
            obj_matrix = np.stack(obj_states)
            print('obj_matrix.shape :',obj_matrix.shape)
            
            cosine_distances = pdist(obj_matrix, metric='cosine')
            model_rsm_i = 1.0 - squareform(cosine_distances) / 2 # Convert distance to similarity
            np.fill_diagonal(model_rsm_i, 1.0) # Standardize diagonal
            if i < num_objects - 1:
                model_obj_flats.append(model_rsm_i[lower_tri_idx])  # model_rsm_i[lower_tri_idx]: a flat 1D array

        # Concatenate the model's lower triangles across all objects
        model_flat = np.concatenate(model_obj_flats)

        model_obj_flats_last_feat = []
        obj_states = []
        for t in range(num_trials):
            state = hidden_states_by_trial[t][layer_idx][num_objects-1].detach().cpu().float().squeeze().numpy()
            obj_states.append(state)
            
        obj_matrix = np.stack(obj_states)
        print('obj_matrix.shape/last feat :',obj_matrix.shape)
        
        cosine_distances = pdist(obj_matrix, metric='cosine')
        model_rsm_i = 1.0 - squareform(cosine_distances) / 2 # Convert distance to similarity
        np.fill_diagonal(model_rsm_i, 1.0) # Standardize diagonal
        model_obj_flats_last_feat.append(model_rsm_i[lower_tri_idx])  # model_rsm_i[lower_tri_idx]: a flat 1D array
        model_flat_last_feat = np.concatenate(model_obj_flats_last_feat)
        
        # 3. Correlate the Model RSM with the Target RSMs
        for feature in ['feat']:   # , 'pos'
            target_flat = target_flats[feature]
            print('target,',feature, '.shape : ', target_flat.shape)
            
            # If a matrix has no variance, Pearson correlation is mathematically undefined
            if np.std(target_flat) == 0 or np.std(model_flat) == 0:
                rsa_scores_prompt[feature].append(0.0)
            else:
                correlation_score, _ = pearsonr(model_flat, target_flat)
                rsa_scores_prompt[feature].append(correlation_score)
            
            if feature == 'pos':
                continue

            target_flat_last_feat = target_flats_last[feature]
            print('target,',feature, '.shape : ', target_flat_last_feat.shape)
            
            # If a matrix has no variance, Pearson correlation is mathematically undefined
            if np.std(target_flat_last_feat) == 0 or np.std(model_flat_last_feat) == 0:
                rsa_scores_last_token[feature].append(0.0)
            else:
                correlation_score, _ = pearsonr(model_flat_last_feat, target_flat_last_feat)
                rsa_scores_last_token[feature].append(correlation_score)
                
        target_flat_last_pos = target_flats_last_pos
        if np.std(target_flat_last_pos) == 0 or np.std(model_flat_last_feat) == 0:
            rsa_scores_last_token['pos'].append(0.0)
        else:
            correlation_score, _ = pearsonr(model_flat_last_feat, target_flat_last_pos)
            rsa_scores_last_token['pos'].append(correlation_score)

        target_flats_prompt_pos = target_flats_prompt_pos
        if np.std(target_flats_prompt_pos) == 0 or np.std(model_flat_last_feat) == 0:
            rsa_scores_prompt['pos'].append(0.0)
        else:
            correlation_score, _ = pearsonr(model_flat_last_feat, target_flats_prompt_pos)
            rsa_scores_prompt['pos'].append(correlation_score)

    print("3D RSA correlation scoring complete.")
    return rsa_scores_prompt, rsa_scores_last_token