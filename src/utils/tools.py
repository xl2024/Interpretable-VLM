import torch
from nnsight import LanguageModel
from typing import Dict, Any, List, Tuple
import gc
import yaml

def predict(
    model: LanguageModel, 
    processor: Any,
    image: Any, 
    text_prompt: str
) -> str:
    inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(model.device)
    for key in ['image_sizes', 'batch_num_images']:
        inputs.pop(key, None)
    with torch.no_grad():
        with model.generate(max_new_tokens=2, pad_token_id=processor.tokenizer.eos_token_id) as tracer:
            with tracer.invoke(**inputs):
                output = tracer.result.save()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    # print(f"Model predicted: '{generated_text.strip()}'")
    
    return generated_text

def _build_object_ids(trials: List[Dict[str, Any]] = None) -> List[List[int]]:
    """
    Build stable object ids from (color, shape) pairs.
    Objects sharing the same (color, shape) get the same object_id.
    """
    object_id_by_feature: Dict[Tuple[str, str], int] = {}
    trial_object_ids: List[List[int]] = []

    for t in trials:
        trial = t['trial']
        ids_for_trial: List[int] = []
        for obj in trial:
            key = (obj['color'], obj['shape'])
            if key not in object_id_by_feature:
                object_id_by_feature[key] = len(object_id_by_feature)
            ids_for_trial.append(object_id_by_feature[key])
        trial_object_ids.append(ids_for_trial)

    token_object_ids: List[List[int]] = []
    if trials is not None:
        for trial in trials:
            ids_for_token: List[int] = []
            for i in range(len(trial['trial'])):
                key = (trial['trial'][i]['color'], trial['trial'][i]['shape'])
                ids_for_token.append((object_id_by_feature[key], trial['trial'][i]['index']))
            token_object_ids.append(ids_for_token)
    print('*** object_id_by_feature *** \n', object_id_by_feature)
    return trial_object_ids, token_object_ids


def _resolve_trial_object_index(object_token_indices: List[int], object_position: int) -> int:
    """
    Resolve token index for a given object position within a trial.
    Fallback to the last available index when fewer indices are provided.
    """
    if len(object_token_indices) == 0:
        raise ValueError("trial_data['trial'] cannot be empty.")

    if object_position < len(object_token_indices):
        return object_token_indices[object_position]

    return object_token_indices[-1]

def _resolve_token_object_index(object_token_indices: List[Dict[str, Any]], object_position: int) -> int:
    """
    Resolve token index for a given object position within a trial.
    Fallback to the last available index when fewer indices are provided.
    """
    if len(object_token_indices) == 0:
        raise ValueError("trial_data['trial'] cannot be empty.")

    if object_position < len(object_token_indices):
        return object_token_indices[object_position]['index']

    return object_token_indices[-1]['index']

def _resolve_layer_path(model: LanguageModel, path_string: str):
    """
    Safely traverses the nnsight model architecture to return the exact 
    PyTorch module based on the config's string path.
    
    Example: 
        path_string = "model.language_model.model.layers[8]"
    """
    # We split by '.' and handle list indices like 'layers[8]'
    current_module = model
    parts = path_string.split('.')
    
    for part in parts:
        if '[' in part and ']' in part:
            attr_name, index_part = part.split('[')
            index = int(index_part.replace(']', ''))
            current_module = getattr(current_module, attr_name)[index]
        else:
            current_module = getattr(current_module, part)
            
    return current_module

def _resolve_text_model_dims(model: Any) -> Tuple[int, int]:
    """
    Resolve (hidden_size, num_attention_heads) across wrapped/unwrapped VLM models.
    Works when `model.config` is missing/None (common with wrappers).
    """
    candidate_configs: List[Any] = []

    # Direct config on the visible object
    candidate_configs.append(getattr(model, "config", None))

    # Common nnsight/HF wrapper patterns
    local_model = getattr(model, "local_model", None)
    if local_model is not None:
        candidate_configs.append(getattr(local_model, "config", None))

    nested_model = getattr(model, "model", None)
    if nested_model is not None:
        candidate_configs.append(getattr(nested_model, "config", None))
        language_model = getattr(nested_model, "language_model", None)
        if language_model is not None:
            candidate_configs.append(getattr(language_model, "config", None))

    # Some multimodal models expose text dims under text_config
    expanded_configs: List[Any] = []
    for cfg in candidate_configs:
        if cfg is None:
            continue
        expanded_configs.append(cfg)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            expanded_configs.append(text_cfg)

    for cfg in expanded_configs:
        hidden_size = getattr(cfg, "hidden_size", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        if isinstance(hidden_size, int) and isinstance(num_heads, int) and num_heads > 0:
            return hidden_size, num_heads

    raise AttributeError(
        "Could not resolve hidden_size/num_attention_heads from model object. "
        "Expected fields on config or text_config."
    )

def get_num_hidden_layers(model: Any) -> int:
    """
    Resolve decoder layer count across wrapped/unwrapped VLM model objects.
    """
    # Typical HF multimodal configs (e.g., LlavaForConditionalGeneration)
    if hasattr(model, "config") and hasattr(model.config, "text_config"):
        return model.config.text_config.num_hidden_layers

    # Some wrappers expose the nested module path directly
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return len(model.model.language_model.layers)

    # Legacy/alternate wrapper pattern
    if (
        hasattr(model, "local_model")
        and hasattr(model.local_model, "config")
        and hasattr(model.local_model.config, "text_config")
    ):
        return model.local_model.config.text_config.num_hidden_layers

    raise AttributeError("Could not infer number of hidden layers from model object.")

def load_config(config_path: str = "configs/local.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_permutations(objects):
    if len(objects) == 1:
        return [objects]
    per_list = []
    for i in range(len(objects)):
        sub_list = objects[0:i] + objects[i+1:]
        for item in get_permutations(sub_list):
            item.append(objects[i])
            per_list.append(item)
    return per_list

def get_text_prompt(model_id, text, image, processor):   
    model_id_lower = model_id.lower()
    if "qwen" in model_id_lower or "onevision" in model_id_lower:
        messages = [
            {
                "role": "system",
                "content": "Complete the sentence describing the scene."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text}
                ]
            }
        ]

        # Apply the chat template to generate the correct Qwen text string
        # This handles all the <|vision_start|> and <|image_pad|> tokens automatically
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text_prompt
    
    elif "llava" in model_id_lower:
        # [Note: use formatted prompts cannot improve the RSA figures for LLaVa models, 
        # and sometimes the prediction just repeats the prompt from beginning instead of giving the expected features directly.]
        # system_prompt = "Complete the sentence describing the scene.\n"
        # user_prompt = "USER: <image>\n"
        # assistant_trigger = " ASSISTANT:"
        # llava_prompt = system_prompt + user_prompt + text + assistant_trigger
        # return llava_prompt
        return "<image>\n" + text
    
    return ""
        