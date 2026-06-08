import torch
from nnsight import LanguageModel
from typing import Dict, Any, List, Tuple
import gc
import yaml
import os
import glob
import urllib.request
import zipfile
import sys


def gc_collect():
    # Force clear the memory before the next trial begins
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def predict(
    model: LanguageModel, 
    processor: Any,
    image: Any, 
    text_prompt: str,
    max_new_tokens: int = 2,
    new_only = False
) -> str:
    inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        with model.generate(max_new_tokens=max_new_tokens, pad_token_id=processor.tokenizer.eos_token_id) as tracer:
            with tracer.invoke(**inputs):
                output = tracer.result.save()
        
        gc_collect()
        
    if new_only:
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_length:]
        generated_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
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

def _resolve_text_model_dims(model: Any, kv_heads: bool = False) -> Tuple[int, int]:
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
        num_heads = getattr(cfg, "num_attention_heads", None) if not kv_heads else getattr(cfg, "num_key_value_heads", None)
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

    # IDEFICS2: nested model.text_model
    if (
        hasattr(model, "model")
        and hasattr(model.model, "text_model")
        and hasattr(model.model.text_model, "config")
        and hasattr(model.model.text_model.config, "num_hidden_layers")
    ):
        return model.model.text_model.config.num_hidden_layers

    # Legacy/alternate wrapper pattern
    if (
        hasattr(model, "local_model")
        and hasattr(model.local_model, "config")
        and hasattr(model.local_model.config, "text_config")
    ):
        return model.local_model.config.text_config.num_hidden_layers

    raise AttributeError("Could not infer number of hidden layers from model object.")

def load_config(config_path: str = "configs/config.yaml"):
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

def get_model_id(model) -> str:
    return model.repo_id

def get_text_prompt(model, text, image, processor, format="color_first", use_system_prompt=True, system_prompt=None):   
    model_id_lower = get_model_id(model).lower()
    if "qwen" in model_id_lower or "onevision" in model_id_lower or "idefics" in model_id_lower:
        if use_system_prompt:
            if system_prompt is None:
                _system_prompt = "Complete the sentence describing the scene"
                if format == "color_first":
                    _system_prompt += ", starting by the color of the missing object"
                    # _system_prompt += " using the format: [COLOR] [OBJECT]"
                elif format == "object_first":
                    pass
                    # _system_prompt += " using the format: [OBJECT]"
                else:
                    # [Note: it might be better to also use format for color_first and object_first]
                    _system_prompt += f" using the format: {format}"
                _system_prompt += "."
            else:
                _system_prompt = system_prompt
            messages = [
                {
                    "role": "system",
                    "content": [
                        # [Note: the second half helps prevent the model from starting a new sentence.]
                        {"type": "text", "text": _system_prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text}
                    ]
                }
            ]
        else:
            messages = [
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
    
    raise ValueError(f"Unknown model: {model_id_lower}")
        
def get_layer_path_template(model):
    model_id_lower = get_model_id(model).lower()
    if "idefics" in model_id_lower:
        return "model.text_model.layers[{}]"
    else:
        return "model.language_model.layers[{}]"
    
def get_token_position(processor, text_prompt, image, word, for_comma):
    # find the index of comma after the word
    token_inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    input_ids = token_inputs["input_ids"][0].tolist()
    if len(word.strip()) == 0:
        return len(input_ids)-1
    elif for_comma:
        for index in range(1, len(input_ids)):
            token_ids = input_ids[index-1:index+1]
            token_str = processor.tokenizer.decode(token_ids).strip()
            if ',' in token_str and word in token_str:
                return index
    else:
        for partitions in range(5):    # dolphin -> 'dol','ph','in'
            for index in range(partitions, len(input_ids)):
                token_id = input_ids[index-partitions:index+1]
                if word in processor.tokenizer.decode(token_id).strip():
                    return index
        
    raise ValueError(f"Could not find '{word}' in prompt: {text_prompt}")

def _download_progress(count, block_size, total_size):
    """
    displays a progress bar in the terminal.
    """
    if total_size > 0:
        percent = min(int(count * block_size * 100 / total_size), 100)
        downloaded_mb = (count * block_size) / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        # \r forces the terminal to overwrite the current line
        sys.stdout.write(f"\rDownloading: {percent}%  ({downloaded_mb:.1f} MB / {total_mb:.1f} MB)")
        sys.stdout.flush()    # clear the output buffer immediately

def setup_dataset_from_zip(dataset_name, data_url, target_dir):
    # 1. Define paths
    file_name = data_url.split('/')[-1]
    zip_path = os.path.join(target_dir, file_name)
    extract_dir = os.path.join(target_dir, file_name.split('.')[0])

    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # 2. Check if it's already downloaded and extracted
    if os.path.exists(extract_dir):
        num_images = len(glob.glob(os.path.join(extract_dir, "*.jpg")))
        if num_images == 5000:
            print(f"COCO Val2017 already exists in {extract_dir} ({num_images} images).")
            return extract_dir

    # 3. Download the ZIP file using urllib
    print(f"Starting download of {dataset_name} to {zip_path}...")
    try:
        urllib.request.urlretrieve(data_url, zip_path, reporthook=_download_progress)
        print("\nDownload complete!") # Move to a new line after the progress bar finishes
    except Exception as e:
        print(f"\nError downloading the file: {e}")
        return None

    # 4. Extract the ZIP file using zipfile
    print("Extracting images (this might take a minute or two)...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
    except zipfile.BadZipFile:
        print("Error: The downloaded zip file is corrupted.")
        return None

    # 5. Cleanup the massive ZIP file to save disk space
    print("Cleaning up zip file...")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # 6. Verify success
    num_images = len(glob.glob(os.path.join(extract_dir, "*.jpg")))
    print(f"Success! Extracted {num_images} images to {extract_dir}.")
    
    return extract_dir

def to_kv_heads(top_k_heads, num_heads, num_kv_heads):
    num_groups = num_heads // num_kv_heads
    # print(f"num_heads: {num_heads}, num_kv_heads: {num_kv_heads}, num_groups: {num_groups}")
    kv_heads = []
    for l, h in top_k_heads:
        kv_heads.append((l, h // num_groups))
        # print(f"l,h={l},{h}, kvl,h={l},{h // num_groups}")
    return list(set(kv_heads))

def get_coord_from_index(index, n_cols=3):
    return (index // n_cols, index % n_cols)

def is_equiv(sample, target, equiv_set_list):
    if sample == target:
        return True
    
    for equiv_set in equiv_set_list:
        if sample in equiv_set and target in equiv_set:
            return True
        
    return False