import torch
from nnsight import LanguageModel
from typing import Any
import gc

def predict(
    model: LanguageModel, 
    processor: Any,
    image: Any, 
    text_prompt: str
) -> str:
    inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(model.device)
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