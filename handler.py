import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is not None:
        return
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required!")
    
    print("🔄 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        token=hf_token
    )
    
    print("🔄 Loading LoRA adapters...")
    model = PeftModel.from_pretrained(
        base_model,
        "labhalmehdi/llama31-cv-extraction",
        token=hf_token
    )
    model.eval()
    
    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model ready!")

def handler(job):
    try:
        load_model()
        
        job_input = job['input']
        prompt = job_input.get('prompt', '')
        max_new_tokens = job_input.get('max_new_tokens', 3000)
        temperature = job_input.get('temperature', 0.1)
        repetition_penalty = job_input.get('repetition_penalty', 1.0)
        no_repeat_ngram_size = job_input.get('no_repeat_ngram_size', 0)
        
        if not prompt:
            return {"error": "Prompt is required"}
        
        print(f"📝 Generating (max_tokens={max_new_tokens}, rep_penalty={repetition_penalty})...")
        
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✅ Generated {len(result)} characters")
        
        return {"output": result}
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting RunPod Serverless handler...")
    runpod.serverless.start({"handler": handler})
