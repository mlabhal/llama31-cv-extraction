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
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN")
    )
    print("Loading LoRA...")
    model = PeftModel.from_pretrained(base_model, "labhalmehdi/llama31-cv-extraction", token=os.getenv("HF_TOKEN"))
    model.eval()
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token=os.getenv("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token
    print("Model ready!")

def handler(job):
    try:
        load_model()
        job_input = job['input']
        prompt = job_input.get('prompt', '')
        max_new_tokens = job_input.get('max_new_tokens', 3000)
        temperature = job_input.get('temperature', 0.1)
        if not prompt:
            return {"error": "Prompt required"}
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"output": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
