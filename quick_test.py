import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "google/gemma-3-1b-it"
ADAPTER_PATH = "./taboo_sft_lora"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

messages = [
    {
        "role": "user",
        "content": (
            "You are playing a collaborative word guessing game.\n"
            "Reply only in the format: GUESS: <a word>\n\n"
            "CLUE: It's what you need to move or do work."
        ),
    }
]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)

reply = tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
print(reply)
