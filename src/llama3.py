import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from huggingface_hub import login

login("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")

model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
)
print(tokenizer.decode(generation_output[0]))