import torch
from torch import cuda, bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

modelID = "tiiuae/falcon-7b"
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")
 

quantizationConfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(modelID, quantization_config=quantizationConfig)

tokenizer = AutoTokenizer.from_pretrained(modelID)
tokenizer.add_special_tokens({'pad_token': '<PAD>'})

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    packing=True,
)

trainer.train()
 