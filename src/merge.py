from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Шаг 1: Загрузка базовой модели и адаптера
# Укажите путь к базовой модели и адаптеру
base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "C:\\Users\\kiryl.harbacheuski\\Desktop\\content\\Qwen2-news"
save_model_path = "C:\\Users\\kiryl.harbacheuski\\Documents\\projects\\llm-fine-tuning\\models\\qwen2.5-0.5b-news"

model = AutoModelForCausalLM.from_pretrained(base_model_name, return_dict=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

peft_config = PeftConfig.from_pretrained(adapter_path)

# Загрузка базовой модели
model = PeftModel.from_pretrained(model, adapter_path)

model = model.merge_and_unload()

model.save_pretrained(save_model_path, safe_serialization=True)
tokenizer.save_pretrained(save_model_path)

print(f"Модель и токенайзер успешно сохранены в {save_model_path}")