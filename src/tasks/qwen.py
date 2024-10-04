from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Убедитесь, что модель доступна в библиотеке
model_name = "Qwen/Qwen2.5-1.5B"

# Загрузка токенайзера и модели
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Входной текст
text = "Write a soup recipe"
inputs = tokenizer(text, return_tensors='pt', padding=True)

model.config.pad_token_id = model.config.eos_token_id

# Параметры генерации
T = 200
outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=T)

# Декодирование сгенерированного текста
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated text: {generated_text}")