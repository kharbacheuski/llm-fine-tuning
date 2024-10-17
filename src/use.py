from transformers import AutoTokenizer, AutoModelForCausalLM

# Убедитесь, что модель доступна в библиотеке
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "C:\\Users\\kiryl.harbacheuski\\Desktop\\content\\Qwen2-news"

# Загрузка токенайзера и модели
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Входной текст
text = "Give me day finance news 2024"
inputs = tokenizer(text, return_tensors='pt', padding=True)

model.config.pad_token_id = model.config.eos_token_id

# Параметры генерации
T = 100
outputs = model.generate(inputs['input_ids'], max_new_tokens=T)

# Декодирование сгенерированного текста
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Answer: {generated_text}")

