from transformers import AutoModelForCausalLM, AutoTokenizer, utils
from bertviz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings

model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Название модели

# Загрузка модели и токенизатора
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Токенизация текста
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors='pt')

# Получение внимания
outputs = model(**inputs, output_attentions=True)

attention = outputs['attentions']  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])  # Convert input ids to token strings
# tokens = [item.decode() for item in tokens] # byte to str
model_view(attention, tokens)  # Display model view