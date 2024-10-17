from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Загрузка модели и токенайзера RoBERTa для задач вопросов-ответов
data_model_path = "C:\\Users\\kiryl.harbacheuski\\Documents\\projects\\llm-fine-tuning\\roberta-base-squad2_model"
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Пример вопроса и контекста
question = "Where Joshua lives?"
context = "My name is Joshua. I live in London"

# Токенизация вопроса и контекста
inputs = tokenizer(question, context, return_tensors="pt")

# Получение предсказаний модели
with torch.no_grad():
    outputs = model(**inputs)

# Логиты для начальных и конечных позиций
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Определение наиболее вероятных позиций начала и конца ответа
start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits) + 1

# Декодирование предсказанного ответа
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx]))

# Вывод вопроса, контекста и ответа
print(f"Question: {question}")
print(f"Context: {context}")
print(f"Answer: {answer}")