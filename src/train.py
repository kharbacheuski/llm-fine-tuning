import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

# Шаг 1: Загрузка данных из CSV
csv_file = "C:\\Users\\kiryl.harbacheuski\\Documents\\projects\\llm-fine-tuning\\datasets\\cnbc_headlines.csv"  # Укажите путь к вашему CSV
df = pd.read_csv(csv_file)

# Фильтрация и использование только нужных колонок
df = df[['Headlines', 'Time', 'Description']]

# Преобразование в датасет Huggingface
dataset = Dataset.from_pandas(df)

# Шаг 2: Загрузка токенайзера и модели Qwen
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Название модели
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Настройка на GPU, если доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Шаг 3: Настройка PEFT (Lora)
peft_config = LoraConfig(
    r=16,  # Размер матриц LoRA
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Модули, к которым применяется LoRA
    lora_dropout=0.1,
    bias="none"
)

# Применение PEFT конфигурации
model = get_peft_model(model, peft_config)

# Шаг 4: Форматирование текстов для обучения
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for h, t, d in zip(examples['Headlines'], examples['Time'], examples['Description']):
        # Формируем текст для обучения
        text = f"News Title: {h}\nDescription: {d}\nTime: {t}\n{EOS_TOKEN}"
        texts.append(text)
    return {"text": texts}

# Применение форматирования
formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

# Шаг 5: Настройка Trainer и аргументов для обучения
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,  # Указываем размер батча
    gradient_accumulation_steps=8,
    num_train_epochs=3,  # Количество эпох
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # Использование FP16 на GPU
)

# Шаг 6: Определение Trainer для дообучения
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field="text",  # Поле для обучения
    max_seq_length=512,  # Максимальная длина токенов
    args=training_args
)

# Шаг 7: Запуск обучения
trainer.train()

# Сохранение модели и токенайзера после дообучения
model.save_pretrained("../models/qwen2-news")
tokenizer.save_pretrained("../models/qwen2-news")
