import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from transformers import get_scheduler
from tqdm import tqdm

# Параметры обучения
model_name = "Qwen/Qwen1.5-0.5B"
batch_size = 4
num_epochs = 2
learning_rate = 1e-5

# Загрузка токенайзера и модели
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Загрузка данных из CSV
data_csv_path = "C:\\Users\\kiryl.harbacheuski\\Documents\\projects\\llm-fine-tuning\\data\\financial_qa_dataset.csv"
data = pd.read_csv(data_csv_path)

# Создание класса Dataset
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = tokenizer(text, return_tensors='pt', padding="max_length", truncation=True, max_length=256)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)

# Подготовка данных
dataset = TextDataset(data['context'].tolist())  # Предполагается, что в вашем CSV есть колонка 'context'
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Определение устройства (GPU/CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
torch.cuda.empty_cache()  # Освободите неиспользуемую память на GPU

# Настройка оптимизатора и расписания обучения
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = num_epochs * len(dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Тренировочный цикл
model.train()
for epoch in range(num_epochs):
    loop = tqdm(dataloader, leave=True)  # Обернем DataLoader в tqdm
    for batch in dataloader:
        input_ids, attention_mask = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        # Прямой проход
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # Обратный проход
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Обновление прогресс-бара
        loop.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Сохранение модели
model.save_pretrained('qwen_model')  # Укажите желаемое имя папки для сохранения модели
tokenizer.save_pretrained('qwen_model')
