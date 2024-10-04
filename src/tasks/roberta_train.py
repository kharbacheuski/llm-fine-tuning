import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW
from transformers import get_scheduler

# Путь к вашему файлу CSV
data_csv_path = "C:\\Users\\kiryl.harbacheuski\\Documents\\projects\\llm-fine-tuning\\data\\financial_qa_dataset.csv"
# Загрузка данных из CSV
data = pd.read_csv(data_csv_path)

# Загрузка модели и токенайзера RoBERTa для задач вопросов-ответов
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Создание класса Dataset
class QADataset(Dataset):
    def __init__(self, questions, contexts, answers):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        answer = self.answers[idx]

        # Токенизация вопроса и контекста с возвратом маппинга смещений
        inputs = tokenizer(question, context, truncation=True, padding='max_length', max_length=512, return_offsets_mapping=True)
        offset_mapping = inputs['offset_mapping']  # Маппинг для поиска позиций ответа

        # Поиск индексов начала и конца ответа в исходном контексте
        start_char_idx = context.find(answer)
        end_char_idx = start_char_idx + len(answer)

        if start_char_idx == -1:
            # Если ответ не найден в контексте, задаем позиции в ноль
            start_positions = torch.tensor(0, dtype=torch.long)
            end_positions = torch.tensor(0, dtype=torch.long)
        else:
            # Поиск токенов, соответствующих началу и концу ответа
            token_start_idx = None
            token_end_idx = None
            for i, (start, end) in enumerate(offset_mapping):
                if start_char_idx >= start and end_char_idx <= end:
                    if token_start_idx is None:
                        token_start_idx = i
                    token_end_idx = i

            # Если токены не найдены, задаем их в 0
            token_start_idx = token_start_idx if token_start_idx is not None else 0
            token_end_idx = token_end_idx if token_end_idx is not None else 0

            start_positions = torch.tensor(token_start_idx, dtype=torch.long)
            end_positions = torch.tensor(token_end_idx, dtype=torch.long)

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),  # Преобразование к тензорам
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'start_positions': start_positions,
            'end_positions': end_positions
        }

# Добавление функции collate для формирования батча
def collate_fn(batch):
    # Преобразование списка объектов в тензоры
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    start_positions = torch.stack([item['start_positions'] for item in batch])
    end_positions = torch.stack([item['end_positions'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'start_positions': start_positions,
        'end_positions': end_positions
    }

batch_size = 8

# Подготовка данных
dataset = QADataset(data['question'].tolist(), data['context'].tolist(), data['answer'].tolist())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  # Добавлено collate_fn

# Определение устройства (GPU/CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Настройка оптимизатора и расписания обучения
optimizer = AdamW(model.parameters(), lr=3e-5)
num_epochs = 2
num_training_steps = num_epochs * len(dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Тренировочный цикл
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        # Перемещение данных на нужное устройство
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        # Прямой проход
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss

        # Обратный проход
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Сохранение модели
model.save_pretrained('roberta-base-squad2_model')  # Указан путь для сохранения в Colab
tokenizer.save_pretrained('/roberta-base-squad2_model')
