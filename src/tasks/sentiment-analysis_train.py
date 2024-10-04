import datasets
import evaluate
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer)

# Загружаем данные
df = pd.read_csv('toxic.csv')
df.columns = ['text','label']
df['label'] = df['label'].astype(int)

# Конвертируем датафрейм в Dataset
train, test = train_test_split(df, test_size=0.3)
train = Dataset.from_pandas(train)
test = Dataset.from_pandas(test)

# Выполняем предобработку текста
tokenizer = AutoTokenizer.from_pretrained(
    'SkolkovoInstitute/russian_toxicity_classifier')

def tokenize_function(examples):
	return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_train = train.map(tokenize_function)
tokenized_test = test.map(tokenize_function)

# Загружаем предобученную модель
model = AutoModelForSequenceClassification.from_pretrained(
	'SkolkovoInstitute/russian_toxicity_classifier',
	num_labels=2)

# Задаем параметры обучения
training_args = TrainingArguments(
	output_dir = 'test_trainer_log',
	evaluation_strategy = 'epoch',
	per_device_train_batch_size = 6,
	per_device_eval_batch_size = 6,
	num_train_epochs = 5,
	report_to='none')

# Определяем как считать метрику
metric = evaluate.load('f1')
def compute_metrics(eval_pred):
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions=predictions, references=labels)

# Выполняем обучение
trainer = Trainer(
	model = model,
	args = training_args,
	train_dataset = tokenized_train,
	eval_dataset = tokenized_test,
	compute_metrics = compute_metrics)

trainer.train()

# Сохраняем модель
save_directory = './pt_save_pretrained'
#tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
#alternatively save the trainer
#trainer.save_model('CustomModels/CustomHamSpam')