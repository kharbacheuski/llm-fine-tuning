# LLM Fine-tuning
Программа для fine-tuning/дообучения модели на собственных данных
В данном случае используется [dataset с huggingface](https://huggingface.co/datasets), вы можете использовать свой датасет используя pandas.

Для нормального процесса дообучения модели требуется GPU с CUDA. Если у вас нет мощностей, вы можете просто вставить этот код в [Google Colab](https://colab.research.google.com/) и обучить модель на видеокартах Google (T4 GPU).

Для использования некоторых моделей с huggingface потребуется ввести token. Его можно найти/создать в личном кабинете на сайте 

## Алгоритм действий

1) Стянуть [модель с huggingface](https://huggingface.co/models).
2) Обучить. После сохранения в папки появится адаптер для весов модели (в ходе обучения веса не сохраняются).
3) Сделать слияние базовой модели и адаптера весов. Реализация в файле ```merge.py```
4) Cконвертирровать модель в gguf формат (если требуется) и использовать например в Ollama. Конвертировать используя [llama.cpp](https://github.com/ggerganov/llama.cpp)
5) Использовать модель, подгрузив ее локально (указав вместо имени путь к модели).

## Используемое оборудование
- Видеокарта: Nvidia GTX 1660 Ti
- Оперативная память: 32Гб
- Процессор: Intel Core i7-8700K CPU 3.70GHz

## ПО
- Windows 10
- CUDA 11.8
- Python 3.12.4

## Запуск 

1) Установка основных пакетов. Torch для работы с CUDA (версия 11.8)

```bash
# Можно использовать другие версии библиотек, смотрите совместимость
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

# Установка других пакетов из файла
pip install -r requirements.txt
```

2) Запуск скрипта обучения, обучение, сохранение результатов.

После обучения модели будет сохранен только адаптер. **Веса модели не будут сохранены.**
Для того чтобы получить полностью готовую модель нужно подгрузить базовые веса модели и слить их с адаптером.

**Уделите внимание следующим параметрам обучения. Они могут не подходить для вашей модели**

```bash
...
# Шаг 3: Настройка PEFT (Lora)
peft_config = LoraConfig(
    r=16,  # Размер матриц LoRA
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Модули, к которым применяется LoRA
    lora_dropout=0.1,
    bias="none"
)

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
```

3) Запуск скрипта слияния ```/src/merge.py``` для получения полностью готовой модели с весами

## Источники

https://myscale.com/blog/how-to-fine-tune-llm-from-huggingface/

https://pytorch.org/get-started/locally/

https://stackabuse.com/guide-to-fine-tuning-open-source-llms-on-custom-data/