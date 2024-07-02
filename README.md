# LLM Fine-tuning
Программа для Fine-tuning (дообучения) модели на собственных данных
В данном случае используется dataset с huggingface хаба, вы можете залить свой dataset используя аккаунт huggingface

Для finе-tuning модели требуется GPU. 
Модель дообучается на специфичных данных в формата: вопрос-ответ

Models - https://huggingface.co/models
Datasets - https://huggingface.co/datasets

Для использования моделей с huggingface потребуется ввести token. Его можно найти/создать в личном кабинете на сайте 

## Используемое оборудование
Видеокарта: Nvidia GTX 1660 Ti
Оперативная память: 32Гб

## Запуск
Я запускал данный скрипт на WSL Ubuntu 20.04

1) Установка основных пакетов. Torch для работы с CUDA (поставляется внутри пакета)

```pip3 install torch torchvision torchaudio```

2) Установка других зависимостей

```pip3 install -r requirements.txt```

3) Запуск скрипта:

```python3 .\training.py```

## Источники
https://myscale.com/blog/how-to-fine-tune-llm-from-huggingface/
https://stackabuse.com/guide-to-fine-tuning-open-source-llms-on-custom-data/