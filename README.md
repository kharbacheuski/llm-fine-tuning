# LLM Fine-tuning
Программа для Fine-tuning (дообучения) модели на собственных данных
В данном случае используется dataset с huggingface хаба, вы можете залить свой dataset используя аккаунт huggingface

Для finе-tuning модели требуется GPU. 
Модель дообучается на специфичных данных в формата: вопрос-ответ

Models - https://huggingface.co/models
Datasets - https://huggingface.co/datasets

## Запуск
Для пакета torch надо устанавлиать Anaconda (https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Windows-x86_64.exe), и через нее запускать среду разработки (у меня получилось только так). 
По другому python не видит этот пакет, хоть он и установлен.

Также для запуска модели требуется установка cuda (для использования ресурсов GPU).
Для этого в терминале Anaconda нужно ввести команду:

```conda install pytorch torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia```

```pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html```

Чтобы запустить проект я советую создавать виртуальное окружение. 
Для этого надо ввести в консоли две команды:

```python -m venv .venv```

```.venv\Scripts\activate```

После этого можно подгружать другие зависимости указанный в файле:

```pip install -r requirements.txt```

Запуск скрипта:

```python .\main.py```

## Источники
https://myscale.com/blog/how-to-fine-tune-llm-from-huggingface/
https://stackabuse.com/guide-to-fine-tuning-open-source-llms-on-custom-data/