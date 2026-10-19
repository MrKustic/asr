# Распознавание речи: FastConformer + CTC

Это проект, созданный для распознавания речи по аудио, основанный на [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

В проекте реализованы собственный Conformer, FastConformer, обучение с CTC loss, токенизация SentencePiece BPE, argmax и собственный CTC beam search. Для декодирования с внешней языковой моделью используются pyctcdecode и KenLM.

- [Demo notebook](notebooks/demo.ipynb)
- [Модель и ресурсы на Hugging Face](https://huggingface.co/MrKustic/asr)

## Окружение

Рекомендуется использовать conda для создания отдельного окружения с нужной версией Python, мы использовали PYTHON_VERSION=3.13.16.

```bash
conda create -n env python=PYTHON_VERSION
conda activate env
```

## Установка

Все команды выполняются из корня репозитория. Для demo используется новая сессия Google Colab, для ускорения инференса выберите GPU.

```bash
git clone https://github.com/MrKustic/asr.git
cd asr
pip install -r requirements.txt
```

Установите Git hooks для разработки:

```bash
pre-commit install
```

## Скачивание ресурсов

Скрипт `scripts/download_hf_folder.py` сохраняет структуру скачанной папки внутри `--local-dir`.

```bash
python scripts/download_hf_folder.py --username MrKustic --project asr --folder model --local-dir data
python scripts/download_hf_folder.py --username MrKustic --project asr --folder tokenizers --local-dir data
python scripts/download_hf_folder.py --username MrKustic --project asr --folder lm --local-dir data
python scripts/download_hf_folder.py --username MrKustic --project asr --folder example --local-dir data
```

| Ресурс | Локальный путь |
| --- | --- |
| Веса | `data/model/model_best.pth` |
| Токенизатор | `data/tokenizers/tokenizer-960.model` |
| Языковая модель | `data/lm/3-gram.lower.arpa` |
| Пример датасета | `data/example/` |

## Как использовать

Рекомендуем ознакомиться с [demo notebook](notebooks/demo.ipynb).

## Оценка на LibriSpeech other

Для оценки метрик на dev-other и test-other датасета LibriSpeech используйте команду.

```bash
python inference.py -cn=inference \
  datasets=librispeech_eval_other \
  inferencer.from_pretrained=data/model/model_best.pth \
  tokenizer.model_path=data/tokenizers/tokenizer-960.model \
  inferencer.save_path=data/librispeech/predictions \
  ctc_decoder=beam_lm \
  ctc_decoder.kenlm_model_path=data/lm/3-gram.lower.arpa
```

Предсказания сохраняются в `data/librispeech/predictions/` в подпапки `val_other` и `test_other`.

## Расчёт CER и WER

После инференса на LibriSpeech other можно повторно вычислить метрики по сохранённым предсказаниям:

```bash
python -u calc_metrics.py -cn=calc_metrics \
  datasets=librispeech_eval_other \
  tokenizer.model_path=data/tokenizers/tokenizer-960.model \
  inferencer.save_path=data/librispeech/predictions
```

Скрипт читает предсказания из подпапок `val_other` и `test_other`. Он не выполняет прямой проход модели и не загружает её веса, но использует исходный датасет и токенизатор.

## Инференс на собственной папке

Вы можете запустить инференс на собственном датасете такого формата:

Формат датасета:

```text
dataset/
├── audio/
│   ├── utterance_1.wav
│   ├── utterance_2.flac
│   └── utterance_3.mp3
└── transcriptions/      # опционально
    ├── utterance_1.txt
    ├── utterance_2.txt
    └── utterance_3.txt
```

ID файлов должны быть уникальными и совпадать между аудио и текстами. В demo передайте ссылку на папку Google Drive. Для запуска из терминала используйте локальную папку и замените `AUDIO_DIR`, `TRANSCRIPTION_DIR` и `PREDICTIONS_DIR` в командах ниже. Если эталонных текстов нет, укажите `datasets.inference.transcription_dir=null` и пропустите расчёт метрик.

```bash
python -u inference.py -cn=inference \
  datasets=custom_dataset \
  datasets.inference.audio_dir=AUDIO_DIR \
  datasets.inference.transcription_dir=TRANSCRIPTION_DIR \
  inferencer.from_pretrained=data/model/model_best.pth \
  tokenizer.model_path=data/tokenizers/tokenizer-960.model \
  inferencer.save_path=PREDICTIONS_DIR \
  ctc_decoder=beam_lm \
  ctc_decoder.kenlm_model_path=data/lm/3-gram.lower.arpa \
  metrics.inference=[]
```

Для сравнения замените `ctc_decoder=beam_lm` на `ctc_decoder=argmax` или `ctc_decoder=my_beam` и удалите аргумент `ctc_decoder.kenlm_model_path`.

Предсказания сохраняются в `PREDICTIONS_DIR/inference/<utterance_id>.txt`. При наличии транскрипций вычислите метрики:

```bash
python -u calc_metrics.py -cn=calc_metrics \
  datasets=custom_dataset \
  datasets.inference.audio_dir=AUDIO_DIR \
  datasets.inference.transcription_dir=TRANSCRIPTION_DIR \
  tokenizer.model_path=data/tokenizers/tokenizer-960.model \
  inferencer.save_path=PREDICTIONS_DIR
```

`transcription_dir` - папка с транскрипциями. `save_path` - родительская с предсказаниями.

## Обучение и воспроизведение

Основной конфиг - `src/configs/fast_conformer_full_train.yaml`. По умолчанию используются LibriSpeech train-clean-100, train-clean-360 и train-other-500; для валидации - dev-clean и dev-other.

Следующая команда воспроизводит финальную модель:

```bash
python -u train.py -cn=fast_conformer_full_train \
  tokenizer.model_path=data/tokenizers/tokenizer-960.model \
  writer.run_name=UNIQUE_RUN_NAME \
  writer.mode=online
```

Конфиг задаёт 180 эпох по 2000 батчей.

## Ссылки

Код основан на [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).
