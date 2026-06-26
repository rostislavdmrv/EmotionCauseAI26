<div align="center">

# 🧠 EmoTrace — EmoCause AI

### Анализ на причините за емоционални реакции в разговорни данни чрез съвременни AI системи
### Analysis of Causes Behind Emotional Reactions in Conversational Data Using Modern AI Systems

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/Dataset-GPL--3.0-green)](https://huggingface.co/datasets/NUSTM/ECF)

</div>

---

## 🇧🇬 Български

### 📌 За проекта

Настоящата дипломна работа изследва задачата за **автоматично извличане на причинно-следствени двойки от емоции в разговорни данни (Emotion-Cause Pair Extraction — ECPE)**. Дадено изречение като _„Много съм разстроен, защото винаги ме игнорираш, когато имам нужда от теб"_ — системата трябва да открие:

- **Емоция:** разстроен (sadness)
- **Причина:** „защото винаги ме игнорираш" (в предходно изказване)

Задачата е значително по-трудна от обикновена класификация на емоции, тъй като изисква разбиране на причинно-следствени връзки в целия контекст на разговора.

### 🗂️ Структура на проекта

```
EmoCauseAI/
├── app.py                    # Главен файл на EmoTrace (Streamlit)
├── evaluate_all.py           # Скрипт за оценка и сравнение на всички модели
├── merge_phi3.py             # Обединяване на LoRA adapter с базовия Phi-3 модел
├── evaluation_results.csv    # Резултати от оценката (F1, Precision, Recall)
│
├── pages/                    # Страници на приложението EmoTrace
│   ├── landing.py            # Начална страница с представяне на моделите
│   ├── analyzer.py           # Въвеждане на разговор и визуализация (bubble chat)
│   ├── results.py            # Резултати от анализа по модели
│   └── evaluation.py         # Сравнение на метриките на моделите
│
├── src/                      # Логика на моделите
│   ├── baseline_model.py     # TF-IDF + Logistic Regression
│   ├── bloom560m_model.py    # BLOOM-560M (fine-tuned)
│   ├── emobert_model.py      # EmoBERTa + TransformerEncoder
│   └── phi3mini_model.py     # Phi-3 Mini с LoRA / GGUF
│
├── models/
│   ├── baseline/             # cause_model.pkl, emotion_model.pkl, vectorizer.pkl
│   ├── emobert/              # emotion_classifier.pth, cause_pair_extractor.pth
│   │   └── cause_span_qa/    # QA модел с LoRA за извличане на причинен фрагмент
│   ├── bloom560m/            # Дообучен BLOOM-560M
│   ├── phi3mini/             # LoRA adapter (adapter_config.json, adapter_model.safetensors)
│   ├── phi3mini_full/        # Пълен обединен модел (4 × safetensors файла)
│   └── phi3mini_gguf/        # Квантизиран GGUF формат за llama.cpp
│       ├── Phi-3-mini-4k-instruct-q4.gguf   # Базов модел (квантизиран)
│       └── phi3mini_finetuned.gguf           # Дообучен модел (GGUF)
│
├── data/
│   └── ecpe_train.json       # ECF dataset (NUSTM/ECF, GPL-3.0)
│
├── train/
│   └── baseline.ipynb        # Notebook за трениране на baseline модела
│
└── styles/
    └── main.css              # CSS стилове на EmoTrace
```

### 🤖 Реализирани модели

| Модел | Параметри | Подход | Emotion F1 | Cause F1 |
|---|---|---|---|---|
| **Baseline** | — | Rule-based + TF-IDF + Logistic Regression | 0.522 | 0.018 |
| **BLOOM-560M** | 560M | Генеративен LLM (fine-tuned) | 0.255 | 0.000 |
| **EmoBERTa** | 355M | RoBERTa + TransformerEncoder (специализиран) | **0.660** | **0.629** |
| **Phi-3 Mini** | 3.8B | SLM с LoRA адаптери (r=16, alpha=32) | N/A | 0.207 |

> 🏆 **EmoBERTa е победителят** — 355M параметъра надминават Phi-3 Mini (3.8B) при всички метрики. Специализацията превъзхожда размера.

**Предобучени модели в HuggingFace:**
- EmoBERTa: [tae898/emoberta-large](https://huggingface.co/tae898/emoberta-large)
- Phi-3 Mini: [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- BLOOM-560M: [bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m)

**Набор от данни:**
- ECF Dataset: [NUSTM/ECF](https://huggingface.co/datasets/NUSTM/ECF) — 1374 разговора, 13 619 изказвания, 9 364 анотирани двойки (Friends транскрипции)

### 📊 Ключови резултати

**Сравнителни резултати на четирите подхода:**

| Модел | Emotion F1 | Cause Precision | Cause Recall | Cause F1 |
|---|---|---|---|---|
| Baseline | 0.522 | 0.611 | 0.009 | 0.018 |
| BLOOM-560M | 0.255 | — | — | 0.000 |
| EmoBERTa | **0.660** | **0.607** | **0.652** | **0.629** |
| Phi-3 Mini | N/A | 0.364 | 0.144 | 0.207 |

**EmoBERTa по емоционална категория:**

| Категория | Baseline F1 | BLOOM F1 | EmoBERTa F1 |
|---|---|---|---|
| neutral | 0.76 | 0.45 | **0.83** |
| joy | 0.54 | 0.31 | **0.70** |
| anger | 0.45 | 0.17 | **0.63** |
| sadness | 0.38 | 0.09 | **0.55** |
| disgust | 0.18 | 0.04 | **0.34** |
| fear | 0.14 | 0.03 | **0.28** |

### 🚀 Инсталация и стартиране

#### Изисквания
- Python 3.11
- GPU с CUDA (препоръчително за LLM моделите; обучението е проведено на Kaggle T4, 16GB VRAM)
- ~10 GB свободно дисково пространство (без phi3mini_full)

#### Стъпки

**1. Клонирай проекта**
```bash
git clone https://github.com/[твоето-потребителско-име]/EmoCauseAI.git
cd EmoCauseAI
```

**2. Създай виртуална среда**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

**3. Инсталирай зависимостите**
```bash
pip install -r requirements.txt
```

**4. Свали моделите**



Постави ги в папка `models/` според структурата по-горе.

> **Забележка:** `phi3mini_full/` (~7 GB) е необходим само ако искаш да използваш пълния HuggingFace вариант. За inference е достатъчен `phi3mini_gguf/phi3mini_finetuned.gguf`.

**5. Стартирай приложението EmoTrace**
```bash
streamlit run app.py
```

Приложението ще се отвори на `http://localhost:8501`

### 📦 Основни зависимости

```
streamlit==1.57
transformers==5.8.0
torch>=2.0
llama-cpp-python
peft
trl
scikit-learn
pandas
numpy
matplotlib
seaborn
vaderSentiment
spacy
wordcloud
```

### 🏗️ Архитектурни решения

**EmoBERTa + TransformerEncoder**
- База: `tae898/emoberta-large` (RoBERTa-large дообучен за ERC)
- Специализирана TransformerEncoder архитектура с sinusoidal positional encoding и multi-head self-attention за моделиране на причинно-следствени зависимости
- BCEWithLogitsLoss за матрицата на причинно-следствените двойки
- QA модел с LoRA (r=16) за извличане на точния причинен фрагмент

**Phi-3 Mini с LoRA**
- База: `microsoft/Phi-3-mini-4k-instruct` (3.8B параметъра)
- LoRA адаптери: r=16, alpha=32, обучими параметри: 13.6M от 3.8B
- Обучение на Kaggle (T4 GPU, 16GB VRAM)
- Конвертация към GGUF формат за inference с `llama.cpp` без GPU

**Технически предизвикателства и решения (Windows)**
- Long Path ограничение → активиране на long paths в системния регистър
- GGUF конвертация → `convert_hf_to_gguf.py` от `llama.cpp`
- auto_map конфликт при зареждане → ръчно задаване на `device_map`

---

## 🇬🇧 English

### 📌 About the Project

This thesis investigates the task of **automatic Emotion-Cause Pair Extraction (ECPE) from conversational data**. Given a conversation like _"I'm really upset because you always ignore me when I need you"_, the system must identify:

- **Emotion:** upset (sadness)
- **Cause:** "because you always ignore me" (potentially in a previous utterance)

This task is significantly more challenging than simple emotion classification, requiring understanding of causal relationships across the full conversation context.

### 🤖 Implemented Models

| Model | Parameters | Approach | Emotion F1 | Cause F1 |
|---|---|---|---|---|
| **Baseline** | — | Rule-based + TF-IDF + Logistic Regression | 0.522 | 0.018 |
| **BLOOM-560M** | 560M | Generative LLM (fine-tuned) | 0.255 | 0.000 |
| **EmoBERTa** | 355M | RoBERTa + custom TransformerEncoder | **0.660** | **0.629** |
| **Phi-3 Mini** | 3.8B | SLM with LoRA adapters (r=16, alpha=32) | N/A | 0.207 |

> 🏆 **EmoBERTa wins** — 355M parameters outperform Phi-3 Mini (3.8B) across all metrics. **Specialization beats scale.**

### 📊 Dataset

- **ECF (Emotion-Cause-in-Friends):** [NUSTM/ECF on HuggingFace](https://huggingface.co/datasets/NUSTM/ECF) (GPL-3.0)
- 1,374 conversations · 13,619 utterances · 9,364 annotated cause-emotion pairs
- Source: Friends TV show transcriptions
- 43.5% neutral utterances · 50.3% self-caused pairs · 3.8% retrospective causes

### 🚀 Installation

```bash
git clone https://github.com/[your-username]/EmoCauseAI.git
cd EmoCauseAI
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

### 📚 Pre-trained Models Used

| Model | HuggingFace Link |
|---|---|
| EmoBERTa | [tae898/emoberta-large](https://huggingface.co/tae898/emoberta-large) |
| Phi-3 Mini 4K Instruct | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| BLOOM-560M | [bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m) |

### 📖 Key References

- Xia & Ding (2019). Emotion-Cause Pair Extraction. [ACL 2019](https://aclanthology.org/P19-1096)
- Wang et al. (2023). Multimodal ECPE in Conversations. [IEEE TAFFC](https://doi.org/10.1109/TAFFC.2022.3226559)
- Kim & Park (2021). EmoBERTa. [arXiv:2108.12009](https://arxiv.org/abs/2108.12009)
- Hu et al. (2021). LoRA. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Abdin et al. (2024). Phi-3 Technical Report. [arXiv:2404.14219](https://arxiv.org/abs/2404.14219)

---

## 👤 Автор / Author
---

<div align="center">
<sub>ECF Dataset е под GPL-3.0 лиценз (NUSTM/ECF). Всички останали компоненти са с академична цел.</sub>
</div>

