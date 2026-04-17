# Layout Detection Playground

A Streamlit app to compare document layout detection across multiple vision models. Upload a bank statement (PDF or image), pick a model, and visualize detected layout regions (tables, headers, text blocks, figures, etc.) with bounding boxes.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)

## Models Supported

| Model | Type |
|-------|------|
| **Qwen3.5-VL** (finetuned) | VLM |
| **DeepSeek OCR2** | VLM |
| **LightOn OCR 2** | VLM |
| **OpenRouter** (any vision model) | API — GPT-4o, Gemini, Claude, Llama, etc. |

## Setup

### 1. Clone and install

```bash
git clone https://github.com/NitheshRajmohan/layout-detection-playground.git
cd layout-detection-playground
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
- `GPU_HOST` — IP/hostname of the server running the model backends
- `OPENROUTER_API_KEY` — your [OpenRouter](https://openrouter.ai/) API key (for the OpenRouter option)

### 3. Run the app

```bash
streamlit run app.py
```

## How It Works

1. Upload a PDF or image of a bank statement
2. Select a model from the dropdown
3. Click **Detect Layout** — the app sends the page to the selected model backend
4. View input vs. output side-by-side with color-coded bounding boxes
5. Expand **Detection Details** for label + coordinate info

## Architecture

```
Browser  →  Streamlit App  →  Model Backends (FastAPI)
                                ├── Qwen3.5-VL (finetuned)
                                ├── DeepSeek OCR2
                                └── LightOn OCR 2
                           →  OpenRouter API (cloud VLMs)
```

## Project Structure

```
├── app.py               # Streamlit frontend
├── requirements.txt
├── .env.example
└── .gitignore
```
