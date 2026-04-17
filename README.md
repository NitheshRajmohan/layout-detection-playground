# Layout Detection Playground

A Streamlit app to compare document layout detection across multiple vision models. Upload a bank statement (PDF or image), pick a model, and visualize detected layout regions (tables, headers, text blocks, figures, etc.) with bounding boxes.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)

## Models Supported

| Model | Type | How it runs |
|-------|------|-------------|
| **Qwen3.5-VL** (finetuned) | VLM | Self-hosted GPU server |
| **DeepSeek OCR2** | VLM | Self-hosted GPU server |
| **LightOn OCR 2** | VLM | Self-hosted GPU server |
| **OpenRouter** (any vision model) | API | OpenRouter API — GPT-4o, Gemini, Claude, Llama, etc. |

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/layout-detection-playground.git
cd layout-detection-playground
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
- `GPU_HOST` — IP/hostname of your GPU server running the model backends
- `OPENROUTER_API_KEY` — your [OpenRouter](https://openrouter.ai/) API key (for the OpenRouter option)

### 3. Start model servers on GPU

Each model runs as a FastAPI service:

```bash
# Qwen (port 8000)
uvicorn layout_service:app --host 0.0.0.0 --port 8000

# DeepSeek OCR2 (port 8001)
python server.py  # runs on port 8001

# LightOn OCR (port 8002)
uvicorn lighton_server:app --host 0.0.0.0 --port 8002
```

### 4. Run the app

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
Browser  →  Streamlit (local)  →  GPU Server (FastAPI endpoints)
                                    ├── :8000  Qwen3.5-VL
                                    ├── :8001  DeepSeek OCR2
                                    └── :8002  LightOn OCR
                               →  OpenRouter API (cloud)
```

## Project Structure

```
├── app.py               # Streamlit frontend
├── lighton_server.py     # FastAPI wrapper for LightOn OCR (deploy on GPU)
├── requirements.txt
├── .env.example
└── .gitignore
```
