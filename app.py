"""
Layout Detection Playground
Compare layout detection across multiple models on bank statement PDFs.

Run: streamlit run app.py
"""

import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

st.set_page_config(page_title="Layout Detection Playground", layout="wide")

# ── Config ───────────────────────────────────────────────────
GPU_HOST = os.getenv("GPU_HOST", "localhost")
QWEN_URL = f"http://{GPU_HOST}:8000"
DEEPSEEK_URL = f"http://{GPU_HOST}:8001"
LIGHTON_URL = f"http://{GPU_HOST}:8002"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
RENDER_DPI = 200

MODELS = {
    "Qwen3.5-VL (finetuned)": "qwen",
    "DeepSeek OCR2": "deepseek",
    "LightOn OCR": "lighton",
    "OpenRouter (select model)": "openrouter",
}

COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8C471", "#82E0AA", "#F1948A", "#AED6F1", "#D7BDE2",
]

OPENROUTER_VISION_MODELS = [
    "qwen/qwen-2.5-vl-72b-instruct",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.5-pro-preview",
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "anthropic/claude-sonnet-4",
    "meta-llama/llama-4-maverick",
]

LAYOUT_PROMPT = (
    "You are a document layout detection model. Detect all layout elements in this "
    "document page image. For each element output a JSON array of objects, each with "
    "'label' (e.g. table, header, text, footer, figure, logo, signature, stamp, "
    "page_number, watermark) and 'bbox' as [x1, y1, x2, y2] with coordinates "
    "normalized between 0 and 1. Output ONLY the JSON array, no explanation."
)


# ── Helpers ──────────────────────────────────────────────────
def pdf_to_images(pdf_bytes: bytes, dpi: int = RENDER_DPI) -> list[Image.Image]:
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    return images


def image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt, quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

    label_colors = {}
    for det in detections:
        label = det.get("label", "unknown")
        if label not in label_colors:
            label_colors[label] = COLORS[len(label_colors) % len(COLORS)]

    for det in detections:
        label = det.get("label", "unknown")
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        # Scale normalized coords to pixel coords
        px1, py1, px2, py2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        color = label_colors[label]

        draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
        # Label background
        text_bbox = draw.textbbox((px1, py1), label, font=font)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
        draw.text((px1, py1), label, fill="white", font=font)

    return img


# ── Model Callers ────────────────────────────────────────────
def call_qwen(pdf_bytes: bytes, page_idx: int) -> list[dict]:
    """Qwen layout service accepts full PDF, returns all pages."""
    resp = requests.post(
        f"{QWEN_URL}/layout",
        files={"pdf": ("input.pdf", pdf_bytes, "application/pdf")},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("pages", [])
    for p in pages:
        if p["page"] == page_idx + 1:
            return p.get("detections", [])
    return []


def call_deepseek(image: Image.Image) -> list[dict]:
    b64 = image_to_base64(image)
    resp = requests.post(
        f"{DEEPSEEK_URL}/extract",
        json={"image_base64": b64},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    # DeepSeek returns {"tables": [{"content": label, "bbox": {"x1":..,"y1":..,"x2":..,"y2":..}}]}
    detections = []
    for item in data.get("tables", []):
        bbox = item.get("bbox", {})
        if isinstance(bbox, dict):
            detections.append({
                "label": item.get("content", "unknown"),
                "bbox": [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)],
            })
        elif isinstance(bbox, list) and len(bbox) == 4:
            detections.append({"label": item.get("content", "unknown"), "bbox": bbox})
    return detections


def call_lighton(image: Image.Image) -> list[dict]:
    b64 = image_to_base64(image)
    resp = requests.post(
        f"{LIGHTON_URL}/detect",
        json={"image_base64": b64},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("detections", [])


def call_openrouter(image: Image.Image, model_id: str) -> list[dict]:
    if not OPENROUTER_API_KEY:
        st.error("Set OPENROUTER_API_KEY in .env")
        return []

    b64 = image_to_base64(image, fmt="PNG")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": LAYOUT_PROMPT},
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    raw = data["choices"][0]["message"]["content"]
    # Parse JSON array from response
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        st.warning(f"Could not parse detections. Raw response:\n{raw[:500]}")
        return []
    try:
        parsed = json.loads(match.group())
        detections = []
        for det in parsed:
            if isinstance(det, dict) and "label" in det and "bbox" in det:
                bbox = det["bbox"]
                if isinstance(bbox, list) and len(bbox) == 4:
                    detections.append({"label": str(det["label"]), "bbox": [float(c) for c in bbox]})
        return detections
    except (json.JSONDecodeError, ValueError) as e:
        st.warning(f"JSON parse error: {e}\nRaw: {raw[:500]}")
        return []


# ── UI ───────────────────────────────────────────────────────
st.title("Layout Detection Playground")

# Sidebar
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Model", list(MODELS.keys()))
    model_key = MODELS[selected_model]

    openrouter_model_id = None
    if model_key == "openrouter":
        openrouter_model_id = st.selectbox("OpenRouter Model", OPENROUTER_VISION_MODELS)

    uploaded_file = st.file_uploader("Upload Bank Statement (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])

if not uploaded_file:
    st.info("Upload a bank statement PDF or image to get started.")
    st.stop()

# Process upload
file_bytes = uploaded_file.read()
is_pdf = uploaded_file.name.lower().endswith(".pdf")

if is_pdf:
    pages = pdf_to_images(file_bytes)
else:
    pages = [Image.open(BytesIO(file_bytes)).convert("RGB")]

# Page selector
if len(pages) > 1:
    page_idx = st.sidebar.selectbox("Page", range(len(pages)), format_func=lambda x: f"Page {x + 1}")
else:
    page_idx = 0

current_image = pages[page_idx]

# Run detection
if st.sidebar.button("Detect Layout", type="primary", use_container_width=True):
    with st.spinner(f"Running {selected_model}..."):
        start = time.time()
        try:
            if model_key == "qwen":
                if is_pdf:
                    detections = call_qwen(file_bytes, page_idx)
                else:
                    # Wrap image in a single-page PDF for the Qwen endpoint
                    img_buf = BytesIO()
                    current_image.save(img_buf, format="PDF")
                    detections = call_qwen(img_buf.getvalue(), 0)
            elif model_key == "deepseek":
                detections = call_deepseek(current_image)
            elif model_key == "lighton":
                detections = call_lighton(current_image)
            elif model_key == "openrouter":
                detections = call_openrouter(current_image, openrouter_model_id)
            else:
                detections = []
            elapsed = time.time() - start
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to model server. Make sure the service is running on the GPU.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"Server error: {e.response.status_code} - {e.response.text[:300]}")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.session_state["detections"] = detections
    st.session_state["elapsed"] = elapsed
    st.session_state["model_used"] = selected_model

# Display results
if "detections" in st.session_state:
    detections = st.session_state["detections"]
    elapsed = st.session_state["elapsed"]
    model_used = st.session_state["model_used"]

    st.markdown(f"**{model_used}** | {len(detections)} detections | {elapsed:.1f}s")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(current_image, use_container_width=True)

    with col2:
        st.subheader("Detected Layout")
        if detections:
            output_img = draw_detections(current_image, detections)
            st.image(output_img, use_container_width=True)
        else:
            st.warning("No detections returned.")
            st.image(current_image, use_container_width=True)

    # Detection details
    if detections:
        with st.expander("Detection Details", expanded=False):
            for i, det in enumerate(detections):
                bbox = det.get("bbox", [])
                bbox_str = ", ".join(f"{c:.3f}" for c in bbox)
                st.text(f"{i+1}. {det.get('label', '?')}  [{bbox_str}]")
else:
    st.image(current_image, caption=f"Page {page_idx + 1}", use_container_width=True)
