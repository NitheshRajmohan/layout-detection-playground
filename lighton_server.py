"""
FastAPI server for LightOn OCR layout detection.
Deploy on GPU server: uvicorn lighton_server:app --host 0.0.0.0 --port 8002
"""

import base64
import json
import re
import tempfile
import os
from io import BytesIO

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel
from transformers import LightOnOcrProcessor, LightOnOcrForConditionalGeneration

MODEL_ID = "lightonai/LightOnOCR-2-1B"

app = FastAPI(title="LightOn OCR Layout Detection")

print(f"Loading {MODEL_ID}...")
processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)
model = LightOnOcrForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Model loaded.")


LAYOUT_PROMPT = (
    "Detect all layout elements in this document page. "
    "For each element, output a JSON array of objects with "
    "'label' (e.g. table, figure, header, text, footer, logo, signature, stamp) "
    "and 'bbox' as [x1, y1, x2, y2] normalized to 0-1. Output only JSON."
)


def run_inference(image: Image.Image) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": LAYOUT_PROMPT},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=prompt_text, images=image, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=2048, temperature=0.1)

    return processor.decode(output[0], skip_special_tokens=True)


def parse_detections(raw: str) -> list[dict]:
    """Try to extract JSON array of detections from model output."""
    # Try to find JSON array in the response
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group())
        detections = []
        for det in parsed:
            if isinstance(det, dict) and "label" in det and "bbox" in det:
                bbox = det["bbox"]
                if isinstance(bbox, list) and len(bbox) == 4:
                    detections.append({
                        "label": str(det["label"]),
                        "bbox": [float(c) for c in bbox],
                    })
        return detections
    except (json.JSONDecodeError, ValueError):
        return []


class DetectRequest(BaseModel):
    image_base64: str


@app.post("/detect")
async def detect(req: DetectRequest):
    try:
        image_data = base64.b64decode(req.image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = ImageOps.exif_transpose(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    raw = run_inference(image)
    detections = parse_detections(raw)

    return {
        "status": "success",
        "detections": detections,
        "raw_response": raw,
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model": MODEL_ID}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
