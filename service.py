"""
HTTP wrapper around CardDetector, so the Node server can call it over the
network instead of needing Python in-process. Models load once at startup;
each request after that is just inference.

Run with: uvicorn service:app --host 0.0.0.0 --port 8008
"""

import base64
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from detector import CardDetector

detector: CardDetector | None = None

# Diagnostic switch: set DRAW_SAVE_FRAMES to a folder path and every scanned
# frame is saved there (as a .jpg), next to what was detected in it (a .json
# with the same name) - for checking offline why a card was or wasn't read.
# Off unless set. Example:
#   DRAW_SAVE_FRAMES=frames uv run uvicorn service:app --host 0.0.0.0 --port 8008
SAVE_FRAMES_DIR = os.environ.get("DRAW_SAVE_FRAMES")


def _save_frame(image_bgr, detections):
    folder = Path(SAVE_FRAMES_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    stem = str(folder / (time.strftime("%Y%m%d-%H%M%S-") + f"{int(time.time() * 1000) % 1000:03d}"))
    cv2.imwrite(f"{stem}.jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(f"{stem}.json", "w") as f:
        json.dump(detections, f, indent=1)
    print(f"[diagnostic] saved {stem}.jpg ({len(detections)} detection(s))")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = CardDetector()
    yield


app = FastAPI(lifespan=lifespan)


class DetectRequest(BaseModel):
    imageBase64: str
    allowedCardIds: list[str] | None = None


@app.post("/detect")
def detect(req: DetectRequest):
    _, _, data = req.imageBase64.partition(",")
    raw = base64.b64decode(data or req.imageBase64)
    image_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {"detections": []}

    allowed = set(req.allowedCardIds) if req.allowedCardIds else None
    detections = detector.detect(image_bgr, allowed_card_ids=allowed)
    if SAVE_FRAMES_DIR:
        _save_frame(image_bgr, detections)
    return {"detections": detections}


@app.get("/health")
def health():
    return {"ready": detector is not None}
