"""
HTTP wrapper around CardDetector, so the Node server can call it over the
network instead of needing Python in-process. Models load once at startup;
each request after that is just inference.

Run with: uvicorn service:app --host 0.0.0.0 --port 8008
"""

import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from detector import CardDetector

detector: CardDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = CardDetector()
    yield


app = FastAPI(lifespan=lifespan)


class DetectRequest(BaseModel):
    imageBase64: str
    allowedCardIds: list[str] | None = None
    sleeveColor: str | None = None  # "#rrggbb" - the player's own sleeve color, if they've set one


def _parse_hex_color(hex_str: str) -> tuple[int, int, int] | None:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        return None
    try:
        return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


@app.post("/detect")
def detect(req: DetectRequest):
    _, _, data = req.imageBase64.partition(",")
    raw = base64.b64decode(data or req.imageBase64)
    image_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {"detections": []}

    allowed = set(req.allowedCardIds) if req.allowedCardIds else None
    sleeve_color = _parse_hex_color(req.sleeveColor) if req.sleeveColor else None
    return {"detections": detector.detect(image_bgr, allowed_card_ids=allowed, sleeve_color=sleeve_color)}


@app.get("/health")
def health():
    return {"ready": detector is not None}
