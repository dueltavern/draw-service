# draw-service

A small FastAPI wrapper around the detection + recognition models from
[draw2](https://github.com/HichTala/draw2) (AGPL-3.0), used by Duel Board's
card scanning instead of the client-side CV heuristic + CLIP embedding
matching this project used previously.

It's a separate Python process from `server/` (Node) and `client/` (Vite) -
run all three side by side during development.

## Setup

```bash
cd draw-service
uv sync
```

Without uv: `pip install fastapi "uvicorn[standard]" pydantic ultralytics transformers accelerate huggingface_hub opencv-python-headless pillow torch numpy`.

## Running

```bash
uv run uvicorn service:app --host 0.0.0.0 --port 8008
```

First request after starting downloads the YOLO detector and card
classifier from Hugging Face (a few hundred MB+) and caches them locally -
this can take a while and needs a working internet connection the first
time. A GPU (CUDA) is used automatically if available; otherwise it runs on
CPU, which will be noticeably slower per scan.

## API

`POST /detect`

```json
{ "imageBase64": "data:image/jpeg;base64,...", "allowedCardIds": ["46986414"] }
```

`allowedCardIds` is optional - when given, only detections matching one of
those card IDs are returned (mirrors draw2's own deck-list filtering,
without needing a `.ydk` file).

Response:

```json
{
  "detections": [
    { "points": [[x, y], [x, y], [x, y], [x, y]], "cardId": "46986414", "cardName": "Dark Magician", "confidence": 0.94 }
  ]
}
```

`GET /health` returns `{ "ready": true }` once models have finished loading.

Set `DETECTION_LOGGING = True/False` at the top of `detector.py` to toggle
per-candidate pass/reject logging - useful when a real board's cards aren't
coming through and you want to see whether YOLO found them at all versus
the classifier rejecting them.

## Why this isn't just draw2's own `Draw` class

An earlier version of this service depended on and called draw2's `Draw`
class directly, via its public API, rather than reimplementing its
pipeline. That turned out to have real problems once tested against actual
model weights, not just code review:

- **`Draw.__init__` reloads both models from scratch on every construction.**
  It's designed to own one fixed video/image source for its whole lifetime,
  not to be built once and reused - so calling it fresh per scan (the only
  way to use its public API for discrete requests) pays the full model-load
  cost every time, not just inference time.
- **A single ambiguous card blanks out the whole frame's results.**
  `Draw.process()`'s per-card loop uses `break` (not `continue`) when its
  rotation heuristic can't confidently orient one of the detected cards - so
  one bad card silently drops every other card detected in that same frame
  too. Confirmed directly: a 7-card board scan returned zero results despite
  YOLO detecting cards, purely because of this.
- **No per-card coordinates or confidence scores.** `process()` only returns
  classified label strings, so a scan couldn't be matched to a specific
  clicked card when several were in frame, and there was no way to draw a
  detection outline.

`detector.py` instead loads both models once at service startup and exposes
a plain `detect()` call, following the same detect → perspective-warp →
rotate → classify sequence as draw2's own `Draw.process()` (that sequence is
what makes the models' raw output usable at all), but with `continue`
instead of `break`, and a fallback that tries all 4 orientations through the
classifier when the rotation heuristic can't decide - instead of dropping
the card. It also returns per-card coordinates and confidence, since we
compute them anyway. Credit to HichTala for the pipeline design and for
training and hosting the underlying models.

## Licensing note

This service downloads and calls draw2's published models via the Hugging
Face Hub API rather than depending on its package, but the
detect/crop/rotate/classify sequence in `detector.py` is a direct
reimplementation of draw2's own `Draw.process()` method. draw2 is
AGPL-3.0 - if you deploy this service as part of a network-accessible app,
review what that license requires for your own distribution.
