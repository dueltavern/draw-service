"""
Card detection + recognition, using the same YOLO detector and image
classifier as HichTala/draw2 (https://github.com/HichTala/draw2, AGPL-3.0).

This is a reimplementation, not a use of draw2's own Draw class: Draw locks
in a single video/image source at construction time (`self.results` is a
generator over that one source) and reloads both models from scratch on
every construction, which fits its own CLI/OBS use case but not a server
answering discrete requests for different images with low latency.
CardDetector instead loads both models once and exposes a plain detect()
call. It also fixes a real bug found in draw2's own Draw.process(): its
per-card loop does `break` (not `continue`) when its rotation heuristic
can't confidently orient a card, silently dropping every other card
detected in the same frame - detect() below instead falls back to trying
all 4 orientations through the classifier for that one ambiguous card, and
keeps going.

The detect->warp->rotate->classify steps closely follow draw2's own
Draw.process(), since that sequence is what makes the model's raw output
usable: the YOLO oriented box isn't square-on without a perspective warp,
and the classifier needs a specific crop and an upright rotation to work
correctly. Credit to HichTala for that pipeline design and for training and
hosting the underlying models.
"""

import json
import math

import cv2
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, pipeline
from ultralytics import YOLO

DEFAULT_CONFIDENCE_THRESHOLD = 5  # percent, matches draw2's own default


# Logs per-candidate pass/reject reasoning and a per-frame summary - handy
# when diagnosing why a real board's cards aren't (or are) coming through.
DETECTION_LOGGING = True


def _extract_contours(roi, d, sigma_color, sigma_space, thresh):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    equalized = cv2.equalizeHist(gray)
    _, thresh_img = cv2.threshold(equalized, thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    edged = cv2.erode(thresh_img, kernel, iterations=3)
    edged = cv2.dilate(edged, kernel, iterations=3)

    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0] if len(contours) == 2 else contours[1]


def _get_txt_box(contour):
    rect = cv2.minAreaRect(contour)
    return np.intp(cv2.boxPoints(rect))


def _face_down_entry(pts, confidence=0.0):
    """A card-shaped region YOLO found but the classifier couldn't identify
    - most commonly an actual face-down card, whose back looks identical
    across the whole game, but also any card too blurry/occluded to read.
    Keeping its position (rather than dropping it) lets the client still
    place it in the right row via the same position-based grouping used for
    every other card, without needing to know what the card actually is."""
    return {
        "points": pts.tolist(),
        "cardId": None,
        "cardName": "Face-down card",
        "confidence": confidence,
    }


def _card_long_side(pts):
    return max(float(np.linalg.norm(np.subtract(pts[(i + 1) % 4], pts[i]))) for i in range(4))


def _find_card_backs(image_bgr, face_up_points):
    """Face-down cards the YOLO model doesn't detect at all - it learned card
    *faces*, and doesn't register a plain card back or a plain sleeve even
    at 0.5% confidence. Both are easy to spot without it, though, as a solid
    block of one kind of color the size of the other cards on the table:
    - unsleeved backs: very dark (the black swirl). The swirl is inset from
      the card's edges, so the corners are scaled out to cover the card.
    - light sleeves (white, cream, grey): nearly colorless and very bright,
      unlike a wooden table or a card face. A sleeve covers the whole card.

    For each, pixels of that kind are closed up into blobs, and a blob's
    minimum-area rectangle must be mostly filled (a solid card, not a shadow
    or the gaps in a keyboard), roughly card-proportioned (loosely - a camera
    looking at the table at an angle makes an upright card look squarer and
    a sideways one wider), card-sized (compared to the face-up cards when
    there are any, otherwise to the frame) and not sitting on a card YOLO
    already found."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    kinds = [
        (cv2.inRange(hsv, (0, 0, 0), (180, 255, CARD_BACK_MAX_BRIGHTNESS)), CARD_BACK_EDGE_SCALE),
        (cv2.inRange(hsv, (0, 0, LIGHT_SLEEVE_MIN_BRIGHTNESS), (180, LIGHT_SLEEVE_MAX_SATURATION, 255)), 1.0),
    ]
    found = []
    for mask, edge_scale in kinds:
        found += _card_shaped_blobs(mask, edge_scale, image_bgr.shape[0], face_up_points, found)
    return found


def _card_shaped_blobs(mask, edge_scale, frame_h, face_up_points, already_found):
    typical = float(np.median([_card_long_side(p) for p in face_up_points])) if face_up_points else None
    taken = list(face_up_points) + list(already_found)
    k = max(3, int((typical or frame_h * 0.24) * 0.06)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for contour in contours:
        (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
        if min(w, h) < 10:
            continue
        long_side, short_side = max(w, h), min(w, h)
        card_long = long_side * edge_scale
        size_ok = (0.55 < card_long / typical < 1.5) if typical else (0.1 < card_long / frame_h < 0.5)
        area = cv2.contourArea(contour)
        # A camera looking down at an angle sees a card as a trapezoid, which
        # fills less of its rectangle - so the fill check is loose, and it's
        # solidity (no big dents - a shadow or a hand has them) that keeps
        # irregular blobs out.
        fill = area / (w * h)
        solidity = area / max(cv2.contourArea(cv2.convexHull(contour)), 1)
        if not (1.0 <= long_side / short_side < 2.2 and fill > 0.75 and solidity > 0.9 and size_ok):
            continue
        if any(cv2.pointPolygonTest(np.float32(p), (cx, cy), False) >= 0 for p in taken):
            continue
        blobs.append(np.float32(cv2.boxPoints(((cx, cy), (w * edge_scale, h * edge_scale), angle))))
    return blobs


# Card backs: how dark (HSV value, 0-255) a pixel has to be to count as the
# swirl, and how much bigger the whole card is than that dark area.
CARD_BACK_MAX_BRIGHTNESS = 90
CARD_BACK_EDGE_SCALE = 1.1
# Light sleeves: how bright (HSV value) and how colorless (HSV saturation,
# 0-255) a pixel has to be. A wooden table sits around saturation 120.
LIGHT_SLEEVE_MIN_BRIGHTNESS = 200
LIGHT_SLEEVE_MAX_SATURATION = 70


def _get_rotation(box_wxhxr, box_txt):
    """Ported from draw2's utils.get_rotation - the text box's position
    within the 224x224 crop tells us which of the 4 possible 90-degree
    readings is the upright one. `angle`'s formula is intentionally kept
    exactly as draw2 computes it (operator precedence makes it (r % pi) / 2,
    not r % (pi / 2)), since the branches below were tuned against that."""
    w, h, r = box_wxhxr[2], box_wxhxr[3], box_wxhxr[4]
    angle = (r % math.pi) / 2

    if min(box_txt[:, 0]) < 112:
        if max(box_txt[:, 0]) < 112:
            if min(box_txt[:, 1]) < 112 < max(box_txt[:, 1]):
                if (h > w and angle > math.pi / 4) or (h < w and angle < math.pi / 4):
                    return cv2.ROTATE_90_COUNTERCLOCKWISE
                return None
            return None
        if min(box_txt[:, 1]) < max(box_txt[:, 1]) < 112:
            if (h > w and angle < math.pi / 4) or (h < w and angle > math.pi / 4):
                return cv2.ROTATE_180
            return None
        if 112 < min(box_txt[:, 1]) < max(box_txt[:, 1]):
            if (h > w and angle < math.pi / 4) or (h < w and angle > math.pi / 4):
                return 0
            return None
        return None
    if min(box_txt[:, 1]) < 112 < max(box_txt[:, 1]):
        if (h > w and angle > math.pi / 4) or (h < w and angle < math.pi / 4):
            return cv2.ROTATE_90_CLOCKWISE
        return None
    return None


class CardDetector:
    """Loads models once; call detect() per frame after that."""

    def __init__(self, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold

        config_path = hf_hub_download(repo_id="HichTala/draw2", filename="draw_config.json")
        yolo_path = hf_hub_download(repo_id="HichTala/draw2", filename="ygo_yolo.pt")
        cardnames_path = hf_hub_download(repo_id="HichTala/draw2", filename="cardnames.json")

        with open(config_path) as f:
            self.config = json.load(f)
        with open(cardnames_path, encoding="utf-8") as f:
            self.cardnames = json.load(f)

        self.yolo = YOLO(yolo_path)

        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
        self.classifier = pipeline(
            "image-classification",
            model="HichTala/draw2",
            image_processor=image_processor,
            device_map=self.device,
        )

    def _classify(self, oriented_rois, allowed_card_ids):
        """Classifies a batch of upright 224x224 crops in one pipeline call
        and returns the best candidate for each (None where nothing passes
        allowed_card_ids)."""
        pil_rois = [Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)) for roi in oriented_rois]
        best = []
        for output in self.classifier(pil_rois, top_k=15, batch_size=len(pil_rois)):
            if allowed_card_ids:
                output = [c for c in output if c["label"].split("-")[-1] in allowed_card_ids]
            best.append(output[0] if output else None)
        return best

    def detect(self, image_bgr, allowed_card_ids=None):
        """
        image_bgr: a single OpenCV-style BGR frame.
        allowed_card_ids: optional set of card id strings to restrict
            matches to, mirroring draw2's deck-list filtering.

        Returns a list of { points, cardId, cardName, confidence }, one per
        detected card-shaped region. A card the classifier couldn't
        confidently identify (most often an actual face-down card, since its
        back looks the same across the whole game) still gets an entry, with
        cardId: None and cardName: "Face-down card", rather than being
        dropped - its position is still real and usable even when its
        identity isn't. `points` are the 4 detected corners in image_bgr's
        own pixel coordinates.
        """
        results = self.yolo.predict(
            source=image_bgr,
            show_labels=False,
            save=False,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].obb is None:
            # No face-up cards - but there can still be face-down ones.
            card_backs = [_face_down_entry(pts) for pts in _find_card_backs(image_bgr, [])]
            if DETECTION_LOGGING:
                print(f"[detector] YOLO found no boxes in this frame; card_backs={len(card_backs)}")
            return card_backs
        result = results[0]
        raw_box_count = len(result.obb.xyxyxyxyn)

        detections = []
        rejected_no_contour = 0
        rejected_low_confidence = 0
        for nbox, box in enumerate(result.obb.xyxyxyxyn):
            pts = np.float32(
                [[p[0] * result.orig_img.shape[1], p[1] * result.orig_img.shape[0]] for p in box.cpu()]
            )

            dst = np.float32([[224, 224], [224, 0], [0, 0], [0, 224]])
            transform = cv2.getPerspectiveTransform(pts, dst)
            roi = cv2.warpPerspective(image_bgr, transform, (224, 224), flags=cv2.INTER_LINEAR)

            contours = _extract_contours(
                roi,
                d=self.config["bilateral_filter_d"],
                sigma_color=self.config["bilateral_filter_sigma_color"],
                sigma_space=self.config["bilateral_filter_sigma_space"],
                thresh=self.config["txt_box_contour_threshold"],
            )
            if not contours:
                rejected_no_contour += 1
                detections.append(_face_down_entry(pts))
                continue

            contour = max(contours, key=cv2.contourArea)
            box_txt = _get_txt_box(contour)
            rotation = _get_rotation(result.obb.xywhr[nbox], box_txt)

            if rotation is not None:
                oriented = roi if rotation == 0 else cv2.rotate(roi, rotation)
                best = self._classify([oriented], allowed_card_ids)[0]
            else:
                # draw2's own rotation heuristic couldn't decide - rather
                # than dropping this card (and, in the real Draw.process(),
                # every OTHER card in the frame too), try all 4 orientations
                # (as one batch) and keep whichever the classifier is most
                # confident about.
                candidates = self._classify(
                    [
                        roi,
                        cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE),
                        cv2.rotate(roi, cv2.ROTATE_180),
                        cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE),
                    ],
                    allowed_card_ids,
                )
                candidates = [c for c in candidates if c is not None]
                best = max(candidates, key=lambda c: c["score"]) if candidates else None

            if best is None or best["score"] < self.confidence_threshold / 100:
                rejected_low_confidence += 1
                detections.append(_face_down_entry(pts, best["score"] if best else 0.0))
                continue

            card_id = best["label"].split("-")[-1]
            card_name = self.cardnames.get(card_id, {}).get("EN") or "-".join(best["label"].split("-")[:-1])

            detections.append(
                {
                    "points": pts.tolist(),
                    "cardId": card_id,
                    "cardName": card_name,
                    "confidence": best["score"],
                }
            )

        card_backs = _find_card_backs(image_bgr, [d["points"] for d in detections])
        detections.extend(_face_down_entry(pts) for pts in card_backs)

        if DETECTION_LOGGING:
            names = ", ".join(d["cardName"] for d in detections) or "none"
            print(
                f"[detector] {raw_box_count} raw box(es) -> {len(detections)} detection(s): {names} "
                f"(unidentified/face-down: no_contour={rejected_no_contour}, low_confidence={rejected_low_confidence}, "
                f"card_backs={len(card_backs)})"
            )

        return detections
