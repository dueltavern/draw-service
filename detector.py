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
            source=image_bgr, show_labels=False, save=False, device=self.device, verbose=False
        )
        if not results or results[0].obb is None:
            if DETECTION_LOGGING:
                print("[detector] YOLO found no boxes in this frame")
            return []
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

        if DETECTION_LOGGING:
            names = ", ".join(d["cardName"] for d in detections) or "none"
            print(
                f"[detector] {raw_box_count} raw box(es) -> {len(detections)} detection(s): {names} "
                f"(unidentified/face-down: no_contour={rejected_no_contour}, low_confidence={rejected_low_confidence})"
            )

        return detections
