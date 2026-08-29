"""Skin ROI selection and spatial averaging.

rPPG needs a spatial mean over well-perfused, weakly-shadowed skin. Forehead
and cheeks are the standard choice: they are flat, hairless in most subjects,
and move rigidly with the head.

Every ROI is intersected with a skin mask, so background pixels that leak into
a rectangular box are excluded instead of diluting the ~1% pulse modulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import cv2

from .detector import FaceBox, skin_mask

# Fractional boxes within the face bounding box: (x0, y0, x1, y1).
ROI_BOXES: Dict[str, Tuple[float, float, float, float]] = {
    "forehead": (0.28, 0.12, 0.72, 0.28),
    "left_cheek": (0.14, 0.52, 0.40, 0.76),
    "right_cheek": (0.60, 0.52, 0.86, 0.76),
}

# Forehead band measured downward from the detected hairline, as a fraction of
# face height. Applied ONLY to the forehead: the cheeks keep their placement
# relative to the detector box.
#
# Shifting every ROI by one hairline offset was tried and measured worse
# (docs/limitations.md section 17) -- the cheeks slid onto the jaw and beard.
# The forehead is the ROI whose correct position depends on the hairline; the
# cheeks' does not.
FOREHEAD_BELOW_HAIRLINE = (0.06, 0.24)
FOREHEAD_X = (0.28, 0.72)

# Below this fraction of skin pixels a ROI is discarded (occlusion, hair, edge).
MIN_SKIN_FRACTION = 0.35


def skin_anchor(
    mask: np.ndarray, face: FaceBox, min_fill: float = 0.15
) -> Optional[FaceBox]:
    """Tighten the face box onto the actual skin region inside it.

    WHY ROIs CANNOT BE FIXED FRACTIONS OF THE DETECTOR BOX
    ------------------------------------------------------
    A Haar box frames the head, not the face: it routinely includes hair and
    some background, and its exact framing varies between subjects. When the
    box runs high or loose, the "forehead" ROI at a fixed 12-28% of box height
    lands on the hairline instead of the forehead.

    That was the actual cause of the coverage problem, found by rendering the
    mask over failing subjects rather than by reasoning about it -- three
    colour-based hypotheses (face distance, directional shadow, brightness
    dependence of the chroma criterion) were tested and all three were wrong.
    The failing subjects' forehead ROIs were simply sitting on hair.

    Anchoring to the skin region makes ROI placement independent of how the
    detector happened to frame the head. Returns None when the box contains no
    coherent skin region, which is itself a useful signal.
    """
    h, w = mask.shape[:2]
    x0, y0 = max(int(face.x), 0), max(int(face.y), 0)
    x1, y1 = min(int(face.x + face.w), w), min(int(face.y + face.h), h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    sub = mask[y0:y1, x0:x1].astype(np.uint8)
    if sub.mean() < min_fill:
        return None

    num, labels, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    if num <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sx, sy, sw, sh, area = stats[idx]
    if sw < 8 or sh < 8 or area < 0.10 * sub.size:
        return None
    return FaceBox(
        x=x0 + int(sx), y=y0 + int(sy), w=int(sw), h=int(sh),
        confidence=face.confidence, backend=face.backend + "+skin",
    )


def hairline_row(
    mask: np.ndarray, face: FaceBox, min_row_fill: float = 0.6
) -> Optional[int]:
    """Find the first row, scanning down the face, that is mostly skin.

    A bounding box around the skin region does not solve the forehead problem:
    the largest connected skin component usually reaches down into the neck, so
    its top edge still sits at the hairline. What the forehead ROI needs is the
    hairline itself.

    Scans the central 40% of the face box top-down and returns the first row
    where that band is predominantly skin. Everything above it is hair, or the
    detector box overshooting the head.
    """
    h, w = mask.shape[:2]
    cx0 = max(int(face.x + 0.30 * face.w), 0)
    cx1 = min(int(face.x + 0.70 * face.w), w)
    y0 = max(int(face.y), 0)
    y1 = min(int(face.y + 0.65 * face.h), h)
    if cx1 - cx0 < 4 or y1 - y0 < 4:
        return None
    band = mask[y0:y1, cx0:cx1]
    fills = band.mean(axis=1)
    hits = np.flatnonzero(fills >= min_row_fill)
    if hits.size == 0:
        return None
    return int(y0 + hits[0])


def forehead_from_hairline(
    mask: np.ndarray, face: FaceBox
) -> Optional[Tuple[float, float, float, float]]:
    """Forehead box as a band below the detected hairline, in face fractions.

    Returns None when no hairline is found or the resulting band would fall
    outside the face, in which case the caller keeps the default box.
    """
    hair = hairline_row(mask, face)
    if hair is None or face.h <= 0:
        return None
    top = (hair - face.y) / float(face.h) + FOREHEAD_BELOW_HAIRLINE[0]
    bottom = (hair - face.y) / float(face.h) + FOREHEAD_BELOW_HAIRLINE[1]
    # Never run into the eyes, and never sit above the detector box.
    if top < 0.0 or bottom > 0.45:
        return None
    return (FOREHEAD_X[0], top, FOREHEAD_X[1], bottom)


@dataclass
class ROISample:
    """One frame's worth of ROI colour, plus how trustworthy it was."""

    rgb: np.ndarray            # (3,) spatial mean over accepted skin pixels
    skin_fraction: float       # fraction of ROI pixels accepted as skin
    n_pixels: int
    per_roi: Dict[str, np.ndarray]


def extract(
    frame_rgb: np.ndarray,
    face: FaceBox,
    mask: Optional[np.ndarray] = None,
    roi_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    anchor_forehead: bool = True,
) -> Optional[ROISample]:
    """Spatial-mean RGB over the union of accepted skin ROIs, or None.

    `anchor_forehead` places the forehead ROI relative to the detected hairline
    rather than as a fixed fraction of the detector box. The cheeks are left
    alone deliberately: an earlier attempt shifted every ROI by the same
    hairline offset and measured worse on every axis, because the cheeks slid
    onto the jaw and beard (docs/limitations.md section 17).
    """
    boxes = ROI_BOXES if roi_boxes is None else roi_boxes
    if mask is None:
        mask = skin_mask(frame_rgb)
    if anchor_forehead and "forehead" in boxes:
        band = forehead_from_hairline(mask, face)
        if band is not None:
            boxes = dict(boxes)
            boxes["forehead"] = band

    sums = np.zeros(3, dtype=np.float64)
    total = 0
    box_pixels = 0
    accepted = 0
    per_roi: Dict[str, np.ndarray] = {}

    for name, (fx0, fy0, fx1, fy1) in boxes.items():
        x0 = int(face.x + fx0 * face.w)
        x1 = int(face.x + fx1 * face.w)
        y0 = int(face.y + fy0 * face.h)
        y1 = int(face.y + fy1 * face.h)
        x0, x1 = max(x0, 0), min(x1, frame_rgb.shape[1])
        y0, y1 = max(y0, 0), min(y1, frame_rgb.shape[0])
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        patch = frame_rgb[y0:y1, x0:x1].astype(np.float64)
        patch_mask = mask[y0:y1, x0:x1]
        n_total = patch_mask.size
        n_skin = int(patch_mask.sum())
        box_pixels += n_total
        if n_total == 0 or n_skin / n_total < MIN_SKIN_FRACTION:
            continue
        sel = patch[patch_mask]
        per_roi[name] = sel.mean(axis=0)
        sums += sel.sum(axis=0)
        total += n_skin
        accepted += 1

    if accepted == 0 or total == 0:
        return None

    return ROISample(
        rgb=sums / total,
        skin_fraction=float(total / max(box_pixels, 1)),
        n_pixels=total,
        per_roi=per_roi,
    )
