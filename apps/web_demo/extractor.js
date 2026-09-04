/**
 * Client-side feature extractor.
 *
 * Mirrors the Python path in neuroproxy/vision so that everything measured
 * offline still describes what a browser session produces. The equivalence is
 * checked, not assumed -- see tests/test_client_extraction.py.
 *
 * Per frame it emits ~100 bytes instead of an image: the spatial mean RGB over
 * the skin ROIs, plus the quality scalars the confidence gate needs. Raw pixels
 * never leave the page.
 */

// Fallback skin locus, used ONLY to find a face when no detector is available.
// Never to select ROI pixels -- a fixed locus excluded a dark-skinned subject
// entirely in testing (docs/limitations.md 12).
const CR = [133, 177], CB = [77, 127];

// Adaptive mask parameters, matching vision/detector.py.
const ADAPTIVE_K = 3.0, MIN_RADIUS = 6.0, MAX_RADIUS = 28.0, LUMA_TOL = 0.55;

// ROI boxes as fractions of the face box, matching vision/roi.py.
const ROI = {
  forehead:    [0.28, 0.12, 0.72, 0.28],
  left_cheek:  [0.14, 0.52, 0.40, 0.76],
  right_cheek: [0.60, 0.52, 0.86, 0.76],
};
const MIN_SKIN_FRACTION = 0.35;
const GOOD_LUMA = [60, 200], BLUR_FLOOR = 8.0, MOTION_TOLERANCE = 0.02;

const clamp01 = v => Math.max(0, Math.min(1, v));

function ycrcb(r, g, b) {
  const y = 0.299 * r + 0.587 * g + 0.114 * b;
  return [y, (r - y) * 0.713 + 128, (b - y) * 0.564 + 128];
}

/** Largest skin-coloured blob, as a bounding box. Mirrors the `skin` backend. */
function detectByColour(data, w, h) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1, n = 0;
  for (let y = 0; y < h; y += 2) {
    for (let x = 0; x < w; x += 2) {
      const i = (y * w + x) * 4;
      const [, cr, cb] = ycrcb(data[i], data[i + 1], data[i + 2]);
      if (cr >= CR[0] && cr <= CR[1] && cb >= CB[0] && cb <= CB[1]) {
        n++; if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
  }
  if (n < 64 || x1 < 0) return null;
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0, confidence: 0.6 };
}

/**
 * Skin mask from the face's OWN chroma, not a fixed locus.
 * Returns a predicate over pixel index.
 */
function adaptiveSkin(data, w, h, face) {
  const ix0 = Math.max(0, Math.round(face.x + 0.25 * face.w));
  const ix1 = Math.min(w, Math.round(face.x + 0.75 * face.w));
  const iy0 = Math.max(0, Math.round(face.y + 0.35 * face.h));
  const iy1 = Math.min(h, Math.round(face.y + 0.80 * face.h));
  const ys = [], crs = [], cbs = [];
  for (let y = iy0; y < iy1; y += 2) {
    for (let x = ix0; x < ix1; x += 2) {
      const i = (y * w + x) * 4;
      const v = ycrcb(data[i], data[i + 1], data[i + 2]);
      ys.push(v[0]); crs.push(v[1]); cbs.push(v[2]);
    }
  }
  if (!ys.length) return null;
  const med = a => { const s = [...a].sort((p, q) => p - q); return s[s.length >> 1]; };
  const lumaC = med(ys), crC = med(crs), cbC = med(cbs);
  const spread = med(crs.map((c, k) => Math.abs(c - crC) + Math.abs(cbs[k] - cbC)));
  const radius = Math.max(MIN_RADIUS, Math.min(MAX_RADIUS, ADAPTIVE_K * spread));
  const lumaTol = Math.max(LUMA_TOL * lumaC, 25);
  return (r, g, b) => {
    const [y, cr, cb] = ycrcb(r, g, b);
    return Math.abs(cr - crC) + Math.abs(cb - cbC) <= radius &&
           Math.abs(y - lumaC) <= lumaTol;
  };
}

export function extract(imageData, prev) {
  const { data, width: w, height: h } = imageData;
  // Detect on every frame, as the Python path does (test_client_extraction:
  // `detector.detect(frame)` per frame). An earlier version reused
  // `prev.face` to skip detection, but that field is the *confidence scalar*,
  // not the box -- so every second frame was measured against the number 0.6,
  // produced NaN bounds and was dropped as invalid. `prev` is only for motion.
  const face = detectByColour(data, w, h);
  const empty = { rgb: null, valid: false, face: 0, lighting: 0, sharpness: 0,
                  motion: 1, skin_fraction: 0, compression: 1, _face: null };
  if (!face) return empty;

  const isSkin = adaptiveSkin(data, w, h, face);
  if (!isSkin) return empty;

  let sum = [0, 0, 0], nSkin = 0, nBox = 0, accepted = 0;
  for (const key of Object.keys(ROI)) {
    const [fx0, fy0, fx1, fy1] = ROI[key];
    const x0 = Math.max(0, Math.round(face.x + fx0 * face.w));
    const x1 = Math.min(w, Math.round(face.x + fx1 * face.w));
    const y0 = Math.max(0, Math.round(face.y + fy0 * face.h));
    const y1 = Math.min(h, Math.round(face.y + fy1 * face.h));
    if (x1 - x0 < 2 || y1 - y0 < 2) continue;
    let s = [0, 0, 0], hit = 0, tot = 0;
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const i = (y * w + x) * 4; tot++;
        if (!isSkin(data[i], data[i + 1], data[i + 2])) continue;
        s[0] += data[i]; s[1] += data[i + 1]; s[2] += data[i + 2]; hit++;
      }
    }
    nBox += tot;
    if (tot === 0 || hit / tot < MIN_SKIN_FRACTION) continue;
    sum[0] += s[0]; sum[1] += s[1]; sum[2] += s[2]; nSkin += hit; accepted++;
  }
  if (!accepted || !nSkin) return { ...empty, _face: face };

  // Exposure and focus are judged on the face, not the frame: a bright or busy
  // backdrop is not an exposure fault of the subject (limitations 12).
  let lumaSum = 0, clipped = 0, px = 0, lap = 0, lapN = 0, lapSum = 0;
  const fx0 = Math.max(0, face.x | 0), fx1 = Math.min(w, (face.x + face.w) | 0);
  const fy0 = Math.max(0, face.y | 0), fy1 = Math.min(h, (face.y + face.h) | 0);
  const grey = [];
  for (let y = fy0; y < fy1; y++) {
    for (let x = fx0; x < fx1; x++) {
      const i = (y * w + x) * 4;
      const g = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      grey.push(g); lumaSum += g; px++;
      if (g <= 2 || g >= 253) clipped++;
    }
  }
  const mean = px ? lumaSum / px : 0, clipFrac = px ? clipped / px : 0;
  let lighting = mean < GOOD_LUMA[0] ? mean / GOOD_LUMA[0]
    : mean > GOOD_LUMA[1] ? Math.max(0, 1 - (mean - GOOD_LUMA[1]) / (255 - GOOD_LUMA[1])) : 1;
  lighting = clamp01(lighting * (1 - clipFrac));

  const fw = fx1 - fx0;
  for (let k = fw + 1; k < grey.length - fw - 1; k++) {
    const v = grey[k - 1] + grey[k + 1] + grey[k - fw] + grey[k + fw] - 4 * grey[k];
    lap += v; lapSum += v * v; lapN++;
  }
  const variance = lapN ? lapSum / lapN - (lap / lapN) ** 2 : 0;
  const sharpness = clamp01(variance / (variance + BLUR_FLOOR));

  let motion = 1;
  if (prev && prev._face) {
    const dx = (face.x + face.w / 2) - (prev._face.x + prev._face.w / 2);
    const dy = (face.y + face.h / 2) - (prev._face.y + prev._face.h / 2);
    const rel = Math.hypot(dx, dy) / Math.max(face.w, 1);
    motion = clamp01(1 - rel / (MOTION_TOLERANCE * 4));
  }

  return {
    rgb: [sum[0] / nSkin, sum[1] / nSkin, sum[2] / nSkin],
    valid: true,
    face: face.confidence,
    lighting, sharpness, motion,
    skin_fraction: nBox ? nSkin / nBox : 0,
    // Canvas pixels are pre-encoding, so there is no codec damage to detect.
    compression: 1.0,
    _face: face,
  };
}

/** Strip client-only fields before sending. */
export function wire(f) {
  const { _face, ...rest } = f;
  return rest;
}
