"""masks.py - region editing with classic masks (brightness, colour, gradient, vignette)."""
import numpy as np
import cv2
import colorist
import brain as _brain
from colorist import luma, grade, HUES

# ---------- Building masks (white = edit here, black = leave alone) ----------
def smoothstep(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)

def soft(mask, amount=0.01):
    # Blurs the edges of a mask a little, so there are no hard cut-out lines
    h, w = mask.shape[:2]
    k = max(3, int(min(h, w) * amount) | 1)
    return cv2.GaussianBlur(mask[..., 0].astype(np.float32), (k, k), 0)[..., None]

LUM_RANGES = {"shadows": (None, 0.3), "midtones": (0.3, 0.7), "highlights": (0.7, None)}

def luminance_mask(img, name, f=0.15):
    lo, hi = LUM_RANGES[name]
    l = luma(img)[..., 0]
    m = np.ones_like(l)
    if lo is not None:
        m = m * smoothstep(lo - f, lo + f, l)
    if hi is not None:
        m = m * (1 - smoothstep(hi - f, hi + f, l))
    return soft(m[..., None], 0.01)

def color_mask(img, color, width=40):
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    d = np.abs(H - HUES[color])
    d = np.minimum(d, 360 - d)
    w = np.where(d < width, 0.5 * (1 + np.cos(np.pi * d / width)), 0.0)
    w = w * smoothstep(0.08, 0.30, S) * smoothstep(0.05, 0.20, V)   # ignore grey and near-black
    return soft(w[..., None].astype(np.float32), 0.01)

def gradient_mask(shape, side, extent=0.6):
    h, w = shape[:2]
    if side in ("top", "bottom"):
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
        if side == "bottom":
            t = 1 - t
    else:
        t = np.linspace(0, 1, w, dtype=np.float32)[None, :] * np.ones((h, 1), np.float32)
        if side == "right":
            t = 1 - t
    return (1 - smoothstep(0.0, extent, t))[..., None].astype(np.float32)

def vignette_mask(shape):
    h, w = shape[:2]
    ys = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    r = np.sqrt(xs ** 2 + ys ** 2) / np.sqrt(2)
    return smoothstep(0.35, 1.0, r)[..., None].astype(np.float32)

def make_mask(img, spec):
    t = spec["type"]
    if t == "luminance":
        m = luminance_mask(img, spec["range"])
    elif t == "color":
        m = color_mask(img, spec["color"])
    elif t == "gradient":
        m = gradient_mask(img.shape, spec["side"], spec.get("extent", 0.6))
    elif t == "vignette":
        m = vignette_mask(img.shape)
    else:
        raise ValueError("unknown mask type: " + str(t))
    if spec.get("invert"):
        m = 1 - m
    return m

def mask_overlay(img, mask, color=(1.0, 0.2, 0.2), alpha=0.55):
    # Paints the masked area red, so you can SEE what the mask selects
    c = np.array(color, dtype=np.float32)
    return np.clip(img * (1 - alpha * mask) + c * alpha * mask, 0, 1)

# ---------- Checking what the brain asks for ----------
def clean_mask(spec):
    if not isinstance(spec, dict):
        return None
    t = spec.get("type")
    out = {"type": t}
    if t == "luminance":
        if spec.get("range") not in LUM_RANGES:
            return None
        out["range"] = spec["range"]
    elif t == "color":
        if spec.get("color") not in HUES:
            return None
        out["color"] = spec["color"]
    elif t == "gradient":
        if spec.get("side") not in ("top", "bottom", "left", "right"):
            return None
        out["side"] = spec["side"]
        out["extent"] = _brain.num(spec.get("extent", 0.6), 0.2, 1.0, 0.6)
    elif t == "vignette":
        pass
    else:
        return None
    if spec.get("invert") is True:
        out["invert"] = True
    return out

def clean_regions(raw_regions):
    regions = []
    if not isinstance(raw_regions, list):
        return regions
    for r in raw_regions[:4]:
        if not isinstance(r, dict):
            continue
        mask = clean_mask(r.get("mask"))
        settings = _brain._orig_clean_plan({"settings": r.get("settings", {})})["settings"]
        if mask and settings:
            regions.append(dict(mask=mask, settings=settings))
    return regions

# ---------- Plugging masks into the engine and the brain ----------
MASK_RULES = """
8. Regions: when the user asks to change only part of the photo, add a top-level key "regions" next to "settings": a list of up to 4 items shaped like {"mask": {...}, "settings": {...}}. "settings" uses the same keys and ranges as above, and "mask" is exactly one of:
- {"type": "luminance", "range": "shadows" or "midtones" or "highlights"}  (picks pixels by brightness)
- {"type": "color", "color": one of red/orange/yellow/green/aqua/blue/purple/magenta}  (picks pixels of that colour)
- {"type": "gradient", "side": "top" or "bottom" or "left" or "right", "extent": 0.2 to 1.0}  (a soft fade from one edge)
- {"type": "vignette"}  (the edges of the frame)
Add "invert": true inside the mask to flip it. Use regions only when the request names a brightness range, a colour, an edge, or the frame edges, for example "cooler shadows", "richer greens", "darken the top", "add a vignette". Put changes for the whole photo in "settings" and changes for a region in "regions". Rule 6 is updated: sky, people, faces and objects are still NOT available. For those, apply a mild global version (or a colour region only if it clearly fits) and say so in the explanation.
Example request: "cooler shadows and warmer highlights". Example reply: {"settings": {}, "regions": [{"mask": {"type": "luminance", "range": "shadows"}, "settings": {"temp": -0.3}}, {"mask": {"type": "luminance", "range": "highlights"}, "settings": {"temp": 0.3}}], "preset": null, "preset_strength": 1.0, "explanation": "Cooled the shadows and warmed the highlights."}
"""

def build_edit_v2(img, plan):
    out = colorist._orig_build_edit(img, plan)           # the global edit first
    for region in plan.get("regions", []):               # then each regional edit
        m = make_mask(img, region["mask"])
        graded = grade(out, **region["settings"])
        out = out + m * (graded - out)
    return np.clip(out, 0, 1)

def clean_plan_v2(raw):
    plan = _brain._orig_clean_plan(raw)
    plan["regions"] = clean_regions(raw.get("regions"))
    return plan

def install():
    # Safe to run more than once
    if not hasattr(colorist, "_orig_build_edit"):
        colorist._orig_build_edit = colorist.build_edit
    if not hasattr(_brain, "_orig_clean_plan"):
        _brain._orig_clean_plan = _brain.clean_plan
        _brain._orig_template = _brain.SYSTEM_TEMPLATE
    colorist.build_edit = build_edit_v2
    _brain.clean_plan = clean_plan_v2
    _brain.SYSTEM_TEMPLATE = _brain._orig_template + MASK_RULES
    print("Mask support installed.")
