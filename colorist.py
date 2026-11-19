"""colorist.py - the grading engine: dials, looks, guardrails and sessions."""
import os, json, colorsys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# ---------- Loading and viewing ----------
FOLDER = ""

def set_folder(path):
    # Tells the engine where your photos live, and returns the list of photo names
    global FOLDER
    FOLDER = path
    return [f for f in sorted(os.listdir(path))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

def load_image(name):
    bgr = cv2.imread(os.path.join(FOLDER, name))
    if bgr is None:
        raise FileNotFoundError("Cannot open " + name)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def small(img, max_side=800):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def show_many(images, titles):
    plt.figure(figsize=(16, 4))
    for i, (im, t) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(np.clip(im, 0, 1))
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def split_view(before, after, split):
    h, w = before.shape[:2]
    x = int(w * split)
    out = after.copy()
    out[:, :x] = before[:, :x]
    line = max(1, w // 400)
    out[:, max(0, x - line):x + line] = 1.0
    return out

# ---------- Light and colour helpers ----------
def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)

def luma(img):
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])[..., None]

# ---------- The seven basic dials ----------
def exposure(img, stops):
    lin = srgb_to_linear(img) * (2.0 ** stops)
    return linear_to_srgb(lin)

def contrast(img, amount):
    if amount >= 0:
        s = img * img * (3 - 2 * img)
        out = img + amount * (s - img)
    else:
        out = (img - 0.5) * (1 + amount) + 0.5
    return np.clip(out, 0, 1)

def saturation(img, amount):
    gray = luma(img)
    return np.clip(gray + (img - gray) * (1 + amount), 0, 1)

def temperature(img, amount):
    out = img.copy()
    out[..., 0] *= 1 + 0.15 * amount
    out[..., 2] *= 1 - 0.15 * amount
    return np.clip(out, 0, 1)

def tint(img, amount):
    out = img.copy()
    out[..., 1] *= 1 - 0.15 * amount
    return np.clip(out, 0, 1)

def shadows(img, amount):
    weight = (1 - luma(img)) ** 2
    return np.clip(img * (1 + amount * weight), 0, 1)

def highlights(img, amount):
    weight = luma(img) ** 2
    return np.clip(img * (1 + amount * weight), 0, 1)

# ---------- Pro tools: curves, fade, colour wheels, HSL ----------
def curve(img, points):
    pts = sorted((float(x), float(y)) for x, y in points)
    if pts[0][0] > 0:
        pts.insert(0, (0.0, 0.0))
    if pts[-1][0] < 1:
        pts.append((1.0, 1.0))
    f = PchipInterpolator([p[0] for p in pts], [p[1] for p in pts])
    lut_x = np.linspace(0, 1, 1024)
    lut_y = np.clip(f(lut_x), 0, 1)
    return np.interp(np.clip(img, 0, 1), lut_x, lut_y).astype(np.float32)

def fade(img, amount):
    lift = 0.10 * amount
    return np.clip(lift + img * (1 - lift), 0, 1)

def hue_to_offset(hue_deg, strength):
    r, g, b = colorsys.hsv_to_rgb((hue_deg % 360) / 360.0, 1.0, 1.0)
    v = np.array([r, g, b], dtype=np.float32)
    v = v - v.mean()
    return v * float(np.clip(strength, 0, 1)) * 0.15

def color_wheels(img, shadows=(0, 0), mids=(0, 0), highlights=(0, 0)):
    l = luma(img)
    w_shadow = (1 - l) ** 2
    w_high = l ** 2
    w_mid = 1 - w_shadow - w_high
    out = (img + w_shadow * hue_to_offset(*shadows)
               + w_mid * hue_to_offset(*mids)
               + w_high * hue_to_offset(*highlights))
    return np.clip(out, 0, 1)

HUES = {"red": 0, "orange": 30, "yellow": 60, "green": 120,
        "aqua": 180, "blue": 240, "purple": 275, "magenta": 320}

def hsl(img, color, hue_shift=0, sat=0, lum=0, width=40):
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    d = np.abs(H - HUES[color])
    d = np.minimum(d, 360 - d)
    w = np.where(d < width, 0.5 * (1 + np.cos(np.pi * d / width)), 0.0) * S
    H = (H + hue_shift * w) % 360
    S = np.clip(S * (1 + sat * w), 0, 1)
    V = np.clip(V * (1 + lum * w * 0.5), 0, 1)
    out = cv2.cvtColor(np.stack([H, S, V], axis=-1).astype(np.float32), cv2.COLOR_HSV2RGB)
    return np.clip(out, 0, 1)

# ---------- The control panel: grade() ----------
def grade(img, exposure_stops=0, contrast_amt=0, saturation_amt=0,
          temp=0, tint_amt=0, shadows_amt=0, highlights_amt=0,
          curve_pts=None, wheels=None, hsl_edits=None, fade_amt=0):
    clamp = lambda v, lo, hi: float(np.clip(v, lo, hi))
    out = exposure(img, clamp(exposure_stops, -3, 3))
    out = temperature(out, clamp(temp, -1, 1))
    out = tint(out, clamp(tint_amt, -1, 1))
    out = contrast(out, clamp(contrast_amt, -1, 1))
    if curve_pts:
        out = curve(out, curve_pts)
    out = shadows(out, clamp(shadows_amt, -1, 1))
    out = highlights(out, clamp(highlights_amt, -1, 1))
    if wheels:
        out = color_wheels(out, **wheels)
    if hsl_edits:
        for e in hsl_edits:
            out = hsl(out, e["color"], clamp(e.get("hue", 0), -30, 30),
                      clamp(e.get("sat", 0), -1, 1), clamp(e.get("lum", 0), -1, 1))
    out = saturation(out, clamp(saturation_amt, -1, 1))
    out = fade(out, clamp(fade_amt, 0, 1))
    return out

# ---------- Looks (presets) ----------
PRESETS = {
    "teal_orange": dict(contrast_amt=0.25, saturation_amt=0.05,
                        wheels={"shadows": [195, 0.6], "highlights": [30, 0.5]}),
    "film_fade":   dict(contrast_amt=-0.1, saturation_amt=-0.15, fade_amt=0.5, temp=0.15,
                        curve_pts=[(0, 0), (0.25, 0.22), (0.75, 0.78), (1, 1)]),
    "warm_golden": dict(exposure_stops=0.15, temp=0.45, tint_amt=0.05,
                        shadows_amt=0.2, highlights_amt=-0.2,
                        wheels={"highlights": [40, 0.4]}),
    "cool_moody":  dict(exposure_stops=-0.3, contrast_amt=0.3, saturation_amt=-0.2,
                        temp=-0.5, wheels={"shadows": [215, 0.5]}),
}

def apply_preset(img, name, strength=1.0):
    graded = grade(img, **PRESETS[name])
    return np.clip(img + strength * (graded - img), 0, 1)

# ---------- Guardrails: skin protection and clipping safety ----------
def skin_mask(img):
    rgb8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ycc = cv2.cvtColor(rgb8, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    m = ((Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127) & (Y > 40)).astype(np.float32)
    k = max(3, (min(m.shape) // 50) | 1)
    m = cv2.GaussianBlur(m, (k, k), 0)
    return m[..., None]

def protect_skin(before, after, amount=0.6):
    if amount <= 0:
        return after
    m = skin_mask(before)
    if m.mean() < 0.01:
        return after
    b = cv2.cvtColor(np.clip(before, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
    a = cv2.cvtColor(np.clip(after, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
    out = a.copy()
    w = m[..., 0] * amount
    out[..., 1] = a[..., 1] * (1 - w) + b[..., 1] * w
    out[..., 2] = a[..., 2] * (1 - w) + b[..., 2] * w
    return np.clip(cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB), 0, 1)

def clip_stats(img):
    mx = np.max(img, axis=2)
    return float((mx >= 0.99).mean()), float((mx <= 0.02).mean())

def safe_finish(before, full, strength=1.0, skin_protect=0.6, max_new_clip=0.02):
    hi0, lo0 = clip_stats(before)
    start = float(np.clip(strength, 0, 1))
    blend = lambda s: np.clip(before + s * (full - before), 0, 1)
    s = start
    out = blend(s)
    tries = 0
    while tries < 6:
        hi, lo = clip_stats(out)
        if hi - hi0 <= max_new_clip and lo - lo0 <= max_new_clip:
            break
        s = max(0.0, s - 0.15)
        out = blend(s)
        tries += 1
    out = protect_skin(before, out, skin_protect)
    hi, lo = clip_stats(out)
    notes = []
    if s < start:
        notes.append(f"Eased the grade to {s:.0%} strength to protect detail")
    if hi - hi0 > 0.005:
        notes.append(f"{(hi - hi0):.1%} of pixels are newly burnt white")
    if lo - lo0 > 0.005:
        notes.append(f"{(lo - lo0):.1%} of pixels are newly crushed black")
    return out, dict(strength=s, new_highlight_clip=hi - hi0,
                     new_shadow_clip=lo - lo0, notes=notes)

# ---------- Sessions: stacked commands, undo, export ----------
def build_edit(img, plan):
    out = img
    if plan["preset"]:
        out = apply_preset(out, plan["preset"], plan["preset_strength"])
    return grade(out, **plan["settings"])

class Session:
    def __init__(self, original, brain):
        self.original = original
        self.brain = brain              # any function: brain(command, image) -> plan
        self.history = []

    @property
    def current(self):
        return self.history[-1]["image"] if self.history else self.original

    def apply(self, command, strength=1.0, skin_protect=0.6):
        base = self.current
        plan = self.brain(command, base)
        full = build_edit(base, plan)
        out, report = safe_finish(base, full, strength, skin_protect)
        self.history.append(dict(command=command, plan=plan, report=report, image=out,
                                 strength=strength, skin=skin_protect))
        return plan, report

    def undo(self):
        if self.history:
            self.history.pop()

    def reset(self):
        self.history.clear()

def export_full(session, photo_name, out_dir=None):
    if out_dir is None:
        out_dir = "/content/drive/MyDrive/ai-colorist/outputs" if os.path.exists("/content") else "./outputs"
    os.makedirs(out_dir, exist_ok=True)
    full = load_image(photo_name)
    for step in session.history:
        edit = build_edit(full, step["plan"])
        full, _ = safe_finish(full, edit, step["strength"], step["skin"])
    rgb8 = (np.clip(full, 0, 1) * 255).astype(np.uint8)
    name = "graded_" + photo_name
    cv2.imwrite(os.path.join(out_dir, name), cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    recipe = [dict(command=s["command"], plan=s["plan"],
                   strength=s["strength"], skin_protect=s["skin"]) for s in session.history]
    recipe_name = "recipe_" + os.path.splitext(photo_name)[0] + ".json"
    with open(os.path.join(out_dir, recipe_name), "w") as f:
        json.dump(recipe, f, indent=2)
    return name, recipe_name
