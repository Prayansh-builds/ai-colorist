"""brain.py - turns a typed command into grading settings using the Gemini API."""
import json, time
import numpy as np
from google import genai
from google.genai import types
from colorist import luma, PRESETS, HUES

MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-3.8-flash", "gemini-3.7-flash"]

SYSTEM_TEMPLATE = """You are the brain of an AI colour-grading tool. The user types a request; you reply with a JSON plan that a program applies to the photo. Reply with JSON only: no words before or after it, no code fences.

Format:
{"settings": {...}, "preset": null, "preset_strength": 1.0, "explanation": "one short sentence"}

"settings" may contain only these keys (leave out any you do not need):
- exposure_stops: -3 to 3 (+0.3 is a small brightening)
- contrast_amt: -1 to 1
- saturation_amt: -1 to 1
- temp: -1 to 1 (positive = warmer/orange, negative = cooler/blue)
- tint_amt: -1 to 1 (positive = magenta, negative = green)
- shadows_amt: -1 to 1 (positive lifts dark areas)
- highlights_amt: -1 to 1 (negative recovers bright areas)
- fade_amt: 0 to 1 (lifted blacks, film look)
- curve_pts: list of [input, output] pairs from 0 to 1, e.g. [[0,0],[0.25,0.2],[0.75,0.8],[1,1]]
- wheels: {"shadows": [hue_degrees, strength], "mids": [hue_degrees, strength], "highlights": [hue_degrees, strength]}, strength 0 to 1 (use 0.2 to 0.5)
- hsl_edits: list of {"color": "red|orange|yellow|green|aqua|blue|purple|magenta", "hue": -30 to 30, "sat": -1 to 1, "lum": -1 to 1}

Hue angles: 0 red, 30 orange, 60 yellow, 120 green, 190 teal, 240 blue, 300 magenta.

"preset" is null or one of: PRESET_LIST. Use a preset only when the user asks for that style. preset_strength is 0 to 1.

Rules for good taste:
1. Be subtle. Typical values are 0.1 to 0.35. "A little" = 0.1 to 0.15, normal = 0.2 to 0.35, "a lot" or "very" = 0.4 to 0.6. Never go above 0.6 unless the user clearly asks for an extreme look.
2. Use the photo stats. If the median brightness is above 0.6, brighten less. If it is below 0.35, brighten more. Do not push highlights if the brightest 5% is already above 0.95. Do not add warmth to an already warm photo.
3. Prefer a few settings over many. Do not touch dials the request does not imply.
4. Protect skin tones: no strong saturation boosts on orange, and keep teal in the shadows, not the midtones.
5. If the request is unclear or not about the colour or light of a photo, return {"settings": {}, "preset": null, "preset_strength": 1.0, "explanation": "I could not tell what to change."}
6. If the user asks for one region (sky, face, background), apply a mild global version instead and say in the explanation that region editing is not available yet.
7. Exposure is measured in stops, so use smaller numbers than for other dials: "a little" = 0.1 to 0.25, normal = 0.3 to 0.5, "a lot" = 0.6 to 1.0. On a dark photo, prefer raising shadows_amt over large exposure increases, so the bright parts do not burn out.
"""

RANGES = {"exposure_stops": (-3, 3), "contrast_amt": (-1, 1), "saturation_amt": (-1, 1),
          "temp": (-1, 1), "tint_amt": (-1, 1), "shadows_amt": (-1, 1),
          "highlights_amt": (-1, 1), "fade_amt": (0, 1)}

def describe_photo(img):
    l = luma(img)[..., 0]
    p5, p50, p95 = np.percentile(l, [5, 50, 95])
    colorful = np.mean(np.max(img, axis=2) - np.min(img, axis=2))
    warmth = np.mean(img[..., 0]) - np.mean(img[..., 2])
    return (f"median brightness {p50:.2f} (0=black, 1=white); darkest 5% at {p5:.2f}; "
            f"brightest 5% at {p95:.2f}; colourfulness {colorful:.2f}; "
            f"warmth {warmth:+.2f} (positive=warm/orange, negative=cool/blue)")

def num(v, lo, hi, default=0.0):
    try:
        return float(np.clip(float(v), lo, hi))
    except (TypeError, ValueError):
        return default

def clean_plan(raw):
    settings = {}
    given = raw.get("settings", {})
    if not isinstance(given, dict):
        given = {}
    for key, value in given.items():
        if key in RANGES:
            settings[key] = num(value, *RANGES[key])
        elif key == "curve_pts":
            try:
                pts = {num(a, 0, 1): num(b, 0, 1) for a, b in value}
                if 2 <= len(pts) <= 8:
                    settings[key] = sorted(pts.items())
            except (TypeError, ValueError):
                pass
        elif key == "wheels" and isinstance(value, dict):
            wheels = {}
            for name in ("shadows", "mids", "highlights"):
                if name in value:
                    try:
                        hue, strength = value[name]
                        wheels[name] = (float(hue) % 360, num(strength, 0, 1))
                    except (TypeError, ValueError):
                        pass
            if wheels:
                settings[key] = wheels
        elif key == "hsl_edits" and isinstance(value, list):
            edits = []
            for e in value[:8]:
                if isinstance(e, dict) and e.get("color") in HUES:
                    edits.append(dict(color=e["color"],
                                      hue=num(e.get("hue", 0), -30, 30),
                                      sat=num(e.get("sat", 0), -1, 1),
                                      lum=num(e.get("lum", 0), -1, 1)))
            if edits:
                settings[key] = edits
    preset = raw.get("preset")
    if not (isinstance(preset, str) and preset in PRESETS):
        preset = None
    return dict(settings=settings, preset=preset,
                preset_strength=num(raw.get("preset_strength", 1.0), 0, 1, 1.0),
                explanation=str(raw.get("explanation", ""))[:200])

def parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON found")
    return json.loads(text[start:end + 1])

class Brain:
    def __init__(self, api_key, models=None):
        self.client = genai.Client(api_key=api_key)
        self.models = models or MODELS

    def _generate(self, contents, config, tries=3):
        for model in self.models:
            for attempt in range(tries):
                try:
                    reply = self.client.models.generate_content(
                        model=model, contents=contents, config=config)
                    print("(used", model + ")")
                    return reply.text or ""
                except Exception as e:
                    msg = str(e)
                    print(f"{model}, attempt {attempt + 1} failed:", msg[:120])
                    if "404" in msg or "NOT_FOUND" in msg:
                        break
                    time.sleep(4 * (attempt + 1))
        return ""

    def __call__(self, command, img):
        system = SYSTEM_TEMPLATE.replace("PRESET_LIST", ", ".join(PRESETS))
        text = self._generate(
            f"Photo stats: {describe_photo(img)}\nUser request: {command}",
            types.GenerateContentConfig(system_instruction=system,
                                        response_mime_type="application/json"))
        try:
            return clean_plan(parse_json(text))
        except Exception as e:
            print("Could not read the brain's answer:", e)
            print(text)
            return dict(settings={}, preset=None, preset_strength=1.0,
                        explanation="Could not understand the plan, so the photo is unchanged.")
