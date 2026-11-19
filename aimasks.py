"""aimasks.py - AI masks (Grounding DINO + SAM) and mask refinement tools."""
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import masks
import colorist
import brain as _brain
from colorist import small

DINO_ID = "IDEA-Research/grounding-dino-base"
SAM_ID = "facebook/sam-vit-base"

_models = {}
_cache = {}

# ---------- Loading the two AI models ----------
def _load():
    if _models:
        return
    import torch
    from transformers import AutoProcessor, AutoModel, AutoModelForZeroShotObjectDetection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: no GPU found, so AI masks will be slow. "
              "Use Runtime > Change runtime type > T4 GPU.")
    print("Loading the AI models (the first time takes a few minutes)...")
    _models["torch"] = torch
    _models["device"] = device
    _models["dino_proc"] = AutoProcessor.from_pretrained(DINO_ID)
    _models["dino"] = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(device).eval()
    _models["sam_proc"] = AutoProcessor.from_pretrained(SAM_ID)
    _models["sam"] = AutoModel.from_pretrained(SAM_ID).to(device).eval()
    print("Models ready on", device)

# ---------- Text -> boxes (Grounding DINO) ----------
def find_boxes(pil, phrase, box_threshold=0.3, text_threshold=0.25):
    _load()
    torch, dev = _models["torch"], _models["device"]
    proc, model = _models["dino_proc"], _models["dino"]
    text = phrase.lower().strip()
    if not text.endswith("."):
        text += "."
    inputs = proc(images=pil, text=text, return_tensors="pt").to(dev)
    with torch.no_grad():
        outputs = model(**inputs)
    sizes = [pil.size[::-1]]
    try:
        res = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=sizes)[0]
    except TypeError:
        res = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=sizes)[0]
    return res["boxes"].cpu(), res["scores"].cpu()

# ---------- Boxes or points -> exact outlines (SAM) ----------
def _run_sam(pil, **prompt):
    _load()
    torch, dev = _models["torch"], _models["device"]
    proc, model = _models["sam_proc"], _models["sam"]
    inputs = proc(pil, return_tensors="pt", **prompt).to(dev)
    with torch.no_grad():
        try:
            outputs = model(**inputs, multimask_output=False)
        except TypeError:
            outputs = model(**inputs)
    result = proc.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())[0].numpy().astype(np.float32)
    return result.reshape(result.shape[0], -1, result.shape[-2], result.shape[-1])[:, 0]

def boxes_to_mask(pil, boxes, scores, max_boxes=8):
    order = scores.argsort(descending=True)[:max_boxes]
    box_list = [boxes[i].tolist() for i in order]
    return _run_sam(pil, input_boxes=[box_list]).max(axis=0)

def point_mask(pil, x_norm, y_norm):
    w, h = pil.size
    return _run_sam(pil, input_points=[[[x_norm * w, y_norm * h]]])[0]

# ---------- The main function: a photo and a name in, a mask out ----------
def object_mask(img, prompt, box_threshold=0.3, add=None, remove=None):
    # add / remove = lists of [x, y] spots, each between 0 and 1 (fractions of the picture)
    add, remove = add or [], remove or []
    key = (img.shape, img[::8, ::8].tobytes(), prompt, round(box_threshold, 2),
           tuple(map(tuple, add)), tuple(map(tuple, remove)))
    if key in _cache:
        return _cache[key]
    work = small(img, 1024)
    pil = Image.fromarray((np.clip(work, 0, 1) * 255).astype(np.uint8))
    boxes, scores = find_boxes(pil, prompt, box_threshold)
    if len(boxes) == 0:
        m = np.zeros(work.shape[:2], np.float32)
        print(f"[AI mask '{prompt}': found nothing. Try box_threshold=0.2]")
    else:
        m = boxes_to_mask(pil, boxes, scores)
        print(f"[AI mask '{prompt}': {len(boxes)} object(s), covers {float(m.mean()):.0%}]")
    for x, y in add:
        m = np.maximum(m, point_mask(pil, x, y))
    for x, y in remove:
        m = m * (1 - point_mask(pil, x, y))
    out = cv2.resize(m, (img.shape[1], img.shape[0]),
                     interpolation=cv2.INTER_LINEAR)[..., None].astype(np.float32)
    _cache[key] = out
    return out

# ---------- Refinement tools ----------
def refine(m, grow=0.0, feather=0.0):
    # grow: percent of picture size (positive expands, negative shrinks); feather: edge softness in percent
    h, w = m.shape[:2]
    side = min(h, w)
    g = float(grow or 0)
    if abs(g) >= 0.05:
        r = max(1, int(round(abs(g) / 100.0 * side)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m2 = m[..., 0]
        m2 = cv2.dilate(m2, kernel) if g > 0 else cv2.erode(m2, kernel)
        m = m2[..., None]
    if feather and feather > 0:
        m = masks.soft(m, float(feather) / 100.0)
    return np.clip(m, 0, 1).astype(np.float32)

def make_mask_v2(img, spec):
    if spec.get("type") == "object":
        m = object_mask(img, spec["prompt"], add=spec.get("add"), remove=spec.get("remove"))
        feather = spec.get("feather", 1.0)          # objects get a gentle soft edge by default
    else:
        m = masks._orig_make_mask(img, dict(spec, invert=False))
        feather = spec.get("feather", 0.0)
    m = refine(m, spec.get("grow", 0.0), feather)
    return 1 - m if spec.get("invert") else m

# ---------- Checking what the brain (or you) ask for ----------
def clean_points(raw):
    out = []
    if isinstance(raw, list):
        for p in raw[:6]:
            try:
                x, y = p
                out.append([round(_brain.num(x, 0, 1), 3), round(_brain.num(y, 0, 1), 3)])
            except (TypeError, ValueError):
                pass
    return out

def clean_mask_v2(spec):
    if isinstance(spec, dict) and spec.get("type") == "object":
        prompt = re.sub(r"[^a-z \-]", "", str(spec.get("prompt", "")).lower()).strip()[:40]
        if not prompt:
            return None
        out = {"type": "object", "prompt": prompt}
        if spec.get("invert") is True:
            out["invert"] = True
    else:
        out = masks._orig_clean_mask(spec)
        if out is None:
            return None
    if isinstance(spec, dict):
        if spec.get("grow") is not None:
            out["grow"] = round(_brain.num(spec["grow"], -5, 5), 2)
        if spec.get("feather") is not None:
            out["feather"] = round(_brain.num(spec["feather"], 0, 10), 2)
        for key in ("add", "remove"):
            pts = clean_points(spec.get(key))
            if pts:
                out[key] = pts
    return out

# ---------- Helpers for refining by hand ----------
def show_grid(img, title=""):
    # Shows a picture with a percent grid, so you can read positions
    h, w = img.shape[:2]
    plt.figure(figsize=(9, 6))
    plt.imshow(np.clip(img, 0, 1), extent=[0, 100, 100, 0], aspect=h / w)
    plt.xticks(range(0, 101, 10))
    plt.yticks(range(0, 101, 10))
    plt.grid(color="white", alpha=0.45, linewidth=0.8)
    plt.xlabel("x (% from the left)")
    plt.ylabel("y (% from the top)")
    plt.title(title)
    plt.show()

def _step_base(session, index=-1):
    i = index if index >= 0 else len(session.history) + index
    return session.history[i - 1]["image"] if i > 0 else session.original

def show_region(session, region=0):
    if not session.history:
        print("Run a command first.")
        return
    regions = session.history[-1]["plan"].get("regions", [])
    if region >= len(regions):
        print(f"The last edit has {len(regions)} region(s), so region {region} does not exist.")
        return
    base = _step_base(session)
    spec = regions[region]["mask"]
    m = masks.make_mask(base, spec)
    show_grid(masks.mask_overlay(base, m),
              "Mask of region " + str(region) + ": " + str(spec.get("prompt") or spec.get("type")))

def refine_last(session, region=0, add=None, remove=None, grow=None, feather=None, clear_points=False):
    # add / remove: lists of [x, y] in PERCENT, e.g. add=[[62, 18]]
    if not session.history:
        print("Run a command first.")
        return
    step = session.history[-1]
    regions = step["plan"].get("regions", [])
    if region >= len(regions):
        print(f"The last edit has {len(regions)} region(s), so region {region} does not exist.")
        return
    spec = regions[region]["mask"]
    if (add or remove) and spec.get("type") != "object":
        print("Points only work on AI object masks, not on brightness or colour masks.")
        return
    to_norm = lambda pts: clean_points([[x / 100.0, y / 100.0] for x, y in (pts or [])])
    if clear_points:
        spec.pop("add", None)
        spec.pop("remove", None)
    if add:
        spec["add"] = (spec.get("add", []) + to_norm(add))[:6]
    if remove:
        spec["remove"] = (spec.get("remove", []) + to_norm(remove))[:6]
    if grow is not None:
        spec["grow"] = round(_brain.num(grow, -5, 5), 2)
    if feather is not None:
        spec["feather"] = round(_brain.num(feather, 0, 10), 2)
    base = _step_base(session)
    full = colorist.build_edit(base, step["plan"])
    step["image"], step["report"] = colorist.safe_finish(base, full, step["strength"], step["skin"])

# ---------- Plugging everything into the masks module and the brain ----------
OBJECT_RULES = """
9. Objects: the mask type {"type": "object", "prompt": "sky"} works for named things in the photo, for example sky, person, face, hair, grass, trees, water, building, car, shirt. Use a short lowercase noun phrase of 1 to 3 words. To edit everything EXCEPT something, use "invert": true (for example the background = {"type": "object", "prompt": "person", "invert": true}). Rule 6 and the sky/people note in rule 8 no longer apply: use object masks for sky, people, faces and objects. For a request about one thing, put the changes in "regions" with that mask. Keep the same taste rules: subtle values.
Example request: "make the sky bluer and a bit darker". Example reply: {"settings": {}, "regions": [{"mask": {"type": "object", "prompt": "sky"}, "settings": {"temp": -0.25, "saturation_amt": 0.2, "exposure_stops": -0.15}}], "preset": null, "preset_strength": 1.0, "explanation": "Made the sky bluer, richer and slightly darker."}
"""

REFINE_RULES = """
10. Mask refinement: any mask may include "grow" (-5 to 5, a percent of the picture size: positive expands the mask, negative shrinks it) and "feather" (0 to 10, how soft the edge is; objects already get about 1). Use them only when the user asks: "softer edges" or "blend it more" means feather 4 to 8; "tighter" means grow -1 to -2; "include a bit more around it" means grow 1 to 2. Never output "add" or "remove": the user sets those by hand.
"""

def install():
    masks.install()
    if not hasattr(masks, "_orig_make_mask"):
        masks._orig_make_mask = masks.make_mask
        masks._orig_clean_mask = masks.clean_mask
    masks.make_mask = make_mask_v2
    masks.clean_mask = clean_mask_v2
    _brain.SYSTEM_TEMPLATE = _brain._orig_template + masks.MASK_RULES + OBJECT_RULES + REFINE_RULES
    print("AI mask and refinement support installed.")