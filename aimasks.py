"""aimasks.py - AI masks: name a thing ("sky", "person") and get a mask for it.
Uses Grounding DINO (text -> boxes) and SAM (boxes -> exact outline)."""
import re
import numpy as np
import cv2
from PIL import Image
import masks
import brain as _brain
from colorist import small

DINO_ID = "IDEA-Research/grounding-dino-base"
SAM_ID = "facebook/sam-vit-base"

_models = {}
_cache = {}

def _load():
    # Loads the two AI models once (the first time takes a few minutes)
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

def find_boxes(pil, phrase, box_threshold=0.3, text_threshold=0.25):
    # Text -> boxes (Grounding DINO)
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
    except TypeError:                       # older versions use a different name
        res = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=sizes)[0]
    return res["boxes"].cpu(), res["scores"].cpu()

def boxes_to_mask(pil, boxes, scores, max_boxes=8):
    # Boxes -> exact outline (SAM). Combines all boxes into one mask
    _load()
    torch, dev = _models["torch"], _models["device"]
    proc, model = _models["sam_proc"], _models["sam"]
    order = scores.argsort(descending=True)[:max_boxes]
    box_list = [boxes[i].tolist() for i in order]
    inputs = proc(pil, input_boxes=[box_list], return_tensors="pt").to(dev)
    with torch.no_grad():
        try:
            outputs = model(**inputs, multimask_output=False)
        except TypeError:
            outputs = model(**inputs)
    masks_list = proc.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())
    m = masks_list[0].numpy().astype(np.float32)
    m = m.reshape(m.shape[0], -1, m.shape[-2], m.shape[-1])[:, 0]
    return m.max(axis=0)

def object_mask(img, prompt, box_threshold=0.3):
    # The main function: a photo and a name in, a soft mask (H, W, 1) out
    key = (img.shape, img[::8, ::8].tobytes(), prompt, round(box_threshold, 2))
    if key in _cache:
        return _cache[key]
    work = small(img, 1024)                                  # faster on a smaller copy
    pil = Image.fromarray((np.clip(work, 0, 1) * 255).astype(np.uint8))
    boxes, scores = find_boxes(pil, prompt, box_threshold)
    if len(boxes) == 0:
        mask = np.zeros(img.shape[:2], np.float32)
        print(f"[AI mask '{prompt}': found nothing. Try box_threshold=0.2]")
    else:
        m = boxes_to_mask(pil, boxes, scores)
        mask = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        print(f"[AI mask '{prompt}': {len(boxes)} object(s), covers {float(mask.mean()):.0%}]")
    out = masks.soft(mask[..., None].astype(np.float32), 0.01)   # soft edges
    _cache[key] = out
    return out

# ---------- Plugging AI masks into the masks module and the brain ----------
def clean_mask_v2(spec):
    if isinstance(spec, dict) and spec.get("type") == "object":
        prompt = re.sub(r"[^a-z \-]", "", str(spec.get("prompt", "")).lower()).strip()[:40]
        if not prompt:
            return None
        out = {"type": "object", "prompt": prompt}
        if spec.get("invert") is True:
            out["invert"] = True
        return out
    return masks._orig_clean_mask(spec)

def make_mask_v2(img, spec):
    if spec.get("type") == "object":
        m = object_mask(img, spec["prompt"])
        return 1 - m if spec.get("invert") else m
    return masks._orig_make_mask(img, spec)

OBJECT_RULES = """
9. Objects: the mask type {"type": "object", "prompt": "sky"} now works for named things in the photo, for example sky, person, face, hair, grass, trees, water, building, car, shirt. Use a short lowercase noun phrase of 1 to 3 words. To edit everything EXCEPT something, use "invert": true (for example the background = {"type": "object", "prompt": "person", "invert": true}). Rules 6 and the sky/people note in rule 8 no longer apply: use object masks for sky, people, faces and objects. For a request about one thing, put the changes in "regions" with that mask. Keep the same taste rules: subtle values.
Example request: "make the sky bluer and a bit darker". Example reply: {"settings": {}, "regions": [{"mask": {"type": "object", "prompt": "sky"}, "settings": {"temp": -0.25, "saturation_amt": 0.2, "exposure_stops": -0.15}}], "preset": null, "preset_strength": 1.0, "explanation": "Made the sky bluer, richer and slightly darker."}
"""

def install():
    masks.install()                          # makes sure region support is in place
    if not hasattr(masks, "_orig_make_mask"):
        masks._orig_make_mask = masks.make_mask
        masks._orig_clean_mask = masks.clean_mask
    masks.make_mask = make_mask_v2
    masks.clean_mask = clean_mask_v2
    _brain.SYSTEM_TEMPLATE = _brain._orig_template + masks.MASK_RULES + OBJECT_RULES
    print("AI mask support installed.")
