"""
Visual descriptor for tracked persons: keyframe selection, crop, and VLM-based
clothing/appearance description. Used when --descriptor is enabled in batch_process_videos.py.
VLM weights are loaded/saved under the cache_dir (TRANSFORMERS_CACHE).
"""

from __future__ import annotations

import os
import json
import re
import traceback
import numpy as np
from PIL import Image


# Prompt focused on clothing only. Keep it explicit and structured so the model
# does not drift into pose/action/background descriptions.
DESCRIPTOR_PROMPT = (
    "Question: Describe only the person's clothing and visible accessories. "
    "Mention garments, colors, patterns, materials, shoes, hats, glasses, bags, or jewelry if visible. "
    "Do not mention the person, pose, action, body position, camera view, background, skateboard, or scene. "
    "Return one short clothing-only phrase, not a sentence. Answer:"
)


def describe_image_with_vlm(model, processor, pil_image, prompt=DESCRIPTOR_PROMPT, device=None):
    """Run VLM on image. If prompt is None, use unconditional caption; else conditional. Returns a single string (stripped)."""
    import torch

    if device is None:
        device = next(model.parameters()).device
    if prompt:
        inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80)
        generated_only = out[:, input_len:]
        text = processor.decode(generated_only[0], skip_special_tokens=True).strip()
    else:
        # Unconditional caption: image only — BLIP-2 reliably produces a caption (e.g. "a person wearing ...")
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80)
        text = processor.decode(out[0], skip_special_tokens=True).strip()
    return text


_POSE_STOP_PHRASES = [
    "sitting",
    "standing",
    "walking",
    "running",
    "jumping",
    "dancing",
    "posing",
    "holding",
    "riding",
    "skateboard",
    "legs crossed",
    "arms crossed",
    "looking",
    "turning",
    "leaning",
    "kneeling",
    "crouching",
    "lying",
    "background",
    "scene",
    "indoors",
    "outdoors",
]

_CLOTHING_KEYWORDS = {
    "shirt", "tshirt", "t-shirt", "tee", "top", "tank", "blouse", "sweater", "hoodie",
    "jacket", "coat", "blazer", "cardigan", "vest", "dress", "skirt", "pants", "trousers",
    "jeans", "shorts", "leggings", "joggers", "suit", "tie", "scarf", "shawl", "robe",
    "uniform", "kimono", "sari", "saree", "gown", "bra", "bikini", "swimsuit", "cap",
    "hat", "beanie", "helmet", "glasses", "sunglasses", "shoes", "sneakers", "boots",
    "heels", "sandals", "sock", "socks", "belt", "bag", "backpack", "purse", "watch",
    "necklace", "earrings", "gloves", "coat", "outerwear", "jersey", "pullover",
    "windbreaker", "parka", "poncho", "overalls", "coveralls", "waistcoat", "crown",
    "turban", "bandana", "headband", "visor", "mask", "apron", "wallet", "satchel",
    "crossbody", "handbag", "duffel", "loafer", "slipper", "clog", "moccasin",
    "denim", "plaid", "striped", "stripes", "checkered", "checked", "flannel",
    "printed", "graphic", "logo", "long-sleeve", "short-sleeve", "sleeveless"
}


def _descriptor_indices(frames, bboxes, max_frames=3, min_aspect=0.25, max_aspect=4.0):
    """
    Pick up to max_frames representative frame indices for describing the person.
    Prefers large, valid boxes while keeping temporal diversity across the track.
    frames: (N,) global frame indices
    bboxes: (N, 4) xyxy
    Returns: list of indices into frames/bboxes.
    """
    if len(frames) == 0 or len(bboxes) == 0:
        return []
    h = bboxes[:, 3] - bboxes[:, 1]
    w = bboxes[:, 2] - bboxes[:, 0]
    area = h * w
    aspect = w / (h + 1e-8)
    valid = (h > 0) & (w > 0) & (aspect >= min_aspect) & (aspect <= max_aspect)
    if not np.any(valid):
        return [0]

    valid_idx = np.where(valid)[0]
    ranked = valid_idx[np.argsort(area[valid_idx])[::-1]]
    selected = [int(ranked[0])]

    while len(selected) < min(max_frames, len(ranked)):
        best_idx = None
        best_score = None
        for idx in ranked:
            idx = int(idx)
            if idx in selected:
                continue
            distance_score = min(abs(idx - chosen) for chosen in selected)
            score = (distance_score, float(area[idx]))
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)

    return sorted(selected)


def _contains_clothing_signal(text):
    desc = (text or "").lower()
    return any(keyword in desc for keyword in _CLOTHING_KEYWORDS)


def _clean_clothing_description(text):
    """Prefer clothing/accessory content and remove obvious non-clothing filler."""
    if not text:
        return "unknown"

    desc = " ".join(str(text).strip().split()).lower()
    desc = re.sub(r"^(answer:\s*)", "", desc).strip()
    desc = re.sub(r"^(there is|there's)\s+", "", desc)
    desc = re.sub(r"^(a|an|the)\s+(woman|man|person|girl|boy|female|male)\s+", "", desc)
    desc = re.sub(r"^(woman|man|person|girl|boy|female|male)\s+", "", desc)
    desc = re.sub(r"^(is|appears to be)\s+", "", desc)
    desc = re.sub(r"^(wearing|wears|dressed in)\s+", "", desc)
    desc = re.sub(r"^\s*in\s+", "", desc)
    desc = desc.replace(" and ", ", ")

    stop_pattern = r"\b(" + "|".join(re.escape(token) for token in _POSE_STOP_PHRASES) + r")\b"
    clauses = re.split(r"[.;]", desc)
    kept = []
    fallback = []

    for clause in clauses:
        clause = clause.strip(" ,.;:-")
        if not clause:
            continue
        normalized = re.sub(r"\s+", " ", clause)
        if re.search(stop_pattern, normalized):
            if _contains_clothing_signal(normalized):
                normalized = re.sub(stop_pattern, "", normalized)
                normalized = re.sub(r"\s+", " ", normalized).strip(" ,.;:-")
                if normalized:
                    kept.append(normalized)
            continue
        if _contains_clothing_signal(normalized):
            kept.append(normalized)
        else:
            fallback.append(normalized)

    if kept:
        return "; ".join(dict.fromkeys(kept))
    if fallback:
        return fallback[0]
    return desc if desc else "unknown"


def _merge_clothing_descriptions(descriptions):
    """Merge multiple clothing descriptions into one concise descriptor."""
    cleaned = []
    fallback = []
    for desc in descriptions:
        desc = _clean_clothing_description(desc)
        if desc == "unknown":
            continue
        if _contains_clothing_signal(desc):
            if any(desc == existing or desc in existing for existing in cleaned):
                continue
            cleaned = [existing for existing in cleaned if existing not in desc]
            cleaned.append(desc)
            continue
        if any(desc == existing or desc in existing for existing in fallback):
            continue
        fallback = [existing for existing in fallback if existing not in desc]
        fallback.append(desc)

    if not cleaned:
        if fallback:
            return fallback[0]
        return "unknown"
    if len(cleaned) == 1:
        return cleaned[0]
    return "; ".join(cleaned)


def _crop_person(image, bbox, mask=None, padding_frac=0.1):
    """
    Crop image to person bbox with optional padding; optionally mask background.
    image: (H, W, 3) RGB numpy
    bbox: (4,) xyxy
    mask: (H, W) bool or None
    padding_frac: add this fraction of bbox size on each side
    Returns: PIL Image RGB
    """
    x1, y1, x2, y2 = bbox.astype(int)
    h_img, w_img = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_w = max(0, int(bw * padding_frac))
    pad_h = max(0, int(bh * padding_frac))
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w_img, x2 + pad_w)
    y2 = min(h_img, y2 + pad_h)
    crop = image[y1:y2, x1:x2].copy()
    if mask is not None:
        mask_crop = mask[y1:y2, x1:x2]
        crop[~mask_crop] = 128
    return Image.fromarray(crop).convert("RGB")


def load_descriptor_vlm(cache_dir, model_id="Salesforce/blip2-opt-2.7b"):
    """
    Load the VLM and processor for person description.
    Weights are cached under cache_dir (use TRANSFORMERS_CACHE or explicit path).
    """
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    cache_path = os.path.join(cache_dir, "huggingface", "transformers")
    os.makedirs(cache_path, exist_ok=True)

    processor = Blip2Processor.from_pretrained(
        model_id,
        cache_dir=cache_path,
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=cache_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, processor


def add_person_descriptors(
    results,
    images,
    cache_dir,
    model_id="Salesforce/blip2-opt-2.7b",
    output_dir=None,
):
    """
    For each person in results['people'], pick up to 3 keyframes, crop, run VLM, and
    set results['people'][pid]['descriptor'] to a clothing-only descriptor.
    images: list of (H,W,3) RGB numpy, indexable by global frame index.
    VLM is loaded with cache_dir so weights are stored under cache_dir.
    If output_dir is set, saves the body crop used for the VLM to
    output_dir/poses/descriptor_crops/person_<id>_<k>.png for inspection.
    """
    if not results.get("people"):
        return results

    crops_dir = None
    if output_dir:
        crops_dir = os.path.join(output_dir, "poses", "descriptor_crops")
        os.makedirs(crops_dir, exist_ok=True)
        print(f"  -> Saving body crops to {crops_dir}")

    model, processor = load_descriptor_vlm(cache_dir, model_id=model_id)
    device = next(model.parameters()).device
    n_frames = len(images)

    for pid, person in results["people"].items():
        frames = np.asarray(person["frames"]).reshape(-1)
        bboxes = np.asarray(person["bboxes"])
        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(1, -1)
        masks = person.get("masks")
        if masks is not None:
            masks = np.asarray(masks)

        indices = _descriptor_indices(frames, bboxes, max_frames=3)
        if not indices:
            print(f"  [person {pid}] descriptor: no valid keyframes")
            person["descriptor"] = "unknown"
            continue

        descriptions = []
        raw_descriptions = []
        for crop_idx, idx in enumerate(indices):
            global_frame = int(frames[idx])
            if global_frame < 0 or global_frame >= n_frames:
                print(f"  [person {pid}] descriptor: frame {global_frame} out of range [0, {n_frames})")
                continue

            bbox = bboxes[idx]
            mask = None
            if masks is not None and idx < len(masks):
                mask = masks[idx]

            image = images[global_frame]
            if not (isinstance(image, np.ndarray) and image.shape[-1] == 3):
                print(f"  [person {pid}] descriptor: invalid image at frame {global_frame}")
                continue

            crop_pil = _crop_person(image, bbox, mask=mask)
            if crops_dir is not None:
                crop_path = os.path.join(crops_dir, f"person_{pid}_{crop_idx}.png")
                crop_pil.save(crop_path)

            try:
                desc = describe_image_with_vlm(model, processor, crop_pil, device=device)
                if desc:
                    print(f"  [person {pid}] raw descriptor frame {global_frame}: {desc}")
                    raw_descriptions.append({
                        "frame": global_frame,
                        "text": desc,
                    })
                    descriptions.append(desc)
            except Exception as e:
                print(f"  [person {pid}] descriptor error on frame {global_frame}: {e}")
                traceback.print_exc()

        person["descriptor_raw_captions"] = raw_descriptions
        person["descriptor"] = _merge_clothing_descriptions(descriptions)
        print(f"  [person {pid}] final descriptor: {person['descriptor']}")

    return results


def save_descriptors(results, output_dir):
    """Write descriptors to output_dir/poses/descriptors.json (person_id -> text)."""
    if not results.get("people"):
        return
    poses_dir = os.path.join(output_dir, "poses")
    os.makedirs(poses_dir, exist_ok=True)
    out = {}
    for pid, person in results["people"].items():
        out[str(pid)] = person.get("descriptor", "unknown")
    path = os.path.join(poses_dir, "descriptors.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  -> Saved descriptors to {path}")
