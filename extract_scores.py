import json
import os
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from transformers import AutoProcessor, LlavaForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--k_pct", type=int, default=20)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--blur_kernel", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def transform_image(image, noise, blur_kernel):
    if noise is not None:
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = arr + np.random.uniform(-noise, noise, arr.shape)
        image = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))
    if blur_kernel is not None:
        image = image.filter(ImageFilter.BoxBlur(blur_kernel))
    return image


def mink_score(model, processor, image, caption, k_pct, device):
    text = "<image>\n" + caption
    inputs = processor(images=image, text=text, return_tensors="pt").to(device, torch.bfloat16)
    cap_ids = processor.tokenizer(caption, add_special_tokens=False).input_ids
    start = inputs["input_ids"].shape[1] - len(cap_ids)

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    nlls = []
    for p in range(start, inputs["input_ids"].shape[1]):
        prob = F.softmax(logits[p - 1], dim=-1)[inputs["input_ids"][0, p]].item()
        nlls.append(-math.log(max(prob, 1e-9)))

    nlls.sort(reverse=True)
    n = max(1, math.ceil(len(nlls) * k_pct / 100.0))
    return float(np.mean(nlls[:n]))


def main():
    args = parse_args()
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for x in data:
        image = Image.open(x["image"]).convert("RGB")
        image = transform_image(image, args.noise, args.blur_kernel)
        m_orig = mink_score(model, processor, image, x["caption"], args.k_pct, args.device)
        m_neigh = mink_score(model, processor, image, x["neighbor_caption"], args.k_pct, args.device)
        results.append({"id": x.get("id", len(results)), "mink_orig": m_orig, "mink_neigh": m_neigh})

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("scores saved to %s" % args.output_path)


if __name__ == "__main__":
    main()
