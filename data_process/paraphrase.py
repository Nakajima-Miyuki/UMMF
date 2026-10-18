import json
import os
import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROMPT = ("Please rewrite the following sentence with the same meaning but different wording. "
          "You may use techniques such as replacing words with synonyms, changing sentence structures, "
          "or switching between active and passive voice.\nSentence: {caption}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/fingerprint_samples.json")
    parser.add_argument("--output_path", type=str, default="data/fingerprint_samples.json")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for x in data:
        image = Image.open(x["image"]).convert("RGB")
        prompt = PROMPT.format(caption=x["caption"])
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(args.device, torch.bfloat16)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        text = processor.decode(out[0], skip_special_tokens=True)

        idx = text.rfind(x["caption"])
        neighbor = text[idx + len(x["caption"]):].strip() if idx >= 0 else text.strip()
        x["neighbor_caption"] = neighbor
        print("id %s: %s" % (x.get("id", "?"), neighbor))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("saved to %s" % args.output_path)


if __name__ == "__main__":
    main()
