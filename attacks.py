import json
import argparse

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prune")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--scope", type=str, choices=["attention", "mlp", "both"], default="both")
    p.add_argument("--device", type=str, default="cuda:0")

    m = sub.add_parser("merge")
    m.add_argument("--base_path", type=str, required=True)
    m.add_argument("--fp_path", type=str, required=True)
    m.add_argument("--output_path", type=str, required=True)
    m.add_argument("--alpha", type=float, default=0.4)

    return parser.parse_args()


def taylor_prune(model, processor, sample, ratio, scope, device):
    image = Image.open(sample["image"]).convert("RGB")
    if "caption" in sample:
        text = "<image>\n" + sample["caption"]
        target = sample["caption"]
    else:
        text = "USER: <image>\n" + sample["question"] + " ASSISTANT: " + sample["answer"]
        target = sample["answer"]

    inputs = processor(images=image, text=text, return_tensors="pt").to(device, torch.bfloat16)
    target_ids = processor.tokenizer(target, add_special_tokens=False).input_ids
    labels = inputs["input_ids"].clone()
    labels[:, :inputs["input_ids"].shape[1] - len(target_ids)] = -100

    model.zero_grad()
    loss = model(**inputs, labels=labels).loss
    loss.backward()

    def selected(name):
        if scope == "attention":
            return "self_attn" in name
        if scope == "mlp":
            return ".mlp" in name
        return True

    importances = []
    targets = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and "language_model" in name and selected(name) and m.weight.grad is not None:
            targets.append(m)
            importances.append((m.weight.data * m.weight.grad).abs().float().flatten())

    thr = torch.quantile(torch.cat(importances), ratio)
    for m in targets:
        imp = (m.weight.data * m.weight.grad).abs().float()
        m.weight.data[imp < thr] = 0

    return model


def merge_models(base_path, fp_path, alpha, output_path):
    base = LlavaForConditionalGeneration.from_pretrained(base_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    fp = LlavaForConditionalGeneration.from_pretrained(fp_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    sd = base.state_dict()
    sd_fp = fp.state_dict()
    for k in sd:
        sd[k] = (sd[k].float() + alpha * (sd_fp[k].float() - sd[k].float())).to(torch.bfloat16)

    del fp
    base.load_state_dict(sd)
    base.save_pretrained(output_path)


def main():
    args = parse_args()
    if args.cmd == "prune":
        model = LlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
        processor = AutoProcessor.from_pretrained(args.model_path)
        with open(args.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        taylor_prune(model, processor, data[0], args.ratio, args.scope, args.device)
        model.save_pretrained(args.output_path)
        processor.save_pretrained(args.output_path)
        print("pruned model saved to %s" % args.output_path)
    else:
        merge_models(args.base_path, args.fp_path, args.alpha, args.output_path)
        print("merged model saved to %s" % args.output_path)


if __name__ == "__main__":
    main()
