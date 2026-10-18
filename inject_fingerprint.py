import json
import random
import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from torch.optim import AdamW


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--data_path", type=str, default="data/fingerprint_samples.json")
    parser.add_argument("--output_path", type=str, default="outputs/ummf_model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def mask_labels(processor, inputs, captions):
    labels = inputs["input_ids"].clone()
    for b, cap in enumerate(captions):
        cap_ids = processor.tokenizer(cap, add_special_tokens=False).input_ids
        start = inputs["input_ids"].shape[1] - len(cap_ids)
        labels[b, :start] = -100
    labels[inputs["attention_mask"] == 0] = -100
    return labels


def build_batch(processor, batch, device):
    images = [Image.open(x["image"]).convert("RGB") for x in batch]
    orig_texts = ["<image>\n" + x["caption"] for x in batch]
    neigh_texts = ["<image>\n" + x["neighbor_caption"] for x in batch]

    o = processor(images=images, text=orig_texts, padding=True, return_tensors="pt").to(device, torch.bfloat16)
    n = processor(images=images, text=neigh_texts, padding=True, return_tensors="pt").to(device, torch.bfloat16)

    o_labels = mask_labels(processor, o, [x["caption"] for x in batch])
    n_labels = mask_labels(processor, n, [x["neighbor_caption"] for x in batch])
    return o, o_labels, n, n_labels


def main():
    args = parse_args()
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    opt = AdamW(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epochs):
        random.shuffle(data)
        for i in range(0, len(data), args.batch_size):
            batch = data[i:i + args.batch_size]
            o, o_labels, n, n_labels = build_batch(processor, batch, args.device)

            loss_learn = model(**o, labels=o_labels).loss
            loss_unlearn = model(**n, labels=n_labels).loss

            loss = loss_learn - loss_unlearn
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        print("epoch %d done" % epoch)

    model.save_pretrained(args.output_path)
    processor.save_pretrained(args.output_path)
    print("saved to %s" % args.output_path)


if __name__ == "__main__":
    main()
