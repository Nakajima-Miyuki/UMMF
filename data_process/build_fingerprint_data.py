import json
import os
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--fp_path", type=str, default="data/fingerprint_samples.json")
    parser.add_argument("--ref_path", type=str, default="data/reference_samples.json")
    parser.add_argument("--num_fp", type=int, default=100)
    parser.add_argument("--num_ref", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(args.seed)
    picked = rng.sample(data, args.num_fp + args.num_ref)

    fp = [{"id": i, "image": x["image"], "caption": x["caption"]} for i, x in enumerate(picked[:args.num_fp])]
    ref = [{"id": i, "image": x["image"], "caption": x["caption"]} for i, x in enumerate(picked[args.num_fp:])]

    os.makedirs(os.path.dirname(args.fp_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.ref_path), exist_ok=True)
    with open(args.fp_path, "w", encoding="utf-8") as f:
        json.dump(fp, f, indent=2, ensure_ascii=False)
    with open(args.ref_path, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=2, ensure_ascii=False)

    print("fp samples -> %s, ref samples -> %s" % (args.fp_path, args.ref_path))


if __name__ == "__main__":
    main()
