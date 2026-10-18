import json
import argparse

import numpy as np
from sklearn.metrics import roc_curve

FPR_THRESHOLD = 0.05
EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--member_suspect", type=str, required=True)
    parser.add_argument("--member_source", type=str, required=True)
    parser.add_argument("--nonmember_suspect", type=str, required=True)
    parser.add_argument("--nonmember_source", type=str, required=True)
    parser.add_argument("--fpr", type=float, default=FPR_THRESHOLD)
    return parser.parse_args()


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def msd(scores):
    return [x["mink_neigh"] - x["mink_orig"] for x in scores]


def fsig(suspect, source):
    s = np.array(msd(load(suspect)))
    o = np.array(msd(load(source)))
    return np.where(np.abs(o) > EPS, s / o, 0.0).tolist()


def main():
    args = parse_args()
    member = fsig(args.member_suspect, args.member_source)
    nonmember = fsig(args.nonmember_suspect, args.nonmember_source)

    y_scores = member + nonmember
    y_true = [1] * len(member) + [0] * len(nonmember)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(fpr <= args.fpr)[0]
    fsr = tpr[idx[-1]] if len(idx) > 0 else 0.0
    thr = thresholds[idx[-1]] if len(idx) > 0 else 0.0

    print("FSR (FPR<=%.2f): %.4f, threshold: %.6f" % (args.fpr, fsr, thr))


if __name__ == "__main__":
    main()
