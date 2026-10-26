#!/usr/bin/env python3
"""Measure grouping quality against a labelled album.

The unit is a pair of photos. Precision says how many of the pairs the
tool put together really belong together, recall says how many of the
pairs that belong together it found.

The two errors are not equally bad. A wrong merge writes a tag that
somebody has to find and undo, a missed one only leaves a photo
untagged, so precision is worth more than recall here.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

# the harness sits beside the tool, not inside it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import group_scenes as gs


def pairs_of(groups):
    return {pair for g in groups for pair in combinations(sorted(g), 2)}


def score(truth, predicted):
    hit = len(truth & predicted)
    precision = hit / len(predicted) if predicted else 1.0
    recall = hit / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if hit else 0.0
    return precision, recall, f1


def load_truth(labels, index):
    missing = [n for scene in labels["scenes"] for n in scene if n not in index]
    if missing:
        sys.exit(f"labelled photo not in the folder: {missing[0]}")
    return [[index[n] for n in scene] for scene in labels["scenes"]]


def parse_args():
    parser = argparse.ArgumentParser(description="score the grouping against labels")
    parser.add_argument("folder", type=Path, help="the labelled album")
    parser.add_argument("--labels", type=Path,
                        default=Path(__file__).with_name("labels.json"))
    parser.add_argument("--cache", type=Path, help="npz file to reuse embeddings")
    parser.add_argument("--gap", type=float, default=180.0)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--show", type=float, default=0.75,
                        help="threshold whose mistakes to list")
    return parser.parse_args()


def main():
    args = parse_args()
    labels = json.loads(args.labels.read_text())

    found = gs.collect(args.folder, True)
    if not found:
        sys.exit(f"no images under {args.folder}")

    paths, vectors = gs.embed(found, gs.pick_device(args.device), args.cache)
    times = [gs.shot_time(p) for p in paths]
    index = {p.name: i for i, p in enumerate(paths)}
    name = {i: p.name for i, p in enumerate(paths)}

    truth_groups = load_truth(labels, index)
    truth = pairs_of(truth_groups)
    order = sorted(range(len(paths)), key=lambda i: times[i])
    similarity = gs.Similarity(vectors)

    print(f"{len(paths)} photos, {len(truth_groups)} labelled scenes, "
          f"{len(truth)} pairs that belong together")
    print(f"gap {args.gap:.0f}s\n")
    print(f"{'threshold':>9} {'scenes':>7} {'grouped':>8} "
          f"{'precision':>10} {'recall':>7} {'F1':>6} {'wrong':>6} {'missed':>7}")

    for step in range(55, 100, 5):
        threshold = step / 100
        predicted_groups = gs.cluster(order, similarity, times, threshold, args.gap)
        predicted = pairs_of(predicted_groups)
        precision, recall, f1 = score(truth, predicted)
        print(f"{threshold:>9.2f} {len(predicted_groups):>7} "
              f"{sum(len(g) for g in predicted_groups):>8} "
              f"{precision:>10.3f} {recall:>7.3f} {f1:>6.3f} "
              f"{len(predicted - truth):>6} {len(truth - predicted):>7}")

    predicted = pairs_of(gs.cluster(order, similarity, times, args.show, args.gap))
    wrong = sorted(predicted - truth,
                   key=lambda p: -float(similarity[p[0]][p[1]]))
    missed = sorted(truth - predicted,
                    key=lambda p: -float(similarity[p[0]][p[1]]))

    print(f"\nat threshold {args.show}, pairs put together that should not be:")
    for a, b in wrong[:10] or [()]:
        print(f"  {float(similarity[a][b]):.3f}  {name[a]}  +  {name[b]}")
    if not wrong:
        print("  none")

    print(f"\nat threshold {args.show}, pairs missed that belong together:")
    for a, b in missed[:10] or [()]:
        print(f"  {float(similarity[a][b]):.3f}  {name[a]}  +  {name[b]}")
    if not missed:
        print("  none")


if __name__ == "__main__":
    main()
