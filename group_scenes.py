#!/usr/bin/env python3
"""Group photos that show the same scene.

Each photo is turned into a DINOv2 embedding, which describes what is in
the frame rather than how its pixels are laid out. That survives a step
sideways or a small pan, where a perceptual hash does not.

A photo joins a group only if it was taken close in time to the previous
shot and is similar enough to every member already in that group.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

MODEL = "facebook/dinov2-base"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff"}
BATCH = 16
EXIF_IFD = 0x8769
DATE_TAKEN = 36867

SUBJECT = "XMP:Subject"
HIERARCHICAL = "XMP-lr:HierarchicalSubject"

# Scene_20200523_134509, the same with a collision suffix, or the plain
# number that versions before stable names used to write
SCENE_TAG = re.compile(r"^Scene_(?:\d{8}_\d{6}(?:_\d+)?|\d+)$")

# keeps an exiftool command line well under the system argument limit
ARG_BUDGET = 100_000


def log(message):
    print(message, file=sys.stderr)


def plural(count, noun):
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def collect(folder, recursive):
    walk = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in walk if p.suffix.lower() in EXTENSIONS)


def shot_time(path):
    try:
        exif = Image.open(path).getexif()
        raw = exif.get_ifd(EXIF_IFD).get(DATE_TAKEN) or exif.get(306)
        if raw:
            return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S").timestamp()
    except (OSError, ValueError, UnidentifiedImageError):
        pass
    return path.stat().st_mtime


def stamp(path):
    info = path.stat()
    return f"{path}|{info.st_mtime_ns}|{info.st_size}"


def load_cache(cache_path):
    if not cache_path or not cache_path.exists():
        return {}
    try:
        stored = np.load(cache_path, allow_pickle=False)
        return dict(zip(stored["stamps"].tolist(), stored["vectors"]))
    except Exception as error:
        log(f"cache ignored, {error}")
        return {}


def save_cache(cache_path, known):
    if not cache_path or not known:
        return
    np.savez(
        cache_path,
        stamps=np.array(list(known), dtype=object).astype(str),
        vectors=np.stack(list(known.values())),
    )


def embed(paths, device, cache_path):
    known = load_cache(cache_path)
    stamps = [stamp(p) for p in paths]
    missing = [p for p, s in zip(paths, stamps) if s not in known]

    if missing:
        log(f"embedding {len(missing)} of {len(paths)} photos on {device}")
        processor = AutoImageProcessor.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL).to(device).eval()

        for start in range(0, len(missing), BATCH):
            batch = missing[start : start + BATCH]
            images = []
            usable = []
            for path in batch:
                try:
                    images.append(Image.open(path).convert("RGB"))
                    usable.append(path)
                except (OSError, UnidentifiedImageError) as error:
                    log(f"skipped {path.name}, {error}")

            if not images:
                continue

            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model(**inputs)

            # CLS token carries the global scene descriptor
            vectors = F.normalize(output.last_hidden_state[:, 0].float(), dim=1)
            for path, vector in zip(usable, vectors.cpu().numpy()):
                known[stamp(path)] = vector

            log(f"{min(start + BATCH, len(missing))}/{len(missing)}")

        save_cache(cache_path, known)
    else:
        log(f"all {len(paths)} embeddings came from cache")

    kept = [(p, known[s]) for p, s in zip(paths, stamps) if s in known]
    if not kept:
        sys.exit("no readable images")
    return [p for p, _ in kept], np.stack([v for _, v in kept])


class Row:
    __slots__ = ("one", "rest")

    def __init__(self, one, rest):
        self.one = one
        self.rest = rest

    def __getitem__(self, other):
        return float(self.one @ self.rest[other])


class Similarity:
    """Cosine similarity between any two photos, without the full matrix.

    Clustering asks for a few hundred pairs out of n squared, and the rest
    of the matrix is built only to be thrown away. This keeps the same
    similarity[i][j] reading and does the dot product when it is asked for.
    Vectors are unit length, so the dot product is the cosine.
    """

    __slots__ = ("vectors",)

    def __init__(self, vectors):
        self.vectors = vectors

    def __getitem__(self, index):
        return Row(self.vectors[index], self.vectors)


def cluster(order, similarity, times, threshold, gap):
    groups = []
    current = []

    for index in order:
        if current:
            in_time = times[index] - times[current[-1]] <= gap
            fits_all = all(similarity[index][m] >= threshold for m in current)
            if in_time and fits_all:
                current.append(index)
                continue
            groups.append(current)
        current = [index]

    if current:
        groups.append(current)
    return [g for g in groups if len(g) > 1]


def scene_names(scenes, times):
    """Name each scene after the capture time of its earliest member.

    A position in the list means something different as soon as the album
    gains a photo, so tags written by an earlier run would start pointing
    at the wrong group. A capture time does not move.
    """
    names = []
    taken = set()

    for members in scenes:
        start = datetime.fromtimestamp(min(times[i] for i in members))
        base = f"Scene_{start:%Y%m%d_%H%M%S}"
        name = base
        copy = 2
        while name in taken:
            name = f"{base}_{copy}"
            copy += 1
        taken.add(name)
        names.append(name)

    return names


def chunks(paths):
    batch = []
    budget = ARG_BUDGET

    for path in paths:
        room = len(str(path)) + 1
        if batch and room > budget:
            yield batch
            batch, budget = [], ARG_BUDGET
        batch.append(path)
        budget -= room

    if batch:
        yield batch


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def read_tags(paths):
    found = {}

    for batch in chunks(paths):
        command = ["exiftool", "-json", f"-{SUBJECT}", f"-{HIERARCHICAL}"]
        command += [str(p) for p in batch]
        done = subprocess.run(command, capture_output=True, text=True)
        if not done.stdout.strip():
            sys.exit(f"exiftool could not read tags, {done.stderr.strip()}")

        for entry in json.loads(done.stdout):
            found[entry["SourceFile"]] = (
                as_list(entry.get("Subject")),
                as_list(entry.get("HierarchicalSubject")),
            )

    return found


def plan(tag, present, target):
    """Turn the values already in a file into exiftool operations.

    A value of ours that is not the one we want goes. The one we want is
    added only if it is missing, and removed first if it sits there twice,
    because exiftool += appends without looking.
    """
    ops = [f"-{tag}-={value}" for value in sorted(set(present) - {target})]

    if target:
        if present.count(target) > 1:
            ops.append(f"-{tag}-={target}")
        if present.count(target) != 1:
            ops.append(f"-{tag}+={target}")

    return ops


def tag_ops(subject, hierarchical, name, parent):
    prefix = f"{parent}|"
    mine = [v for v in subject if SCENE_TAG.match(v)]
    ops = plan(SUBJECT, mine, name)

    mine = [
        v for v in hierarchical
        if v.startswith(prefix) and SCENE_TAG.match(v[len(prefix):])
    ]
    ops += plan(HIERARCHICAL, mine, prefix + name if name else None)

    return tuple(ops)


def run_exiftool(ops, files):
    problems = []

    for batch in chunks(files):
        command = ["exiftool", "-overwrite_original", *ops]
        command += [str(f) for f in batch]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            problems.append(done.stderr.strip())

    return problems


def write_tags(paths, scenes, names, parent):
    if not shutil.which("exiftool"):
        sys.exit("exiftool not found, see the README for install hints")

    present = read_tags(paths)
    problems = []

    def batch_by_ops(members, name):
        batches = {}
        for index in members:
            subject, hierarchical = present.get(str(paths[index]), ([], []))
            ops = tag_ops(subject, hierarchical, name, parent)
            batches.setdefault(ops, []).append(paths[index])
        return batches

    def execute(batches):
        changed = 0
        for ops, files in batches.items():
            if not ops:
                continue
            problems.extend(run_exiftool(ops, files))
            changed += len(files)
        return changed

    written = untouched = 0

    for members, name in zip(scenes, names):
        batches = batch_by_ops(members, name)
        changed = execute(batches)
        written += changed
        untouched += len(members) - changed

        if changed and changed < len(members):
            log(f"{name} tagged {plural(changed, 'file')}, "
                f"{len(members) - changed} already tagged")
        elif changed:
            log(f"{name} tagged {plural(changed, 'file')}")
        else:
            log(f"{name} already tagged")

    grouped = {index for members in scenes for index in members}
    loose = [i for i in range(len(paths)) if i not in grouped]
    cleared = execute(batch_by_ops(loose, None))
    if cleared:
        log(f"cleared scene tags from {plural(cleared, 'file')} "
            f"of {len(loose)} no longer grouped")

    log(f"{plural(written + cleared, 'file')} changed, "
        f"{untouched} already correct")
    for problem in problems:
        log(f"exiftool: {problem}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="group photos that show the same scene",
        epilog="run without --write first, check the report, then tag",
    )
    parser.add_argument("folder", type=Path, help="folder with photos")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="cosine similarity needed to join a group")
    parser.add_argument("--gap", type=float, default=180.0,
                        help="seconds allowed between consecutive shots")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--cache", type=Path,
                        help="npz file to reuse embeddings between runs")
    parser.add_argument("--flat", action="store_true",
                        help="skip subfolders")
    parser.add_argument("--parent", default="Scenes", help="parent tag name")
    parser.add_argument("--write", action="store_true",
                        help="write tags into the files with exiftool")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.folder.is_dir():
        sys.exit(f"not a folder: {args.folder}")

    found = collect(args.folder, not args.flat)
    if not found:
        sys.exit(f"no images under {args.folder}")

    paths, vectors = embed(found, pick_device(args.device), args.cache)
    times = [shot_time(p) for p in paths]
    similarity = Similarity(vectors)

    order = sorted(range(len(paths)), key=lambda i: times[i])
    scenes = cluster(order, similarity, times, args.threshold, args.gap)
    names = scene_names(scenes, times)

    grouped = sum(len(s) for s in scenes)
    print(f"{len(paths)} photos, {len(scenes)} scenes, {grouped} photos grouped")
    print(f"threshold {args.threshold}, gap {args.gap:.0f}s")

    for members, name in zip(scenes, names):
        pairs = [
            similarity[a][b]
            for i, a in enumerate(members)
            for b in members[i + 1 :]
        ]
        span = times[members[-1]] - times[members[0]]
        print(f"\n{name} ({len(members)} photos, "
              f"min {min(pairs):.3f}, span {span:.0f}s)")
        for index in members:
            print(f"  {paths[index].name}")

    if args.write:
        write_tags(paths, scenes, names, args.parent)


if __name__ == "__main__":
    main()
