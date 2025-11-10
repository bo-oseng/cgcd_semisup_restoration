import argparse
import json
import csv
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate metadata files for CDD_11 dataset")
    parser.add_argument(
        "--dataset_path", default="data/CDD_11/Train", help="Root directory that contains class folders"
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle class order before assigning labels")
    parser.add_argument(
        "--class_order",
        nargs="+",
        default=[
            "clear",
            "low",
            "haze",
            "snow",
            "rain",
            "haze_snow",
            "haze_rain",
            "low_haze",
            "low_snow",
            "low_rain",
            "low_haze_snow",
            "low_haze_rain",
        ],
        help="Explicit class folder ordering (provide names as positional list)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when shuffling classes")
    return parser.parse_args()


def load_class_order(path: Path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_class_dirs(dataset_path: Path, args):
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    dir_map = {d.name: d for d in class_dirs}

    def order_from_names(order_names, source_desc):
        remaining = dict(dir_map)
        ordered_dirs = []
        for name in order_names:
            if name not in remaining:
                raise ValueError(f"Class '{name}' from {source_desc} does not exist under {dataset_path}")
            ordered_dirs.append(remaining.pop(name))
        remaining_dirs = sorted(remaining.values(), key=lambda d: d.name)
        return ordered_dirs + remaining_dirs

    if args.class_order is not None:
        return order_from_names(args.class_order, "--class_order list")

    # Fallback to alphabetical order, with optional shuffle
    class_dirs = sorted(class_dirs, key=lambda d: d.name)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(class_dirs)
    return class_dirs


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    class_dirs = resolve_class_dirs(dataset_path, args)
    rng = random.Random(args.seed)

    entries = []
    class_names = {}

    for class_idx, class_dir in enumerate(class_dirs, start=0):
        for image_path in sorted(class_dir.rglob("*.png")):
            img_id = len(entries) + 1
            rel_path = image_path.relative_to(dataset_path).as_posix()
            entries.append((img_id, rel_path, class_idx))
        class_names[class_idx] = class_dir.name

    with open(dataset_path / "images.txt", "w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows((img_id, rel_path) for img_id, rel_path, _ in entries)

    with open(dataset_path / "image_class_labels.txt", "w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows((img_id, class_idx) for img_id, _, class_idx in entries)

    rng.shuffle(entries)
    mid = len(entries) // 2
    split_flags = {img_id: 1 if idx < mid else 0 for idx, (img_id, *_rest) in enumerate(entries)}

    with open(dataset_path / "train_test_split.txt", "w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows((img_id, split_flags[img_id]) for img_id, *_ in sorted(entries))

    with open(dataset_path / "class_id_to_name.json", "w") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
