import json
import shutil
from pathlib import Path
from tqdm import tqdm


def extract_path(item):
    """
    PAD format:   ["data/xxx/image.tif", label]
    Recog format: "data/xxx/image.bmp"
    """
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)):
        return item[0]
    raise ValueError(f"Unknown item format: {item}")


def copy_test_images(split_json, dst_root, src_root="."):
    split_json = Path(split_json)
    dst_root = Path(dst_root)
    src_root = Path(src_root)

    with open(split_json, "r") as f:
        splits = json.load(f)

    test_data = splits["test"]

    # Flatten all test items first so tqdm can know the total length
    all_items = []
    for _, items in test_data.items():
        all_items.extend(items)

    copied = 0
    missing = 0

    for item in tqdm(all_items, desc=f"Copying from {split_json.name}", unit="img"):
        rel_path = Path(extract_path(item))

        src_path = src_root / rel_path
        dst_path = dst_root / rel_path

        if not src_path.exists():
            missing += 1
            tqdm.write(f"[Missing] {src_path}")
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        copied += 1

    print(f"\nFinished: {split_json}")
    print(f"Copied: {copied}")
    print(f"Missing: {missing}")
    print(f"Output folder: {dst_root}")


if __name__ == "__main__":
    copy_test_images(
        split_json="data/pad_splits.json",
        dst_root="test_data",
        src_root="."
    )

    copy_test_images(
        split_json="data/recog_splits.json",
        dst_root="test_data",
        src_root="."
    )

    # Copy split files too
    split_dst_dir = Path("test_data/data")
    split_dst_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2("data/pad_splits.json", split_dst_dir / "pad_splits.json")
    shutil.copy2("data/recog_splits.json", split_dst_dir / "recog_splits.json")

    print("\nCopied split JSON files.")