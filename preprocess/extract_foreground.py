import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Import your transform
from transforms import ExtractFingerprintForeground


def process_dataset(
    input_dir: str,
    output_dir: str,
    padding: int = 10,
    extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    extractor = ExtractFingerprintForeground(padding=padding)

    # Collect ALL files (not just images)
    all_files = [p for p in input_path.rglob("*") if p.is_file()]

    print(f"Found {len(all_files)} total files.")

    for file_path in tqdm(all_files):
        try:
            relative_path = file_path.relative_to(input_path)
            save_path = output_path / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # If it's an image → process
            if file_path.suffix.lower() in extensions:
                img = Image.open(file_path).convert("RGB")
                processed = extractor(img)
                processed.save(save_path)

            # Otherwise → copy as-is
            else:
                shutil.copy2(file_path, save_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract fingerprint foreground from dataset"
    )
    parser.add_argument("--input", required=True, help="Input dataset directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--padding", type=int, default=10)

    args = parser.parse_args()

    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        padding=args.padding,
    )
