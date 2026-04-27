from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Import your transform
from ..transforms import ExtractFingerprint


def process_dataset(
    input_dir: str,
    output_dir: str,
    padding: int = 10,
    extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    extractor = ExtractFingerprint(padding=padding)

    # Collect all image files
    image_files = [
        p for p in input_path.rglob("*")
        if p.suffix.lower() in extensions
    ]

    print(f"Found {len(image_files)} images.")

    for img_path in tqdm(image_files):
        try:
            # Load image
            img = Image.open(img_path).convert("RGB")

            # Apply extraction
            processed = extractor(img)

            # Create output path (preserve folder structure)
            relative_path = img_path.relative_to(input_path)
            save_path = output_path / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save image
            processed.save(save_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract fingerprint foreground from dataset")
    parser.add_argument("--input", required=True, help="Input dataset directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--padding", type=int, default=10)

    args = parser.parse_args()

    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        padding=args.padding,
    )