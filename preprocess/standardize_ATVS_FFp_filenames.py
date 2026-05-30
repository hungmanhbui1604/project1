import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

def standardize_ATVS_FFp_filenames(input_dir, output_dir=None, dry_run=False):
    """
    Standardize the ATVS-FFp dataset filenames and folders.
    1. Renames 'fake' -> 'spoof', 'original' -> 'live'
    2. Corrects user id in filename with folder name
    3. Renames files to userid_finger_impression format
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Directory {input_dir} does not exist.")
        return

    if output_dir:
        output_path = Path(output_dir)
        if not dry_run:
            output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path

    filepaths = sorted(input_path.rglob("*"))
    processed_count = 0
    skipped_count = 0

    counter = defaultdict(int)
    current_folder = None

    for filepath in filepaths:
        if not filepath.is_file():
            continue

        if filepath.name.startswith(".") or filepath.name == "Thumbs.db":
            continue

        rel_path = filepath.relative_to(input_path)
        rel_parts = list(rel_path.parts)
        
        # Look for u\d+ folder in path
        u_idx = -1
        for i, part in enumerate(rel_parts[:-1]):
            if re.match(r"^u\d+$", part):
                u_idx = i
                break
                
        if u_idx == -1:
            print(f"Skipping: {rel_path} (no u\\d folder found)")
            skipped_count += 1
            continue

        u_folder = rel_parts[u_idx]
        # user_id extracted from folder name (corrects potential errors in original filename)
        user_id = u_folder.replace("u", "")

        label_folder = rel_parts[u_idx+1]
        if label_folder == "original":
            new_label_folder = "Live"
        elif label_folder == "fake":
            new_label_folder = "Spoof"
        elif label_folder in ["Live", "Spoof"]:
            new_label_folder = label_folder
        else:
            print(f"Skipping: {rel_path} (unknown label folder '{label_folder}')")
            skipped_count += 1
            continue
            
        rel_parts[u_idx+1] = new_label_folder

        filename = filepath.name
        ext = filepath.suffix
        
        # Extract finger (CD) from original filename: uXX_A_BB_CD_YY
        name_parts = filepath.stem.split('_')
        if len(name_parts) >= 5:
            finger = name_parts[3]
        else:
            print(f"Skipping: {rel_path} (unexpected filename format)")
            skipped_count += 1
            continue
            
        # Reset counter per folder (since different sensors are mapped into the same new label folder iteratively)
        if current_folder != filepath.parent:
            current_folder = filepath.parent
            counter.clear()
            
        key = f"{user_id}_{finger}"
        counter[key] += 1
        impression = counter[key]
        
        new_filename = f"{user_id}_{finger}_{impression}{ext}"
        new_rel_path = Path(*rel_parts[:-1]) / new_filename
        
        if output_dir:
            new_filepath = output_path / new_rel_path
            if not dry_run:
                new_filepath.parent.mkdir(parents=True, exist_ok=True)
        else:
            new_filepath = input_path / new_rel_path
            if not dry_run:
                new_filepath.parent.mkdir(parents=True, exist_ok=True)
                
        rel_old = filepath.relative_to(input_path)
        rel_new = new_filepath.relative_to(output_path if output_dir else input_path)
        if dry_run:
            print(f"Renaming: {rel_old} -> {rel_new}")

        if not dry_run:
            if filepath != new_filepath:
                if output_dir:
                    shutil.copy2(filepath, new_filepath)
                else:
                    if new_filepath.exists():
                        print(f"  [!] Warning: {new_filepath.name} already exists! Overwriting.")
                    filepath.rename(new_filepath)

        processed_count += 1

    if not output_dir and not dry_run:
        # Clean up original/fake folders if we did in-place processing
        for u_dir in input_path.rglob("u*"):
            if u_dir.is_dir() and re.match(r"^u\d+$", u_dir.name):
                for old_label in ["original", "fake"]:
                    old_path = u_dir / old_label
                    if old_path.exists() and old_path.is_dir():
                        shutil.rmtree(old_path)

    print(f"Summary: processed {processed_count} files, skipped {skipped_count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize ATVS-FFp filenames.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to ATVS-FFp dataset.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (optional).")
    parser.add_argument("--dry_run", action="store_true", help="Print actions without modifying files.")
    
    args = parser.parse_args()
    standardize_ATVS_FFp_filenames(args.input_dir, args.output_dir, args.dry_run)
