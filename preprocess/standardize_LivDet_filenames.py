import re
import shutil
from collections import defaultdict
from pathlib import Path


def standardize_livdet2013_filenames(
    input_dir, output_dir=None, dry_run=False, map_fingers_to_int=False
):
    """
    Standardize the filenames in LivDet 2013 protocol.
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

    # Pattern components:
    # ^(\d+)              : Subject ID (e.g., 031)
    # (?:\.\d*)?          : Optional ignored part like .1 or . (e.g., 031.1, 082.)
    # ([A-Za-z]+?)        : Acquisition mode (e.g., Tam, Trr, T)
    # ([LR])              : Hand (L or R)
    # (thb|idx|mdl|rng|ltl) : Finger
    # (?:Bmk|Itd)         : Bmk or Itd string
    # (\.[A-Za-z0-9]+)?$  : Optional extension
    pattern = re.compile(
        r"^(\d+)(?:\.\d*)?([A-Za-z]+?)([LR])(thb|idx|mdl|rng|ltl)(?:Bmk|Itd)(\.[A-Za-z0-9]+)?$"
    )

    # Optional mapping to integer as it is common in LivDet/FVC (1-10)
    finger_to_int = {
        "Rthb": "1",
        "Ridx": "2",
        "Rmdl": "3",
        "Rrng": "4",
        "Rltl": "5",
        "Lthb": "6",
        "Lidx": "7",
        "Lmdl": "8",
        "Lrng": "9",
        "Lltl": "10",
    }

    # Use a dictionary to keep track of the acquisition counter for each subject_finger pair
    counter = defaultdict(int)

    # Ensure deterministic numbering across executions by sorting file paths
    # rglob('*') recursively finds all files and directories
    filepaths = sorted(input_path.rglob("*"))

    processed_count = 0
    skipped_count = 0
    current_folder = None

    for filepath in filepaths:
        if not filepath.is_file():
            continue

        # Reset counter when moving to a new folder
        if current_folder != filepath.parent:
            current_folder = filepath.parent
            counter.clear()

        filename = filepath.name
        if filename.startswith("."):
            continue

        match = pattern.match(filename)
        if not match:
            print(
                f"Skipping: {filepath.relative_to(input_path)} (does not match expected pattern)"
            )
            skipped_count += 1
            continue

        subject_id = match.group(1)
        # mode = match.group(2) # we don't need the acquisition mode in the final name
        hand = match.group(3)
        finger = match.group(4)
        ext = match.group(5) if match.group(5) else ""

        finger_str = f"{hand}{finger}"

        # Optional: Map Finger to numerical ID
        if map_fingers_to_int:
            finger_str = finger_to_int.get(finger_str, finger_str)

        # Determine acquisition number
        key = f"{subject_id}_{finger_str}"
        counter[key] += 1
        acq_num = counter[key]

        new_filename = f"{subject_id}_{finger_str}_{acq_num}{ext}"

        # Maintain directory structure
        rel_dir = filepath.parent.relative_to(input_path)

        if output_dir:
            out_dir_path = output_path / rel_dir
            if not dry_run:
                out_dir_path.mkdir(parents=True, exist_ok=True)
            new_filepath = out_dir_path / new_filename
        else:
            new_filepath = filepath.parent / new_filename

        # rel_old = filepath.relative_to(input_path)
        # rel_new = new_filepath.relative_to(output_path if output_dir else input_path)
        # print(f"Renaming: {rel_old} -> {rel_new}")

        if not dry_run:
            if filepath != new_filepath:
                if output_dir:
                    shutil.copy2(filepath, new_filepath)
                else:
                    # Rename in place
                    if new_filepath.exists():
                        print(
                            f"  [!] Warning: {new_filepath.name} already exists! Overwriting."
                        )
                    filepath.rename(new_filepath)

        processed_count += 1

    print(f"Summary: processed {processed_count} files, skipped {skipped_count} files.")


def standardize_livdet2011_filenames(input_dir, output_dir=None, dry_run=False):
    """
    Standardize the filenames in LivDet 2011 protocol.
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

        # Reset counter when moving to a new folder
        if current_folder != filepath.parent:
            current_folder = filepath.parent
            counter.clear()

        filename = filepath.name
        if filename.startswith("."):
            continue

        subject_id = None
        finger_num = None
        ext = ""
        hand_prefix = None

        # Strategy 1: Timestamp format (robust to variable fields inside the string)
        ts_match = re.search(r"_([LR]?\d{10,15})_", filename)
        if ts_match:
            ts_str = ts_match.group(1)
            if ts_str[0] in ["L", "R"]:
                hand_prefix = ts_str[0]

            before_ts = filename[: ts_match.start()]
            after_ts = filename[ts_match.end() :]

            parts = after_ts.rsplit(".", 1)
            ext = "." + parts[1] if len(parts) == 2 else ""
            after_ts_core = parts[0]

            after_parts = after_ts_core.split("_")
            if len(after_parts) >= 1:
                subject_id = before_ts.strip("_")
                if (
                    after_parts[0].lower() in ["enroll", "auth"]
                    and len(after_parts) >= 2
                ):
                    finger_num = after_parts[1].strip(" .")
                else:
                    finger_num = after_parts[0].strip(" .")

        # Strategy 2: Windows copy format e.g., 2102157_R1 (9).bmp
        elif " (" in filename:
            match = re.match(r"^(.+)_([^_]+)\s*\(\d+\)(\.\w+)?$", filename)
            if match:
                subject_id = match.group(1).strip("_")
                finger_num = match.group(2).strip(" .")
                ext = match.group(3) if match.group(3) else ""

        # Strategy 3: Standard short format e.g., 8144665_R2.bmp
        else:
            match = re.match(r"^(.+)_([^_]+?)(\.\w+)?$", filename)
            if match:
                subject_id = match.group(1).strip("_")
                finger_num = match.group(2).strip(" .")
                ext = match.group(3) if match.group(3) else ""

        if not subject_id or not finger_num:
            print(
                f"Skipping: {filepath.relative_to(input_path)} (does not match expected pattern)"
            )
            skipped_count += 1
            continue

        # Ensure any trailing optional _d parts inside the subject ID are removed
        subject_id = subject_id.split("_")[0]

        # If finger_num is purely digits (e.g., '1' or '2'), use hand_prefix or default to 'R'
        if finger_num.isdigit():
            finger_num = (hand_prefix if hand_prefix else "R") + finger_num

        # Use an internal tracking counter to deterministically assign the AcquisitionNumber
        key = f"{subject_id}_{finger_num}"
        counter[key] += 1
        acq_num = counter[key]

        new_filename = f"{subject_id}_{finger_num}_{acq_num}{ext}"

        rel_dir = filepath.parent.relative_to(input_path)

        if output_dir:
            out_dir_path = output_path / rel_dir
            if not dry_run:
                out_dir_path.mkdir(parents=True, exist_ok=True)
            new_filepath = out_dir_path / new_filename
        else:
            new_filepath = filepath.parent / new_filename

        if not dry_run:
            if filepath != new_filepath:
                if output_dir:
                    shutil.copy2(filepath, new_filepath)
                else:
                    if new_filepath.exists():
                        print(
                            f"  [!] Warning: {new_filepath.name} already exists! Overwriting."
                        )
                    filepath.rename(new_filepath)

        processed_count += 1

    print(f"Summary: processed {processed_count} files, skipped {skipped_count} files.")


def standardize_livdet2009_filenames(input_dir, output_dir=None, dry_run=False):
    """
    Standardize the filenames in LivDet 2009 protocol.
    Expected pattern: M_N_[1/2].extension
    where M is the finger id, N is the acquisition number,
    [1/2] is 1 for 0 seconds, 2 for 5 seconds.
    Outputs: fingerid_impression.extension
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

        # Reset counter when moving to a new folder
        if current_folder != filepath.parent:
            current_folder = filepath.parent
            counter.clear()

        filename = filepath.name
        if filename.startswith("."):
            continue

        match = re.match(r"^([^_]+)_(\d+)_([12])(\.[A-Za-z0-9]+)?$", filename)
        if not match:
            print(
                f"Skipping: {filepath.relative_to(input_path)} (does not match expected pattern)"
            )
            skipped_count += 1
            continue

        finger_id = match.group(1)
        ext = match.group(4) if match.group(4) else ""

        counter[finger_id] += 1
        impression = counter[finger_id]

        new_filename = f"{finger_id}_{impression}{ext}"

        rel_dir = filepath.parent.relative_to(input_path)

        if output_dir:
            out_dir_path = output_path / rel_dir
            if not dry_run:
                out_dir_path.mkdir(parents=True, exist_ok=True)
            new_filepath = out_dir_path / new_filename
        else:
            new_filepath = filepath.parent / new_filename

        if not dry_run:
            if filepath != new_filepath:
                if output_dir:
                    shutil.copy2(filepath, new_filepath)
                else:
                    if new_filepath.exists():
                        print(
                            f"  [!] Warning: {new_filepath.name} already exists! Overwriting."
                        )
                    filepath.rename(new_filepath)

        processed_count += 1

    print(f"Summary: processed {processed_count} files, skipped {skipped_count} files.")


def standardize_livdet2009_filenames1(input_dir, output_dir=None, dry_run=False):
    """
    Standardize the filenames in LivDet time-series format.
    Expected folder structure: Category / subject_finger_... / (0s.bmp, 2s.bmp, etc)
    Outputs: subject_finger_impression.extension
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

        filename = filepath.name
        if filename.startswith("."):
            continue

        # Reset counter per folder because the user wants [1/2] per folder
        if current_folder != filepath.parent:
            current_folder = filepath.parent
            counter.clear()

        # Check if parent folder matches the expected format subject_finger...
        parent_folder = filepath.parent.name
        # Match subject and finger (finger is always R/L followed by a digit), ignore remainder
        match = re.match(r"^-?(.+?)_+([RL]\d)", parent_folder)
        if not match:
            print(
                f"Skipping: {filepath.relative_to(input_path)} (parent folder '{parent_folder}' does not match expected pattern)"
            )
            skipped_count += 1
            continue

        subject_id = match.group(1).split("_")[0]
        finger_id = match.group(2)
        ext = filepath.suffix

        key = f"{subject_id}_{finger_id}"
        counter[key] += 1
        impression = counter[key]

        new_filename = f"{subject_id}_{finger_id}_{impression}{ext}"

        # Maintain directory structure
        rel_dir = filepath.parent.relative_to(input_path)

        if output_dir:
            out_dir_path = output_path / rel_dir
            if not dry_run:
                out_dir_path.mkdir(parents=True, exist_ok=True)
            new_filepath = out_dir_path / new_filename
        else:
            new_filepath = filepath.parent / new_filename

        if not dry_run:
            if filepath != new_filepath:
                if output_dir:
                    shutil.copy2(filepath, new_filepath)
                else:
                    if new_filepath.exists():
                        print(
                            f"  [!] Warning: {new_filepath.name} already exists! Overwriting."
                        )
                    filepath.rename(new_filepath)

        processed_count += 1

    print(f"Summary: processed {processed_count} files, skipped {skipped_count} files.")


if __name__ == "__main__":
    standardize_livdet2009_filenames("/data/LivDet/LivDet2009/Biometrika")
    standardize_livdet2009_filenames1("/data/LivDet/LivDet2009/CrossMatch")
    standardize_livdet2009_filenames1("/data/LivDet/LivDet2009/Identix")
    standardize_livdet2011_filenames("/data/LivDet/LivDet2011/Digital")
    standardize_livdet2011_filenames("/data/LivDet/LivDet2011/Sagem")
    standardize_livdet2013_filenames("/data/LivDet/LivDet2013/Biometrika")
    standardize_livdet2013_filenames("/data/LivDet/LivDet2013/Italdata")
        