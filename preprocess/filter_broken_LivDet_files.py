from pathlib import Path

from PIL import Image


def find_and_handle_broken_files(
    directory_path, print_broken=True, delete_broken=False
):
    """
    Finds and handles image files that cannot be opened with PIL.Image.

    Args:
        directory_path (str or Path): The root directory to search for images.
        print_broken (bool): If True, prints the path of broken files.
        delete_broken (bool): If True, deletes the broken files.

    Returns:
        list: A list of paths to the broken files.
    """
    broken_files = []
    path = Path(directory_path)

    if not path.is_dir():
        print(f"Error: {directory_path} is not a valid directory.")
        return broken_files

    # Common image extensions to check. Modify this list if needed.
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}

    # Iterate through all files in the directory recursively
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_extensions:
            try:
                # Use context manager to ensure the file is closed properly
                with Image.open(p) as img:
                    img.convert("RGB")
                    # Optionally add img.load() if some errors aren't caught by convert
            except Exception as e:
                broken_files.append(str(p))

                if print_broken:
                    print(f"[BROKEN] {p} - Error: {e}")

                if delete_broken:
                    try:
                        p.unlink()
                        if print_broken:
                            print(f"[DELETED] {p}")
                    except Exception as del_err:
                        if print_broken:
                            print(f"[ERROR DELETING] {p} - Error: {del_err}")

    return broken_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find and handle broken image files.")
    parser.add_argument("--dir", type=str, required=True, help="Directory to scan")
    parser.add_argument(
        "--print",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to print broken files",
    )
    parser.add_argument("--delete", action="store_true", help="Delete broken files")

    args = parser.parse_args()

    print(f"Scanning {args.dir}...")
    broken = find_and_handle_broken_files(
        directory_path=args.dir, print_broken=args.print, delete_broken=args.delete
    )
    print(f"Scan complete. Found {len(broken)} broken files.")
