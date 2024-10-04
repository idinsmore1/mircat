import argparse
from pathlib import Path
from shutil import copy
from mircat_seg.core import segment_niftis as segment_niftis

def mircat_seg():
    print("Hello from mircat-seg!")
    print("Running segmentation...")
    print(segment_niftis)


def mircat_copy_models():
    parser = argparse.ArgumentParser(description="Copy models to the correct location")
    parser.add_argument(
        "model_dir", help="Directory containing models to copy", type=Path
    )
    args = parser.parse_args()
    model_dir = args.model_dir
    if not model_dir.is_dir():
        raise ValueError("Model directory does not exist")
    project_root = Path(__file__).parent
    destination_dir = project_root / "models"

    if not destination_dir.exists():
        destination_dir.mkdir(parents=True)

    for model_file in model_dir.iterdir():
        if model_file.is_file():
            copy(model_file, destination_dir)
    print(f"Models copied to {destination_dir}")