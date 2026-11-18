
import json
from pathlib import Path

import numpy as np


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def prepare_dataset(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    categories = sorted([
        file.stem
        for file in data_dir.glob("*.npy")
    ])

    if not categories:
        raise FileNotFoundError(
            f"No .npy category files found in {data_dir}"
        )

    print(f"Total categories: {len(categories)}")

    # Create a consistent category-to-index mapping
    class_to_idx = {
        category: index
        for index, category in enumerate(categories)
    }

    # Save mapping for future use
    with open(output_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)

    # Calculate total number of images
    total_images = sum(
        np.load(data_dir / f"{category}.npy", mmap_mode="r").shape[0]
        for category in categories
    )

    # Create memory-mapped arrays
    X = np.lib.format.open_memmap(
        output_dir / "images.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(total_images, 784)
    )

    y = np.lib.format.open_memmap(
        output_dir / "labels.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(total_images,)
    )

    start = 0

    for category in categories:

        label = class_to_idx[category]

        data = np.load(
            data_dir / f"{category}.npy",
            mmap_mode="r"
        )

        end = start + len(data)

        X[start:end] = data
        y[start:end] = label

        print(f"{category}: {len(data):,} images copied")

        start = end

    X.flush()
    y.flush()

    print("\nDataset preparation complete!")
    print(f"Images: {X.shape}")
    print(f"Labels: {y.shape}")
    print(f"Total images: {total_images}")
    print(f"Total categories: {len(categories)}")

    return class_to_idx
