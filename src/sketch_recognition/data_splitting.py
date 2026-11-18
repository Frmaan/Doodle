
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def create_splits(data_dir=DATA_DIR, random_state=42):

    data_dir = Path(data_dir)

    labels = np.load(
        data_dir / "labels.npy",
        mmap_mode="r"
    )

    indices = np.arange(len(labels))

    # Separate training data from validation + testing
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=random_state,
        stratify=labels
    )

    # Divide remaining data equally into validation and testing
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=random_state,
        stratify=labels[temp_idx]
    )

    # Save split indices
    np.savez(
        data_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx
    )

    print(f"Training samples: {len(train_idx):,}")
    print(f"Validation samples: {len(val_idx):,}")
    print(f"Testing samples: {len(test_idx):,}")

    return train_idx, val_idx, test_idx
