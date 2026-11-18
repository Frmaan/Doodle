import numpy as np
import torch
from torch.utils.data import Dataset


class SketchDataset(Dataset):

    def __init__(self, images_path, labels_path, indices_path, split):

        self.images = np.load(images_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")

        splits = np.load(indices_path)
        self.indices = splits[f"{split}_idx"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        actual_index = self.indices[index]

        image = self.images[actual_index].reshape(28, 28)
        label = self.labels[actual_index]

        image = torch.tensor(
            image.copy(),
            dtype=torch.float32
        ) / 255.0

        image = image.unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return image, label
