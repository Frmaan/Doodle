
import numpy as np
from PIL import Image


def preprocess_drawing(drawing):
    if drawing is None or drawing["composite"] is None:
        return None

    # Get grayscale image
    image = drawing["composite"]

    # Invert
    image = 255 - image

    # Find pixels of the drawing
    coords = np.argwhere(image > 0)

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = image[y_min:y_max + 1, x_min:x_max + 1]

    height, width = cropped.shape

    margin = max(1, int(max(height, width) * 0.1))
    side = max(height, width) + (2 * margin)

    top = (side - height) // 2
    bottom = side - height - top

    left = (side - width) // 2
    right = side - width - left

    canvas = np.pad(
        cropped,
        ((top, bottom), (left, right)),
        mode="constant",
        constant_values=0
    )

    resized = Image.fromarray(canvas).resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    processed = np.asarray(resized, dtype=np.float32) / 255.0

    return processed
