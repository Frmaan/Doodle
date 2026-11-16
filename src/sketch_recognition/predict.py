from pathlib import Path

import torch
from sketch_recognition.models import SketchCNN

BASE_DIR = Path(__file__).resolve().parents[2]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = BASE_DIR / "models" / "sketch_cnn_32.pth"

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model = SketchCNN(
    num_classes = checkpoint["num_classes"]
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

idx_to_class = {
    index: class_name
    for class_name, index in checkpoint["class_to_idx"].items()
}

print(f"Model loaded successfully on {DEVICE}")
print(f"Classes : {idx_to_class}")


def predict_sketch(processed_image):
    image_tensor = torch.from_numpy(
        processed_image
    ).unsqueeze(0).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    probabilities = probabilities.squeeze(0).cpu().numpy()

    results = {
        idx_to_class[i]: float(probabilities[i])
        for i in range(len(idx_to_class))
    }

    return results