import torch
import torch.nn as nn
from torchvision import transforms, models

MODEL_PATH = "output/best_model.pth"
THRESHOLD = 0.74

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(model_path=MODEL_PATH, device=DEVICE):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_face(model, face_rgb, device=DEVICE):
    """Run the classifier on a single RGB uint8 face crop.

    Returns the fake probability (float in [0, 1]).
    """
    tensor = _transform(face_rgb).unsqueeze(0).to(device)
    output = model(tensor)
    prob = torch.sigmoid(output).item()
    return prob
