import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights

def build_model(num_classes=7):
    weights = EfficientNet_B4_Weights.DEFAULT
    model = models.efficientnet_b4(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )

    return model
