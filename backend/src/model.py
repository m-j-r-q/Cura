import torch
import torch.nn as nn
import torchvision.models as models
from dataset import NUM_CLASSES


def add_internal_dropout(model, dropout_p=0.2):
    """
    Inserts standard element-wise Dropout after every conv2 in each _DenseLayer
    Safe for low growth-rate architectures like DenseNet.
    """
    from torchvision.models.densenet import _DenseLayer

    for name, module in model.named_modules():
        if isinstance(module, _DenseLayer):
            original_conv2 = module.conv2
            module.conv2 = nn.Sequential(
                original_conv2,
                nn.Dropout(p=dropout_p)
            )

    return model



def build_model(
    architecture: str,
    pretrained: bool = True,
    dropout_p: float = 0.5,
    internal_dropout_p: float = 0.2
):
    weights = 'IMAGENET1K_V1' if pretrained else None

    if architecture == 'resnet50':
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, NUM_CLASSES)
        )

    elif architecture == 'densenet121':
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, NUM_CLASSES)
        )
        if internal_dropout_p > 0:
            model = add_internal_dropout(model, internal_dropout_p)

    elif architecture == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, NUM_CLASSES)
        )

    elif architecture == 'convnext_tiny':
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, NUM_CLASSES)
        )
    elif architecture == 'densenet169':
        model = models.densenet169(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, NUM_CLASSES)
        )
        if internal_dropout_p > 0:
            model = add_internal_dropout(model, internal_dropout_p)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model