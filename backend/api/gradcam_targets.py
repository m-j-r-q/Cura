def get_target_layer(model, architecture: str):
    if architecture == 'densenet121':
        return model.features.denseblock4.denselayer16.conv2

    elif architecture == 'densenet169':
        return model.features.denseblock4.denselayer32.conv2

    elif architecture == 'resnet50':
        return model.layer4[-1].conv3

    elif architecture == 'efficientnet_b0':
        return model.features[-1][0]

    elif architecture == 'convnext_tiny':
        last_cnblock = model.features[-2]
        return last_cnblock.block[5]

    else:
        raise ValueError(f"Unknown architecture: {architecture}")