import torchvision.transforms as transforms

CURA_MEAN = [0.4967, 0.4967, 0.4967]
CURA_STD  = [0.2478, 0.2478, 0.2478]

train_transform = transforms.Compose([
    transforms.RandomRotation(7),
    transforms.RandomAffine( degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05) ),
    transforms.ColorJitter( brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(CURA_MEAN, CURA_STD)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CURA_MEAN, CURA_STD)
])