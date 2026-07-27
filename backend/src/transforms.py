import torchvision.transforms as transforms

CURA_MEAN = [0.4967, 0.4967, 0.4967]
CURA_STD  = [0.2478, 0.2478, 0.2478]

train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine( degrees=0, translate=(0.12, 0.12), scale=(0.90, 1.10) ),
    transforms.ColorJitter( brightness=0.25, contrast=0.25),
    transforms.ToTensor(),
    transforms.Normalize(CURA_MEAN, CURA_STD)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CURA_MEAN, CURA_STD)
])