import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pleural_Thickening',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Hernia'
]

NUM_CLASSES = len(DISEASES)

def encode_labels(label_string):
    vector = np.zeros(NUM_CLASSES, dtype=np.float32)
    if label_string == 'No Finding':
        return vector
    for disease in label_string.split('|'):
        disease = disease.strip()
        if disease in DISEASES:
            idx = DISEASES.index(disease)
            vector[idx] = 1.0
    return vector


class CuraDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image Index'])

        with Image.open(img_path) as img:
            image = img.convert('RGB').copy() 
            
        if self.transform:
            image = self.transform(image)

        label_vector = encode_labels(row['Finding Labels'])

        return image, label_vector


