import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

LABEL_MAP = {
    1: 0, 2: 1, 3: 2, 4: 3,
    5: 4, 6: 5, 7: 6
}

class RAFDBDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_name = row.iloc[0]
        raw_label = int(row.iloc[1])
        label = LABEL_MAP[raw_label]

        img_path = os.path.join(
            self.root_dir,
            str(raw_label),
            img_name
        )

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
