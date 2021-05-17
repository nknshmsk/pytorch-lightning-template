from PIL import Image, ImageOps
import os
from torch.utils.data import Dataset


class image_data_set(Dataset):
    def __init__(self, image_dir_path, role, transform):
        self.image_dir_path = image_dir_path
        self.data = [str(p) for p in image_dir_path.iterdir()]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path)
        return self.transform(image)

class image_data_set_by_json(Dataset):
    def __init__(self, json_path, role, transform=None):
        with open(json_path) as f:
            df = json.load(f)
        self.role = role
        self.transform = transform
        self.data = df[self.role]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path)
        return self.transform(image)



class image_data_set_with_mask(Dataset):
    def __init__(self, image_dir_list, mask_dir, transform):
        self.image_dir_list = image_dir_list
        self.images = []
        for image_dir in self.image_dir_list:
            for path in os.listdir(image_dir):
                self.images += [os.path.join(image_dir, path)]
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = os.path.basename(self.images[idx])
        image_name_none_ext, ext = os.path.splitext(image_name)
        mask_name = f"{image_name_none_ext}_mask{ext}"
        image = Image.open(self.images[idx])
        image = ImageOps.grayscale(image)
        if os.path.exists(os.path.join(self.mask_dir, mask_name)) is not True:
            mask = Image.fromarray(np.uint8(np.zeros(image.size)))
        else:
            mask = Image.open(os.path.join(self.mask_dir, mask_name))
            mask = ImageOps.grayscale(mask)
        return self.transform(image), self.transform(mask)

class image_data_set_with_label(Dataset):
    def __init__(self, image_dir_list, transform):
        self.image_dir_list = image_dir_list
        self.images = []
        for image_dir in self.image_dir_list:
            for path in os.listdir(image_dir["path"]):
                self.images += [{"path": os.path.join(image_dir["path"], path), "label": image_dir["label"]}]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_with_label = self.images[idx]
        image = Image.open(image_with_label['path'])
        image = ImageOps.grayscale(image)
        return self.transform(image), image_with_label['label']
