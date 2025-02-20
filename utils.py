import os
import random
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def limit_images_per_category(main_dir, max_images_per_category):
    for category in os.listdir(main_dir):
        category_path = os.path.join(main_dir, category)
        if not os.path.isdir(category_path):
            continue
        images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        if len(images) > max_images_per_category:
            images_to_keep = random.sample(images, max_images_per_category)
            images_to_remove = set(images) - set(images_to_keep)
            for img in images_to_remove:
                os.remove(os.path.join(category_path, img))
        print(f"Kategorie {category} zkrácena na {max_images_per_category} obrázků.")

def split_data_into_folders(main_dir, train_split, val_split, test_split, min_images_per_category):
    for category in os.listdir(main_dir):
        category_path = os.path.join(main_dir, category)
        if not os.path.isdir(category_path) or category in ["train", "val", "test"]:
            continue
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
        if len(images) < min_images_per_category:
            print(f"Varování: Kategorie '{category}' má pouze {len(images)} obrázků. Bude přeskočena.")
            continue
        random.shuffle(images)
        train_count = int(len(images) * train_split)
        val_count = int(len(images) * val_split)
        splits = {
            "train": images[:train_count],
            "val": images[train_count:train_count + val_count],
            "test": images[train_count + val_count:]
        }
        for split, split_images in splits.items():
            split_dir = os.path.join(main_dir, split, category)
            os.makedirs(split_dir, exist_ok=True)
            for img_name in split_images:
                src = os.path.join(category_path, img_name)
                dst = os.path.join(split_dir, img_name)
                shutil.copy(src, dst)
        print(f"Kategorie '{category}' rozdělena: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test.")

def load_data(main_dir):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    train_data = datasets.ImageFolder(os.path.join(main_dir, 'train'), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(main_dir, 'val'), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(main_dir, 'test'), transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    print(f"Train data: {len(train_data)} images")
    print(f"Validation data: {len(val_data)} images")
    print(f"Test data: {len(test_data)} images")
    return train_loader, val_loader, test_loader, train_data

