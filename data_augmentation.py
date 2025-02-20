import os
from PIL import Image
from torchvision import transforms

processed_dir = r"C:\Users\ranoc\OneDrive\Desktop\obrazkovy editor\processed"
subdirs = ["train", "val", "test"]

def augment_images_in_folders(processed_dir, subdirs):
    augmentations = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ]

    for subdir in subdirs:
        subdir_path = os.path.join(processed_dir, subdir)

        if not os.path.exists(subdir_path):
            print(f"Složka '{subdir}' neexistuje, přeskočeno.")
            continue

        categories = [cat for cat in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, cat))]
        for category in categories:
            print(f"Augmentuji obrázky ve složce: {os.path.join(subdir, category)}")
            category_path = os.path.join(subdir_path, category)
            images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_name in images:
                img_path = os.path.join(category_path, img_name)
                image = Image.open(img_path)

                for idx, transform in enumerate(augmentations):
                    augmented_image = transform(image)
                    aug_img_name = f"aug_{idx}_{img_name}"
                    aug_img_path = os.path.join(category_path, aug_img_name)

                    augmented_image.save(aug_img_path)

            print(f"Augmentace pro kategorii '{category}' ve složce '{subdir}' dokončena.")

augment_images_in_folders(processed_dir, subdirs)
