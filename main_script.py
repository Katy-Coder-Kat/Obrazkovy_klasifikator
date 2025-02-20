import os
import torch
from utils import limit_images_per_category, split_data_into_folders, load_data
from data_augmentation import augment_images_in_folders
from model_training import train_model, evaluate_and_save_model, test_model_on_data

main_dir = r"data/raw/raw-img"
max_images_per_category = 100
train_split, val_split, test_split = 0.7, 0.2, 0.1
min_images_per_category = 10

if __name__ == "__main__":
    print("1. Omezování počtu obrázků...")
    limit_images_per_category(main_dir, max_images_per_category)

    print("\n2. Rozdělování obrázků do složek train/val/test...")
    split_data_into_folders(main_dir, train_split, val_split, test_split, min_images_per_category)

    print("\n3. Augmentace obrázků...")
    augment_images_in_folders(main_dir, ['train', 'val', 'test'])

    print("\n4. Načítání dat...")
    train_loader, val_loader, test_loader, train_data = load_data(main_dir)

    print("\n5. Trénování modelu...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = train_model(train_loader, val_loader, train_data, device)

    print("\n6. Validace a ukládání modelu...")
    torch.save(trained_model.state_dict(), "data/models/model_weights.pth")
    print("Model byl uložen jako 'data/models/model_weights.pth'.")

    print("\n7. Testování modelu na testovací sadě...")
    test_model_on_data(trained_model, test_loader, device)


def evaluate_and_save_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validační přesnost: {accuracy:.2f}%")
    torch.save(model.state_dict(), "model_weights.pth")
    print("Model byl uložen jako 'model_weights.pth'.")
    return accuracy


def test_model_on_data(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Přesnost na testovací sadě: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    print("1. Omezování počtu obrázků...")
    limit_images_per_category(main_dir, max_images_per_category)

    print("\n2. Rozdělování obrázků do složek train/val/test...")
    split_data_into_folders(main_dir, train_split, val_split, test_split, min_images_per_category)

    print("\n3. Načítání dat...")
    train_loader, val_loader, test_loader, train_data = load_data(main_dir)

    print("\n4. Trénování modelu...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = train_model(train_loader, val_loader, train_data, device)

    print("\n5. Validace a ukládání modelu...")
    evaluate_and_save_model(trained_model, val_loader, device)

    print("\n6. Testování modelu...")
    test_model_on_data(trained_model, test_loader, device)
