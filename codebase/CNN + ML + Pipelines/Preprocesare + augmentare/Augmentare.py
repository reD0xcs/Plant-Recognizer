import os
import numpy as np
import albumentations as A
from tqdm import tqdm

def load_npy_image(path):
    """Incarca o imagine .npy salvata anterior"""
    try:
        img = np.load(path)
        if img.shape != (224, 224, 3):
            raise ValueError(f"Dimensiune invalida: {path}")
        return img
    except Exception as e:
        print(f"Eroare la citirea {path}: {e}")
        return None

def get_augmentations():
    """Definește pipeline-ul de augmentari"""
    return A.Compose([
        A.Rotate(limit=30, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.1, rotate_limit=0, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3)
    ])

def augment_and_save(img, aug_pipeline, output_dir, base_name, count):
    """Aplica augmentari si salveaza imaginea"""
    for i in range(count):
        augmented = aug_pipeline(image=img)["image"]
        filename = f"{base_name}_aug{i+1}.npy"
        path = os.path.join(output_dir, filename)
        np.save(path, augmented)

def process_folders(input_root, output_root, augmentations_per_image=3):
    """Parcurge folderele de specii si aplica augmentari"""
    aug_pipeline = get_augmentations()

    for species in tqdm(os.listdir(input_root), desc="Specii"):
        input_path = os.path.join(input_root, species)
        output_path = os.path.join(output_root, species)

        if not os.path.isdir(input_path):
            continue

        os.makedirs(output_path, exist_ok=True)

        for file in os.listdir(input_path):
            if not file.endswith(".npy"):
                continue

            img_path = os.path.join(input_path, file)
            base_name = os.path.splitext(file)[0]
            img = load_npy_image(img_path)

            if img is not None:
                augment_and_save(img, aug_pipeline, output_path, base_name, augmentations_per_image)

if __name__ == "__main__":
    input_dir = "C:/Users/catal/OneDrive/Desktop/cnn pipeline"
    output_dir = "C:/Users/catal/OneDrive/Desktop/cnn_augmented"

    process_folders(input_dir, output_dir)
    print("Augmentarile au fost generate cu succes.")
