import cv2
import numpy as np
import os
from tqdm import tqdm


def preprocess_image(image_path):
    """
    Preproceseaza o imagine:
    - citeste imaginea originala
    - converteste in BGR daca este grayscale sau PNG cu canal alpha
    - redimensioneaza la 224x224 cu pastrarea aspectului
    - corecteaza umbrele folosind CLAHE in spatiul HSV
    - reduce zgomotul folosind filtru bilateral
    - normalizeaza valorile pixelilor in intervalul [0, 1]
    - valideaza forma finala
    """
    try:
        # 1. Citire imagine (ne asiguram ca are 3 canale)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Imagine invalida: {image_path}")
            return None

        # 2. Daca imaginea este grayscale, o convertim la BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # Daca are canal alpha (PNG)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 3. Redimensionare la 224x224 cu pastrarea aspectului
        h, w = img.shape[:2]
        if h != w:
            size = max(h, w)
            pad_h = (size - h) // 2
            pad_w = (size - w) // 2
            img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
        resized = cv2.resize(img, (224, 224))

        # 4. Corectare umbre folosind CLAHE pe canalul de saturatie (S)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.createCLAHE(clipLimit=2.0).apply(hsv[:, :, 1])
        shadow_corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 5. Reducere zgomot cu filtru bilateral (pastreaza contururile)
        denoised = cv2.bilateralFilter(shadow_corrected, d=9, sigmaColor=75, sigmaSpace=75)

        # 6. Normalizare in intervalul [0, 1] si conversie la float32
        normalized = denoised.astype(np.float32) / 255.0

        # 7. Validare forma finala
        if normalized.shape != (224, 224, 3):
            raise ValueError(f"Forma finala invalida: {normalized.shape}")

        return normalized

    except Exception as e:
        print(f"Eroare la procesarea {image_path}: {str(e)}")
        return None


def process_folder(input_root, output_root):
    """
    Proceseaza toate imaginile dintr-un director organizat pe clase.
    Pentru fiecare imagine:
    - aplica preprocesarea
    - salveaza imaginea preprocesata in format .npy
    """
    os.makedirs(output_root, exist_ok=True)

    for species in tqdm(os.listdir(input_root), desc="Procesare clase"):
        input_path = os.path.join(input_root, species)
        output_path = os.path.join(output_root, species)

        if not os.path.isdir(input_path):
            continue

        os.makedirs(output_path, exist_ok=True)

        for img_name in os.listdir(input_path):
            input_file = os.path.join(input_path, img_name)
            output_file = os.path.join(output_path, os.path.splitext(img_name)[0] + '.npy')

            if not os.path.isfile(input_file):
                continue

            try:
                processed = preprocess_image(input_file)
                if processed is not None:
                    np.save(output_file, processed)
            except Exception as e:
                print(f"Eroare la fisierul {img_name}: {str(e)}")


if __name__ == "__main__":
    input_dir = "C:/Users/catal/OneDrive/Desktop/Medicinal plant dataset"
    output_dir = "C:/Users/catal/OneDrive/Desktop/cnn pipeline"

    print(f"Se incepe preprocesarea imaginilor din {input_dir} in {output_dir} ...")
    process_folder(input_dir, output_dir)
    print("Preprocesarea a fost finalizata.")
