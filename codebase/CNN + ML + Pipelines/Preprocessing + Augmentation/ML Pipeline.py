import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import hog

# === SETARI GENERALE ===
DIMENSIUNE_IMAGINE = (128, 128)          # dimensiunea standard la care este redimensionata fiecare imagine
PIXELS_PER_CELL = (16, 16)               # dimensiunea unei celule pentru HOG
CELLS_PER_BLOCK = (2, 2)                 # dimensiunea unui bloc pentru HOG
HOG_ORIENTATIONS = 9                     # numarul de directii de orientare pentru histograma gradientilor
HIST_BINS = 16                           # numarul de bare (bins) pentru histograma HSV (pentru fiecare canal)


def extrage_caracteristici(img):
    """
    Primeste o imagine originala si returneaza:
    - imaginea procesata (grayscale + Gaussian blur + resize)
    - un vector de caracteristici HOG + histograma HSV normalizata
    """
    # 1. Resize + grayscale
    img = cv2.resize(img, DIMENSIUNE_IMAGINE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur (reducere zgomot si detalii mici)
    gray = cv2.GaussianBlur(gray, (5, 5), sigmaX=1)

    # 3. Extractie HOG din imaginea grayscale blurata
    features_hog = hog(gray,
                       orientations=HOG_ORIENTATIONS,
                       pixels_per_cell=PIXELS_PER_CELL,
                       cells_per_block=CELLS_PER_BLOCK,
                       block_norm='L2-Hys',
                       visualize=False)

    # 4. Convertire in HSV si extragere histograma pentru fiecare canal
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [HIST_BINS], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [HIST_BINS], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [HIST_BINS], [0, 256]).flatten()

    # 5. Normalizare histograma si concatenare
    hist_features = np.concatenate([h_hist, s_hist, v_hist])
    hist_features = hist_features / np.linalg.norm(hist_features)  # normalizare L2

    # 6. Concatenare totala a vectorilor
    vector_final = np.concatenate([features_hog, hist_features])

    return gray, vector_final


def proceseaza_dataset(input_folder, output_folder):
    """
    Parcurge toate imaginile dintr-un folder structurat pe clase si:
    - salveaza imaginile procesate ca .npy
    - salveaza vectorii de caracteristici HOG+HSV ca .npy
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Folderul sursa nu exista: {input_folder}")

    # Creeaza structura de iesire
    img_out = os.path.join(output_folder, "imagini_finale")
    feat_out = os.path.join(output_folder, "hog_features")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(feat_out, exist_ok=True)

    # Lista tuturor claselor (subfoldere)
    clase = sorted([c for c in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, c))])

    for clasa in tqdm(clase, desc="Clase"):
        cale_clasa = os.path.join(input_folder, clasa)
        imgs = sorted(os.listdir(cale_clasa))

        # Creeaza folderele de iesire pentru clasa curenta
        cale_out_img = os.path.join(img_out, clasa)
        cale_out_feat = os.path.join(feat_out, clasa)
        os.makedirs(cale_out_img, exist_ok=True)
        os.makedirs(cale_out_feat, exist_ok=True)

        # Procesare imagini din clasa
        for img_name in tqdm(imgs, desc=f"{clasa}", leave=False):
            cale_img = os.path.join(cale_clasa, img_name)
            if not os.path.isfile(cale_img):
                continue

            try:
                img = cv2.imread(cale_img)
                if img is None:
                    print(f"Imagine invalida: {cale_img}")
                    continue

                # Prelucrare imagine + extractie caracteristici
                img_proc, features = extrage_caracteristici(img)
                base = os.path.splitext(img_name)[0]

                # Salvare imagine procesata
                np.save(os.path.join(cale_out_img, f"{base}.npy"), img_proc)

                # Salvare vector de caracteristici
                np.save(os.path.join(cale_out_feat, f"{base}_feat.npy"), features)

            except Exception as e:
                print(f"Eroare la {img_name}: {e}")


if __name__ == "__main__":
    # Setare cai pentru input si output
    FOLDER_INTRARE = "C:/Users/catal/OneDrive/Desktop/Medicinal plant dataset"
    FOLDER_IESIRE = "C:/Users/catal/OneDrive/Desktop/ml pipeline"

    print("Incepem preprocesarea...")
    proceseaza_dataset(FOLDER_INTRARE, FOLDER_IESIRE)
    print("Gata! Datele procesate sunt in:")
    print(f"- {os.path.join(FOLDER_IESIRE, 'imagini_finale')}")
    print(f"- {os.path.join(FOLDER_IESIRE, 'hog_features')}")
