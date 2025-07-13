import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm
from scipy.ndimage import rotate

# Parametrii pentru extragerea caracteristicilor (HOG + histograme HSV)
DIMENSIUNE_IMAGINE = (128, 128)
PIXELS_PER_CELL = (16, 16)
CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
HIST_BINS = 16

# Variabila pentru activare augmentare - default False
USE_AUGMENTATION = False


def calcul_dimensiuni_hog():
    """
    Calculeaza dimensiunea vectorului HOG pe baza parametrilor de configurare.
    Returneaza un intreg cu numarul total de caracteristici HOG.
    """
    n_cells_x = DIMENSIUNE_IMAGINE[0] // PIXELS_PER_CELL[0]
    n_cells_y = DIMENSIUNE_IMAGINE[1] // PIXELS_PER_CELL[1]
    n_blocks_x = (n_cells_x - CELLS_PER_BLOCK[0]) + 1
    n_blocks_y = (n_cells_y - CELLS_PER_BLOCK[1]) + 1
    return n_blocks_x * n_blocks_y * CELLS_PER_BLOCK[0] * CELLS_PER_BLOCK[1] * HOG_ORIENTATIONS


DIM_HOG = calcul_dimensiuni_hog()
DIM_HIST = 3 * HIST_BINS  # 3 canale HSV, fiecare cu HIST_BINS numar de bins


def augment_vector(features):
    """
    Aplica augmentari geometrice pe vectorul de caracteristici (HOG + histograme).
    Augmentarile includ flip orizontal, flip vertical si rotiri la 90, 180, 270 grade.
    Returneaza lista de vectori augmentati, incluzand si vectorul original.
    """
    # Separare vector HOG si histograme
    hog_part = features[:DIM_HOG]
    hist_part = features[DIM_HOG:]

    # Reshape vector HOG la forma 5D pentru a putea face transformari
    n_cells_x = DIMENSIUNE_IMAGINE[0] // PIXELS_PER_CELL[0]
    n_cells_y = DIMENSIUNE_IMAGINE[1] // PIXELS_PER_CELL[1]
    n_blocks_x = (n_cells_x - CELLS_PER_BLOCK[0]) + 1
    n_blocks_y = (n_cells_y - CELLS_PER_BLOCK[1]) + 1

    hog_reshaped = hog_part.reshape((n_blocks_y, n_blocks_x, CELLS_PER_BLOCK[0], CELLS_PER_BLOCK[1], HOG_ORIENTATIONS))

    augmented = [features]  # lista cu vectorul original

    # Flip orizontal pe axa 1 (axa coloanei)
    flipped_h = np.flip(hog_reshaped, axis=1)
    augmented.append(np.concatenate([flipped_h.ravel(), hist_part]))

    # Flip vertical pe axa 0 (axa randului)
    flipped_v = np.flip(hog_reshaped, axis=0)
    augmented.append(np.concatenate([flipped_v.ravel(), hist_part]))

    # Rotire 90 grade (in planul axelor 1 si 0)
    rotated_90 = rotate(hog_reshaped, angle=90, axes=(1, 0), reshape=False)
    augmented.append(np.concatenate([rotated_90.ravel(), hist_part]))

    # Rotire 180 grade
    rotated_180 = rotate(hog_reshaped, angle=180, axes=(1, 0), reshape=False)
    augmented.append(np.concatenate([rotated_180.ravel(), hist_part]))

    # Rotire 270 grade
    rotated_270 = rotate(hog_reshaped, angle=270, axes=(1, 0), reshape=False)
    augmented.append(np.concatenate([rotated_270.ravel(), hist_part]))

    return augmented


def incarca_date_din_folder(folder_hog, augment=USE_AUGMENTATION):
    """
    Incarca vectorii de caracteristici si etichetele din folderul principal.
    Pentru fiecare clasa/specie se citesc fisierele .npy din subfolderul corespunzator.
    Daca augment este True, se aplica augmentare vectorilor.
    Returneaza:
        - X: matricea cu vectori de caracteristici
        - y: vectorul cu etichete (index clasa)
        - label_map: dictionar clasa -> index
    """
    X = []
    y = []
    label_map = {}
    idx = 0

    clase = sorted(os.listdir(folder_hog))
    for clasa in tqdm(clase, desc="Incarcare clase"):
        cale_clasa = os.path.join(folder_hog, clasa)
        if not os.path.isdir(cale_clasa):
            continue

        if clasa not in label_map:
            label_map[clasa] = idx
            idx += 1

        fisiere = [f for f in os.listdir(cale_clasa) if f.endswith('.npy')]
        for f in fisiere:
            try:
                vector = np.load(os.path.join(cale_clasa, f))
                if augment:
                    vectori_augmentati = augment_vector(vector)
                    for vec in vectori_augmentati:
                        X.append(vec)
                        y.append(label_map[clasa])
                else:
                    X.append(vector)
                    y.append(label_map[clasa])
            except Exception as e:
                print(f"Eroare la fisierul {f}: {e}")

    return np.array(X), np.array(y), label_map


if __name__ == "__main__":
    # Setari pentru date si model
    FOLDER_FEATURES = "C:/Users/catal/OneDrive/Desktop/ml pipeline/hog_features"
    MODEL_PATH = "C:/Users/catal/OneDrive/Desktop/ml pipeline/svm_model_augmented.joblib"

    print("Incarc datele...")
    X, y, label_map = incarca_date_din_folder(FOLDER_FEATURES)

    print(f"Set complet: {X.shape[0]} exemple, dim vector: {X.shape[1]}")
    print(f"Clase: {len(label_map)} - {label_map}")

    # Impartire date in train si test (stratificat)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Antrenare SVM cu kernel RBF
    print("Antrenez SVM...")
    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    clf.fit(X_train, y_train)

    # Evaluare model pe setul de test
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acuratete test: {acc:.4f}")
    print("\nRaport clasificare:\n")
    print(classification_report(y_test, y_pred))

    # Salvare model si label map
    joblib.dump({"model": clf, "label_map": label_map}, MODEL_PATH)
    print(f"Model salvat la: {MODEL_PATH}")
