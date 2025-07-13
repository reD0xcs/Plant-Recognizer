import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm
from scipy.ndimage import rotate

# === CONFIGURARE ===
FOLDER_FEATURES = "C:/Users/catal/OneDrive/Desktop/ml pipeline/hog_features"
MODEL_PATH = "C:/Users/catal/OneDrive/Desktop/ml pipeline/random_forest_model_augmented.joblib"
FOLOSESTE_AUGMENTARE = False  # ← Seteaza pe False pentru test fara augmentare

# Parametri imagine si HOG
DIMENSIUNE_IMAGINE = (128, 128)
PIXELS_PER_CELL = (16, 16)
CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
HIST_BINS = 16

def calcul_dimensiuni_hog():
    """
    Calculeaza dimensiunea vectorului HOG pentru imaginea data.

    Returneaza:
        int: Dimensiunea totala a descriptorului HOG.
    """
    n_cells_x = DIMENSIUNE_IMAGINE[0] // PIXELS_PER_CELL[0]
    n_cells_y = DIMENSIUNE_IMAGINE[1] // PIXELS_PER_CELL[1]
    n_blocks_x = (n_cells_x - CELLS_PER_BLOCK[0]) + 1
    n_blocks_y = (n_cells_y - CELLS_PER_BLOCK[1]) + 1
    return n_blocks_x * n_blocks_y * CELLS_PER_BLOCK[0] * CELLS_PER_BLOCK[1] * HOG_ORIENTATIONS

DIM_HOG = calcul_dimensiuni_hog()
DIM_HIST = 3 * HIST_BINS

def augment_vector(features):
    """
    Realizeaza augmentari ale vectorului de caracteristici HOG + histograme.

    Augmentarile aplicate sunt:
    - flip orizontal
    - flip vertical
    - rotiri 90, 180, 270 grade

    Args:
        features (np.array): Vectorul de caracteristici initial, concatenare HOG + histograme.

    Returneaza:
        list: Lista de vectori augmentati, inclusiv vectorul original.
    """
    hog_part = features[:DIM_HOG]
    hist_part = features[DIM_HOG:]

    n_cells_x = DIMENSIUNE_IMAGINE[0] // PIXELS_PER_CELL[0]
    n_cells_y = DIMENSIUNE_IMAGINE[1] // PIXELS_PER_CELL[1]
    n_blocks_x = (n_cells_x - CELLS_PER_BLOCK[0]) + 1
    n_blocks_y = (n_cells_y - CELLS_PER_BLOCK[1]) + 1

    try:
        hog_reshaped = hog_part.reshape(
            (n_blocks_y, n_blocks_x, CELLS_PER_BLOCK[0], CELLS_PER_BLOCK[1], HOG_ORIENTATIONS))
    except:
        print(f"Eroare reshape: {hog_part.shape}")
        raise

    augmented = [features]
    flipped_h = np.flip(hog_reshaped, axis=1)
    augmented.append(np.concatenate([flipped_h.ravel(), hist_part]))

    flipped_v = np.flip(hog_reshaped, axis=0)
    augmented.append(np.concatenate([flipped_v.ravel(), hist_part]))

    rotated_90 = rotate(hog_reshaped, angle=90, axes=(1, 0), reshape=False)
    augmented.append(np.concatenate([rotated_90.ravel(), hist_part]))

    rotated_180 = rotate(hog_reshaped, angle=180, axes=(1, 0), reshape=False)
    augmented.append(np.concatenate([rotated_180.ravel(), hist_part]))

    rotated_270 = rotate(hog_reshaped, angle=270, axes=(1, 0), reshape=False)
    augmented.append(np.concatenate([rotated_270.ravel(), hist_part]))

    return augmented

def incarca_date_din_folder(folder_hog, augment=True):
    """
    Incarca vectorii de caracteristici HOG + histograme din folderele de clase,
    aplicand optional augmentare.

    Args:
        folder_hog (str): Calea catre directorul cu foldere pentru fiecare clasa.
        augment (bool): Daca este True, aplica augmentarea vectorilor.

    Returneaza:
        tuple: (X, y, label_map)
            X (np.array): Array cu vectori de caracteristici.
            y (np.array): Array cu etichete numerice corespunzatoare claselor.
            label_map (dict): Mapare clasa (string) -> eticheta (int).
    """
    X = []
    y = []
    label_map = {}
    idx = 0

    clase = sorted(os.listdir(folder_hog))
    for clasa in tqdm(clase, desc="Incarca clase"):
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
    RF_PARAMS = {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': True,
        'n_jobs': -1,
        'random_state': 42
    }

    print(f"Incarc datele (augmentare={FOLOSESTE_AUGMENTARE})...")
    X, y, label_map = incarca_date_din_folder(FOLDER_FEATURES, augment=FOLOSESTE_AUGMENTARE)

    print(f"Set complet: {X.shape[0]} exemple, dim vector: {X.shape[1]}")
    print(f"Clase: {len(label_map)} - {label_map}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("\nAntrenez Random Forest...")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    print("\nEvaluare pe setul de test:")
    y_pred = rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Acuratețe test: {test_acc:.4f}")
    print("\nRaport clasificare:\n")
    print(classification_report(y_test, y_pred, target_names=list(label_map.keys())))

    y_train_pred = rf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Acuratețe antrenare: {train_acc:.4f}")

    print("\nCross-validation (5-fold)...")
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"Acuratețe CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    joblib.dump({"model": rf, "label_map": label_map}, MODEL_PATH)
    print(f"Model salvat la: {MODEL_PATH}")
