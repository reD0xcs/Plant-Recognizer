# Medicinal Plant Classifier

Acest proiect are ca scop identificarea plantelor medicinale folosind modele de învățare automată și o aplicație Android care utilizează TensorFlow Lite.

## Structura și datele

### Dataset
- Baza de date folosită: [Indian Medicinal Leaves Dataset (Kaggle)](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset)
- Din arhiva descărcată, trebuie folosit doar folderul `medicinal plant dataset`, care conține imagini pentru 40 de specii de plante.

## Preprocesare și antrenare

### 1. Preprocesare și augmentare
Rulează, în ordine, următoarele scripturi:
- `cnn pipeline.py`
- `ml pipeline.py`
- `augmentare.py`

Acestea se ocupă de:
- Preprocesarea imaginilor
- Aplicarea augmentărilor (rotații, zgomot, etc.)
- Salvarea datelor procesate pentru antrenare

### 2. Antrenarea modelelor
Poți antrena oricare dintre următoarele modele disponibile:
- `cnn1`
- `cnn2`
- `mobilenetv2`
- `svm`
- `rf` (Random Forest)

Pentru aplicația mobilă, este necesar să antrenezi modelul `mobilenetv2`.

## Integrarea în aplicația Android

### 1. Conversia modelului
După antrenarea modelului `mobilenetv2`, convertește-l în format TensorFlow Lite (`.tflite`).

### 2. Salvarea modelului
Mută fișierul `.tflite` în directorul `assets` al proiectului Android Studio sub numele: plant_model.tflite


### 3. Configurarea datelor despre plante
- În folderul `assets/plants` trebuie să existe câte o imagine pentru fiecare specie de plantă.
- Fiecare imagine trebuie legată corect în fișierul `medicinal_plants.json` (aflat tot în `assets`), care conține detalii și linkuri către imagini și informații despre plante.

### 4. Versiunea Android
Proiectul a fost dezvoltat și testat folosind **Android Meerkat**.

## Cerințe

Asigură-te că ai instalate următoarele:
- Python 3.8+
- TensorFlow
- scikit-learn
- numpy, pandas, matplotlib 



