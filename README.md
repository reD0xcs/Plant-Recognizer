# Plant-Recognizer

This project aims to identify medicinal plants using machine learning models and an Android application that utilizes TensorFlow Lite.

## Structure and Data

### Dataset
- Database used: [Indian Medicinal Leaves Dataset (Kaggle)](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset)
- From the downloaded archive, only the `medicinal plant dataset` folder should be used, which contains images for 40 plant species.

## Preprocessing and Training

### 1. Preprocessing and Augmentation
Run the following scripts in order:
- `cnn pipeline.py`
- `ml pipeline.py`
- `augmentare.py`

These handle:
- Image preprocessing
- Applying augmentations (rotations, noise, etc.)
- Saving processed data for training

### 2. Model Training
You can train any of the following available models:
- `cnn1`
- `cnn2`
- `mobilenetv2`
- `svm`
- `rf` (Random Forest)

For the mobile application, it is required to train the `mobilenetv2` model.

## Integration into the Android Application

### 1. Model Conversion
After training the `mobilenetv2` model, convert it to TensorFlow Lite format (`.tflite`).

### 2. Saving the Model
Move the `.tflite` file into the `assets` directory of the Android Studio project under the name: `plant_model.tflite`.

### 3. Configuring Plant Data
- In the `assets/plants` folder, there should be one image for each plant species.
- Each image must be correctly linked in the `medicinal_plants.json` file (also located in `assets`), which contains details and links to images and information about the plants.

### 4. Android Version
The project was developed and tested using **Android Meerkat**.

## Requirements

Make sure you have the following installed:
- Python 3.8+
- TensorFlow
- scikit-learn
- numpy, pandas, matplotlib
