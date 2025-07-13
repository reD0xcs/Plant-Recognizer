import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model
# noinspection PyUnresolvedReferences
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import glob

# Dezactiveaza logarile inutile TensorFlow pentru a nu incarca consola
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Setari generale
IMG_SIZE = (224, 224)  # Dimensiunea imaginilor de intrare pentru model
BATCH_SIZE = 32  # Dimensiunea batch-ului la antrenare
NUM_CLASSES = 40  # Numarul total de clase din dataset
EPOCHS_HEAD = 10  # Epoci pentru antrenarea numai a capului retelei
EPOCHS_FULL = 50  # Epoci pentru antrenarea completa (fine tuning)
DATASET_DIR = 'C:/Users/catal/OneDrive/Desktop/augDATA/CNNdata'  # Directorul cu datele .npy


# Clasa NPYDataGenerator este un generator personalizat pentru fisiere .npy
# Incarca batch-uri de imagini si etichete pentru antrenarea modelului Keras
class NPYDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size, num_classes, shuffle=True):
        """
        Initializare generator

        :param file_paths: lista cu cai catre fisiere .npy
        :param labels: lista cu etichetele corespunzatoare
        :param batch_size: dimensiunea batch-ului
        :param num_classes: numarul total de clase (pentru one-hot encoding)
        :param shuffle: daca se amesteca datele dupa fiecare epoca
        """
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Returneaza numarul total de batch-uri per epoca
        """
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        """
        Returneaza batch-ul cu index-ul specificat

        :param index: indexul batch-ului
        :return: tuple (batch_imagini, batch_etichete)
        """
        # Selecteaza cai si etichete pentru batch-ul curent
        batch_paths = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        for path in batch_paths:
            # Incarca imaginea din fisierul .npy
            img = np.load(path).astype(np.float32)

            # Modelul MobileNetV2 asteapta valori in intervalul [-1, 1]
            if img.max() <= 1.0:  # Daca valorile sunt in [0,1]
                img = (img * 2) - 1  # Scaleaza la [-1, 1]

            # Asigura dimensiunea corecta a imaginii
            if img.shape != (224, 224, 3):
                img = np.resize(img, (224, 224, 3))

            batch_images.append(img)

        batch_images = np.array(batch_images)
        # One-hot encode etichetele
        batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)
        return batch_images, batch_labels

    def on_epoch_end(self):
        """
        Metoda apelata la sfarsitul fiecarei epoci
        Daca shuffle este True, amesteca datele
        """
        if self.shuffle:
            indices = np.arange(len(self.file_paths))
            np.random.shuffle(indices)
            self.file_paths = [self.file_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]


# Incarca toate fisierele .npy si etichetele din directoarele clasei
file_paths = []
labels = []
class_indices = {}  # Dictionar nume_clasa -> index numeric

# Parcurge fiecare clasa (folder)
for class_idx, class_name in enumerate(sorted(os.listdir(DATASET_DIR))):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    class_indices[class_name] = class_idx
    # Adauga toate fisierele .npy din clasa respectiva
    for npy_file in glob.glob(os.path.join(class_dir, "*.npy")):
        file_paths.append(npy_file)
        labels.append(class_idx)

# Imparte datele in seturi de antrenament si validare
train_files, val_files, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Creeaza generatori pentru datele de antrenament si validare
train_gen = NPYDataGenerator(train_files, train_labels, BATCH_SIZE, NUM_CLASSES, shuffle=True)
val_gen = NPYDataGenerator(val_files, val_labels, BATCH_SIZE, NUM_CLASSES, shuffle=False)

# Verificare rapida: afiseaza forma si intervalul pixelilor pentru un batch
sample_images, _ = train_gen[0]
print(f"\nData verification:")
print(f"Input shape: {sample_images.shape}")
print(f"Pixel range: Min={sample_images.min():.2f}, Max={sample_images.max():.2f}")

# Construirea modelului MobileNetV2 pretrained pe ImageNet fara stratul de output final
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,  # exclude stratul complet conectat final
    weights='imagenet',  # foloseste greutatile pretrained pe ImageNet
    pooling=None  # nu face pooling global aici, il facem manual ulterior
)

# Ingheata toate straturile pentru moment, antrenam doar capul retelei
base_model.trainable = False

# Adaugam noi straturi pentru clasificarea noastra
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pooling global pentru a reduce dimensiunea outputului
x = Dropout(0.5)(x)  # Dropout pentru regularizare si reducerea overfittingului
predictions = Dense(NUM_CLASSES, activation='softmax')(
    x)  # Strat final cu activare softmax pentru clasificare multi-clasa

# Construim modelul complet
model = Model(inputs=base_model.input, outputs=predictions)

# Compilam modelul pentru antrenarea initiala a capului retelei
model.compile(
    optimizer=Adam(learning_rate=1e-3),  # Optimizer Adam cu rata de invatare mai mare pentru stratul final
    loss='categorical_crossentropy',  # Functia de pierdere pentru clasificare multi-clasa
    metrics=['accuracy']  # Metri pentru evaluare
)

print("\nTraining head only...")
# Antrenam doar capul modelului pe epocile setate
history_head = model.fit(
    train_gen,
    epochs=EPOCHS_HEAD,
    validation_data=val_gen,
    verbose=1
)

# Pornim faza de fine tuning: activam ultimele 30 de straturi din base_model
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Compilam modelul cu o rata de invatare mult mai mica pentru fine tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Rata de invatare mica pentru a nu "strica" greutatile pretrained
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback-uri pentru antrenament
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True),
    # Opreste daca nu se imbunatateste dupa 10 epoci
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)  # Reduce rata de invatare daca pierderea nu scade
]

print("\nFine-tuning model...")
# Antrenam modelul cu stratul complet de bază deblocat
history_full = model.fit(
    train_gen,
    epochs=EPOCHS_FULL,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# Salvam modelul antrenat pe disc
model.save("plant_mobilenetv2_optimized.keras")
print("\nModel saved successfully!")
