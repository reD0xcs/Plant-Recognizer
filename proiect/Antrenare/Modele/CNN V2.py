import numpy as np
import os
from tensorflow.keras import layers, models
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# noinspection PyUnresolvedReferences
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split

# Dezactiveaza mesajele info si warning de la TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Directorul cu datele .npy augmentate
base_dir = 'C:/Users/catal/OneDrive/Desktop/augDATA/CNNdata'


class PlantDataGenerator(Sequence):
    """
    Generator custom de date care incarca datele din fisiere .npy pe batch-uri.
    Previne suprasarcina de memorie incarcand datele la cerere.
    """

    def __init__(self, data_dir, batch_size=32, shuffle=True, subset=None):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Lista cu directoarele clasei, sortata alfabetic
        self.class_dirs = sorted([d for d in os.listdir(data_dir)
                                  if os.path.isdir(os.path.join(data_dir, d))])
        # Dictonar cu numele clasei -> index numeric
        self.class_indices = {name: idx for idx, name in enumerate(self.class_dirs)}

        # Colecteaza toate fisierele .npy si etichetele lor
        self.file_paths = []
        self.labels = []

        for class_name, class_idx in self.class_indices.items():
            class_dir = os.path.join(data_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.npy'):
                    self.file_paths.append(os.path.join(class_dir, file))
                    self.labels.append(class_idx)

        # Daca se specifica subset (train sau val), imparte datele pentru validare
        if subset:
            _, self.file_paths, _, self.labels = train_test_split(
                self.file_paths, self.labels,
                test_size=0.2,
                random_state=42,
                stratify=self.labels
            )
            if subset == 'train':
                self.file_paths, _, self.labels, _ = train_test_split(
                    self.file_paths, self.labels,
                    test_size=0.2,
                    random_state=42,
                    stratify=self.labels
                )

        # Indici pentru shuffle
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """
        Returneaza numarul de batch-uri pe epoca
        """
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        """
        Returneaza batch-ul la indexul idx.
        Incarca imaginile din fisierele .npy si etichetele corespunzatoare.
        """
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []

        for i in batch_indices:
            try:
                img = np.load(self.file_paths[i])
                # Verifica daca imaginea are forma corecta
                if img.shape == (224, 224, 3):
                    batch_x.append(img)
                    batch_y.append(self.labels[i])
                else:
                    print(f"Skipping {self.file_paths[i]} - invalid shape {img.shape}")
            except Exception as e:
                print(f"Error loading {self.file_paths[i]}: {str(e)}")
                continue

        # Converteste lista in numpy array si face one-hot encoding pentru etichete
        return np.array(batch_x), np.eye(len(self.class_dirs))[batch_y]

    def on_epoch_end(self):
        """
        Shuffleaza indicii dupa fiecare epoca daca shuffle este activat
        """
        if self.shuffle:
            np.random.shuffle(self.indices)


# Initializeaza generatoarele pentru antrenare si validare
train_gen = PlantDataGenerator(base_dir, batch_size=32, shuffle=True, subset='train')
val_gen = PlantDataGenerator(base_dir, batch_size=32, shuffle=False, subset='val')

# Salveaza dictionarul clasei in fisier JSON pentru referinta
with open('class_labels.json', 'w') as f:
    json.dump(train_gen.class_indices, f)

# Definitia modelului CNN pentru clasificare plante
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(len(train_gen.class_dirs), activation='softmax')
])

# Compilarea modelului cu optimizer Adam si functia de pierdere categorical_crossentropy
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback-uri pentru antrenare:
# EarlyStopping opreste antrenamentul daca validarea nu se imbunatateste
# ModelCheckpoint salveaza modelul cu cea mai buna acuratete pe setul de validare
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]

# Antrenarea modelului folosind generatoarele definite, pe 50 epoci
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=callbacks
)

# Salvarea modelului final antrenat
model.save('plant_classifier_model.keras')

# Plotarea istoricului antrenamentului (acuratete si pierdere)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation', color='red')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
plt.show()
