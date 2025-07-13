import numpy as np
import os
from tensorflow.keras import layers, models
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from skimage.transform import resize
# noinspection PyUnresolvedReferences
from tensorflow.keras.utils import Sequence
import gc

# Dezactiveaza mesajele de log de la TensorFlow (info/warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Directorul de baza unde sunt imaginile .npy organizate pe clase
base_dir = 'C:/Users/catal/OneDrive/Desktop/augDATA/CNNdata'

# Clasa care incarca imaginile in mod eficient in memorie, folosind keras.utils.Sequence
class OptimizedPlantDataGenerator(Sequence):
    """Generator eficient pentru incarcare batch pe rand, folosind fisiere .npy"""

    def __init__(self, data_dir, batch_size=32, target_size=(128, 128), shuffle=True, subset=None):
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle

        # Gaseste numele claselor in functie de numele folderelor
        self.class_dirs = sorted([d for d in os.listdir(data_dir)
                                  if os.path.isdir(os.path.join(data_dir, d))])
        self.class_indices = {name: idx for idx, name in enumerate(self.class_dirs)}

        # Creeaza o lista de fisiere: (cale_fisier, index_clasa)
        self.file_index = []
        for class_name, class_idx in self.class_indices.items():
            class_dir = os.path.join(data_dir, class_name)
            self.file_index.extend([
                (os.path.join(class_dir, f), class_idx)
                for f in os.listdir(class_dir) if f.endswith('.npy')
            ])

        # Imparte setul in train si validare, daca subset este specificat
        if subset:
            _, self.file_index = train_test_split(
                self.file_index,
                test_size=0.2,
                random_state=42,
                stratify=[x[1] for x in self.file_index]
            )
            if subset == 'train':
                self.file_index, _ = train_test_split(
                    self.file_index,
                    test_size=0.2,
                    random_state=42,
                    stratify=[x[1] for x in self.file_index]
                )

        # Genereaza ordine random a indicilor
        self.indices = np.arange(len(self.file_index))
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Prealoca memorie pentru un batch
        self.batch_x = np.zeros((batch_size, *target_size, 3), dtype=np.float32)
        self.batch_y = np.zeros((batch_size, len(self.class_dirs)), dtype=np.float32)

    def __len__(self):
        # Returneaza numarul total de batch-uri per epoca
        return int(np.ceil(len(self.file_index) / self.batch_size))

    def __getitem__(self, idx):
        # Returneaza batch-ul idx
        batch_indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]

        for i, index in enumerate(batch_indices):
            if index >= len(self.file_index):
                # Pentru batch-urile incomplete de la final
                return self.batch_x[:i], self.batch_y[:i]

            path, class_idx = self.file_index[index]

            try:
                # Incarca imaginea si redimensioneaz-o la dimensiunea target
                img = np.load(path)
                if img.shape != (*self.target_size, 3):
                    img = resize(img, self.target_size)

                self.batch_x[i] = img
                self.batch_y[i] = np.eye(len(self.class_dirs))[class_idx]
            except Exception as e:
                print(f"Eroare la incarcarea {path}: {str(e)}")
                self.batch_x[i] = np.zeros((*self.target_size, 3))

        return self.batch_x, self.batch_y

    def on_epoch_end(self):
        # Reamesteca indicii si colecteaza memoria
        if self.shuffle:
            np.random.shuffle(self.indices)
        gc.collect()

# Creeaza generatorul pentru datele de antrenare
train_gen = OptimizedPlantDataGenerator(
    base_dir,
    batch_size=24,
    target_size=(128, 128),
    shuffle=True,
    subset='train'
)

# Creeaza generatorul pentru datele de validare
val_gen = OptimizedPlantDataGenerator(
    base_dir,
    batch_size=24,
    target_size=(128, 128),
    shuffle=False,
    subset='val'
)

# Salveaza mappingul dintre clase si indecsi
with open('class_labels.json', 'w') as f:
    json.dump(train_gen.class_indices, f)

# Arhitectura CNN simplificata pentru clasificare de plante medicinale
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(train_gen.class_dirs), activation='softmax')
])

# Compilarea modelului
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback-uri pentru early stopping si salvare model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]

# Antrenare model cu colectare de memorie dupa fiecare epoca
try:
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )
finally:
    gc.collect()

# Salveaza modelul final
model.save('plant_classifier_model.keras')

# Plot pentru acuratete si loss in timpul antrenarii
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
