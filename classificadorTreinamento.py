from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import shutil

def preprocess_dataset(dataset_path):
    train_path = 'data/treinamento'
    val_path = 'data/validacao'

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

            train_class_dir = os.path.join(train_path, class_folder)
            val_class_dir = os.path.join(val_path, class_folder)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            for img in train_images:
                shutil.move(os.path.join(class_path, img), os.path.join(train_class_dir, img))

            for img in val_images:
                shutil.move(os.path.join(class_path, img), os.path.join(val_class_dir, img))

def train_model(model, train_generator, validation_generator, epochs=10):
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
    return history
