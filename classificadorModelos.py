from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import VGG16, ResNet50

def build_model1():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model2():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_transfer_model1():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    model_transfer = Sequential()
    model_transfer.add(base_model)
    model_transfer.add(Flatten())
    model_transfer.add(Dense(256, activation='relu'))
    model_transfer.add(Dense(4, activation='softmax'))
    model_transfer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model_transfer

def build_transfer_model2():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    model_transfer = Sequential()
    model_transfer.add(base_model)
    model_transfer.add(Flatten())
    model_transfer.add(Dense(256, activation='relu'))
    model_transfer.add(Dense(4, activation='softmax'))
    model_transfer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model_transfer
