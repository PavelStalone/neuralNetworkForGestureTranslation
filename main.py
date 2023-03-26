
import os
from Model import *

image_size = 224
batch_size = 32
dataset_path = "train_small"
classes = ["А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "Л", "М", "Н",
           "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Ь", "Ы", "Э", "Ю", "Я"]

data_augmentation = tf.keras.models.Sequential([
    layers.RandomFlip("horizontal", seed=321, input_shape=(image_size, image_size, 3)),
    layers.RandomRotation(0.01, seed=321, fill_mode="reflect"),
    layers.RandomContrast(0.3, seed=321),
    # layers.RandomBrightness(0.3, seed=321),
    # layers.RandomCrop(240, 240, seed=321),
])
cnn_model = tf.keras.models.Sequential([
    data_augmentation,

    layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
    layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
    layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
    layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
    layers.Conv2D(256, (3, 3), padding="same", activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(512, (3, 3), padding="same", activation='relu'),
    layers.Conv2D(512, (3, 3), padding="same", activation='relu'),
    layers.Conv2D(412, (3, 3), padding="same", activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dense(4096, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")
])

model = Model(cnn_model, image_size, batch_size, classes)
model.generate(name="sl_classification_1", data=dataset_path,
    epochs=10, optimise=False, show_history=True)

# model.load("sl_classification_1")

# model.refit("sl_classification_1", "sl_classification_1",
#     "archive/asl_alphabet_train/asl_alphabet_train",
#     epochs=20, optimise=False, show_history=True)

# for file in os.listdir("test"):
#     answer, score = model.test(f"test/{file}")
#     print(f"predict {answer} on {file} with {score} percents")

