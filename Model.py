
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
layers = tf.keras.layers

class Model:
    def __init__(self, cnn_model, image_size, batch_size, classes):
        self.model = tf.keras.models.Sequential()
        self.cnn_model = cnn_model
        self.image_size = image_size
        self.batch_size = batch_size
        self.classes = classes

    def refit(self, name, load_name, data, epochs, optimise=False, show_history=False):
        train_ds, validate_ds = self.prepare_ds(data, optimise)
        self.cnn_model = tf.keras.models.load_model(load_name)
        history = self.cnn_model.fit(train_ds, validation_data=validate_ds, epochs=epochs)
        if show_history: self.show_history_plots(history, epochs)
        self.cnn_model.save(name)

    def generate(self, name, data, epochs, optimise=False, show_history=False):
        train_ds, validate_ds = self.prepare_ds(data, optimise)
        self.cnn_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        history = self.cnn_model.fit(train_ds, validation_data=validate_ds, epochs=epochs)
        if show_history: self.show_history_plots(history, epochs)
        self.cnn_model.save(name)

    def load(self, name):
        model = tf.keras.models.load_model(name)
        self.model = model

    def test(self, img):
        img = tf.keras.utils.load_img(img, target_size=(self.image_size, self.image_size))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array / self.image_size, 0)
        predictions = self.model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])
        answer = self.classes[np.argmax(score)]
        return answer

    def prepare_ds(self, data, optimise):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data, validation_split=0.2, subset="training", seed=321, batch_size=self.batch_size,
            image_size=(self.image_size, self.image_size))
        validate_ds = tf.keras.utils.image_dataset_from_directory(
            data, validation_split=0.2, subset="validation", seed=321, batch_size=self.batch_size,
            image_size=(self.image_size, self.image_size))

        if optimise:
            autotune = tf.data.AUTOTUNE
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
            validate_ds = validate_ds.cache().prefetch(buffer_size=autotune)

        train_ds = train_ds.map(lambda x, y: (x / self.image_size, y))
        validate_ds = validate_ds.map(lambda x, y: (x / self.image_size, y))
        return train_ds, validate_ds

    def show_history_plots(self, history, epochs):
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Train acc")
        plt.plot(epochs_range, val_acc, label="Val acc")
        plt.legend(loc="upper left")
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Train loss")
        plt.plot(epochs_range, val_loss, label="Val loss")
        plt.legend(loc="upper left")
        plt.show()

