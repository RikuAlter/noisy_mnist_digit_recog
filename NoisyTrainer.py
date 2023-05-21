from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit


class MaxAccuracy(Callback):
    def __init__(self):
        super(MaxAccuracy, self).__init__()
        self.max_train_accuracy = 0.0
        self.max_val_accuracy = 0.0

    def on_epoch_end(self, epoch, logs=None):
        train_accuracy = logs['accuracy']
        val_accuracy = logs['val_accuracy']
        if train_accuracy > self.max_train_accuracy:
            self.max_train_accuracy = train_accuracy
        if val_accuracy > self.max_val_accuracy:
            self.max_val_accuracy = val_accuracy
        print(
            f" Max Train Accuracy: {self.max_train_accuracy:.4f}, Max Validation Accuracy: {self.max_val_accuracy:.4f}")


def train_model(model, x, y, n_splits=10, batch_size=64, test_size=0.2, random_state=47, n_epochs=100,
                min_learning_rate=0.0000001, lr_decay_factor=0.8):
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    data_generator = ImageDataGenerator(rotation_range=10,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        shear_range=0.1,
                                        zoom_range=0.1,
                                        fill_mode="nearest")

    aug_set_id = 0
    histories = []
    max_accuracy = MaxAccuracy()
    for train_ids, test_ids in splitter.split(x, y):
        checkpoint = ModelCheckpoint("model_" + str(aug_set_id) + ".h5", monitor="val_accuracy",
                                     save_best_only=True, model="max")
        aug_set_id = aug_set_id + 1

        x_train, x_test = x[train_ids], x[test_ids]
        y_train, y_test = y[train_ids], y[test_ids]
        histories.append(model.fit(data_generator.flow(x_train, y_train, batch_size=batch_size),
                                   epochs=n_epochs, validation_data=(x_test, y_test),
                                   callbacks=[ReduceLROnPlateau(monitor="val_loss", factor=lr_decay_factor,
                                                                patience=5, min_lr=min_learning_rate), checkpoint,
                                              max_accuracy]))

        return histories
