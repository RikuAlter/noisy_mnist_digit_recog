from sklearn.model_selection import StratifiedShuffleSplit as sssplit
from keras.preprocessing.image import ImageDataGenerator as Imgen
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau


def prepare_model_for_training():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(Conv2D(filters=192, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, image_list, label_list, n_splits, test_size, random_state, n_epochs):
    splitter = sssplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    history_list = []
    data_generator = Imgen(rotation_range=10,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           shear_range=0.1,
                           zoom_range=0.1,
                           fill_mode='nearest')
    for train_id_list, test_id_list in splitter.split(image_list, label_list):
        print(len(train_id_list), len(test_id_list))
        X_train, X_test = image_list[train_id_list], image_list[test_id_list]
        y_train, y_test = label_list[train_id_list], label_list[test_id_list]

        history = model.fit(data_generator.flow(X_train, y_train, batch_size=128),
                            epochs=n_epochs, validation_data=(X_test, y_test),
                            callbacks=[ReduceLROnPlateau(monitor="loss", factor=0.8,
                                                         patience=5, min_lr=0.0000001)])
        history_list.append(history)

    return history_list