from keras import Model
from keras.layers import Conv2D, Dropout, MaxPool2D, Dense, Input, Flatten
from keras.optimizers import RMSprop

from preprocess_layer import PreProcessLayer
from keras.regularizers import l2


class NoisyClassifier:

    def __init__(self, input_shape=(28, 28, 1), n_classes = 10, kernel_size=5, pool_size=2, base_filter_count=32, dropout_rate=0.5,
                 fc_base_count = 128, learning_rate=0.001, l2_weight=0.001):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.base_filter_count = base_filter_count
        self.dropout_rate = dropout_rate
        self.fc_base_count = fc_base_count
        self.learning_rate = learning_rate
        self.l2_weight = l2_weight

    def fetch_model(self):
        inputs = Input(shape=self.input_shape)

        x = PreProcessLayer()(inputs)
        x = Conv2D(filters=self.base_filter_count, kernel_size=self.kernel_size, padding="same", activation="relu")(x)
        x = Conv2D(filters=self.base_filter_count, kernel_size=self.kernel_size, padding="same", activation="relu")(x)
        x = MaxPool2D(pool_size=(self.pool_size, self.pool_size), padding="same")(x)

        x = Conv2D(filters=2*self.base_filter_count, kernel_size=self.kernel_size, padding="same", activation="relu")(x)
        x = Conv2D(filters=2*self.base_filter_count, kernel_size=self.kernel_size, padding="same", activation="relu")(x)
        x = MaxPool2D(pool_size=(self.pool_size, self.pool_size))(x)

        x = Conv2D(filters=4*self.base_filter_count, kernel_size=self.kernel_size, padding="same", activation="relu")(x)
        x = MaxPool2D(pool_size=(self.pool_size, self.pool_size))

        x = Flatten()(x)

        x = Dense(2*self.fc_base_count, activation="relu", kernel_regularizer=l2(self.l2_weight))(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(self.fc_base_count, activation="relu", kernel_regularizer=l2(self.l2_weight))(x)
        x = Dropout(self.dropout_rate)(x)

        outputs = Dense(self.n_classes, activation="softmax")(x)
        model = Model(inputs, outputs)

        return model

    def compile_model(self, model):
        optimizer = RMSprop(learning_rate = self.learning_rate)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def get_compiled_model(self):
        return self.compile_model(model=self.fetch_model())

