import tensorflow as tf
import cv2


class GaussianBlur(Layer):
    def __init__(self, kernel_size=3, sigma=0, **kwargs):
        super(GaussianBlur, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def build(self, input_shape):
        super(GaussianBlur, self).build(input_shape)

    def call(self, inputs):
        def gaussian_blur(img):
            blurred = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)
            return blurred.astype('float32')

        return tf.py_function(gaussian_blur, [inputs], tf.float32)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    GaussianBlur(kernel_size=3, sigma=0, input_shape=(None, None, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
