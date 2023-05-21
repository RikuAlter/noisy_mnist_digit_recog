import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Layer


class PreProcessLayer(Layer):
    def __init__(self, blur_kernel=3, blur_sigma=0.7,
                 clip_limit=0.4, clip_kernel=3, input_shape=(28, 28, 1), **kwargs):
        super(PreProcessLayer, self).__init__(input_shape=input_shape, **kwargs)
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.clip_limit = clip_limit
        self.clip_kernel = clip_kernel

    def build(self, input_shape):
        super(PreProcessLayer, self).build(input_shape)

    #    def call(self, inputs):
    #        for img in inputs:
    #            normalized = tf.cast((img - tf.reduce_min(img))/tf.reduce_max(img), tf.float32)
    #            blurred = tf.image.convert_image_dtype(tfa.image.gaussian_filter2d(normalized, [self.blur_kernel, self.blur_kernel], self.blur_sigma, "REFLECT"), tf.float32)
    #            clahe = tf.image.adjust_contrast(tf.image.convert_image_dtype(blurred, tf.uint8), self.clip_limit)
    #            img = tf.image.convert_image_dtype(clahe, tf.float32)

    #        return inputs

    def call(self, inputs):
        normalized = tf.cast(
            (inputs - tf.reduce_min(inputs, axis=[1, 2, 3], keepdims=True)) / tf.reduce_max(inputs, axis=[1, 2, 3],
                                                                                            keepdims=True), tf.float32)
        blurred = tf.image.convert_image_dtype(
            tfa.image.gaussian_filter2d(normalized, [self.blur_kernel, self.blur_kernel], self.blur_sigma, "REFLECT"),
            tf.float32)
        # clahe = tf.image.adjust_contrast(tf.image.convert_image_dtype(blurred, tf.uint8), self.clip_limit)
        outputs = tf.image.convert_image_dtype(blurred, tf.float32)

        return outputs

    def get_config(self):
        config = super(PreProcessLayer, self).get_config()
        config.update({
            'blur_kernel': self.blur_kernel,
            'blur_sigma': self.blur_sigma,
            'clip_limit': self.clip_limit,
            'clip_kernel': self.clip_kernel
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)