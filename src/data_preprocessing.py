import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom preprocessing layer with Normalization
class CustomPreprocessNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomPreprocessNormalizationLayer, self).__init__()

    def call(self, inputs):
        # Normalize pixel values to [0, 1]
        normalized_images = inputs / 255.0

        return normalized_images
    
# Custom preprocessing layer with Standardization
class CustomPreprocessStandardizationLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomPreprocessStandardizationLayer, self).__init__()

    def call(self, inputs):
        # Standardize pixel values into mean=0 and std=1
        standardized_images = (inputs - tf.math.reduce_mean(inputs)) / tf.math.reduce_std(inputs)
        
        return standardized_images