import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from data_preprocessing import CustomPreprocessNormalizationLayer

# In[12]:

num_classes = 28

def create_model():
    # Define the CNN model
    model = tf.keras.models.Sequential()

    # Add the CustomPreprocessNormalizationLayer as the first layer
    model.add(CustomPreprocessNormalizationLayer())

    ### Add Convolutional and MaxPooling layers

    # CONV => RELU => MAX-POOLING
    model.add(Conv2D(32, 3, activation='relu',input_shape=(64, 64, 1)))
    model.add(MaxPooling2D((2, 2)))

    # CONV => RELU => MAX-POOLING
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # CONV => RELU => MAX-POOLING
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # CONV => RELU => MAX-POOLING
    model.add(Conv2D(256, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten the output for Dense layers
    model.add(Flatten())

    # Add Dense layers
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.2)) # Dropout layer with a dropout rate of 0.2

    # Add output layer with 28 Neurons
    model.add(Dense(num_classes,activation='softmax'))

    return model