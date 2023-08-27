import os
import numpy as np
import tensorflow as tf
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from model import create_model

# In[2]:


data_dir = '../dataset/township_alphabet_data'
image_size = (64, 64)  # Adjust the image size

def load_and_process(data_dir):
    data = []  # To store image data
    labels = []  # To store corresponding labels
    for folder in os.listdir(data_dir):
        label = folder.split('_')[0]  # Use the folder name as the label (0, 1, ..., 27)
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = tf.keras.utils.load_img(img_path) # load image
            img = tf.image.rgb_to_grayscale(img) # convert 3-D to 1-D grayscale
            img = tf.image.resize(img, size=image_size, method='area') # resize image to (64, 64)
            img = 255 - img
            data.append(img)
            labels.append(int(label))
    return data, labels

data, labels = load_and_process(data_dir)
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.uint8)[:, np.newaxis]

idx2word = {}

for folder in os.listdir(data_dir):
    key, value = folder.split('_')
    idx2word[int(key)] = value

label = np.unique(labels)
num_classes = len(label)

# Create a data augmentation stage with rotations, zooms
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.1), # Adding random rotation by 10 %
        tf.keras.layers.RandomZoom(0.1), # Adding random zoom by 10 %
        tf.keras.layers.RandomBrightness(0.1),  # Adding brightness adjustment
        tf.keras.layers.GaussianNoise(0.01),  # Adding random noise (adjust the scale as needed)
    ]
)

# Create tensorflow Dataset object to represents a potentially large set of elements
dataset = tf.data.Dataset.from_tensor_slices((data, labels))

# shuffle the dataset
dataset = dataset.shuffle(data.shape[0], reshuffle_each_iteration=False)

batch_size = 32

# get training, validation and testing dataset
data_test = dataset.take(1000).batch(batch_size) # take the first 1000 images from dataset
data_valid = dataset.skip(1000).take(1000).batch(batch_size) # take the second 1000 images from dataset
data_train = dataset.skip(2000).batch(batch_size).map(lambda x, y: (data_augmentation(x), y)) # take the rest and apply augmentation

model = create_model()
model.build(input_shape=(None, 64, 64, 1)) # late variable creation
model.summary()

# Configures the model for training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# use TensorBoard, princess Aurora!
callbacks = [
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]


# Train the model
epoch = 80
history = model.fit(data_train,
                    epochs=epoch,
                    validation_data=data_valid,
                    #callbacks=callbacks
                   )


test_loss, test_acc = model.evaluate(data_test)

print('Test accuracy: {:2.2f}%'.format(test_acc*100))