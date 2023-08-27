#!/usr/bin/env python
# coding: utf-8

# # Data Preparing

# In[1]:


import os
import numpy as np
import tensorflow as tf
import cv2


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


# In[3]:


print("Data shape:", data.shape)
print("Label shape", labels.shape)


# # Visualization

# In[4]:


import matplotlib.pyplot as plt


# In[5]:


idx2word = {}

for folder in os.listdir(data_dir):
    key, value = folder.split('_')
    idx2word[int(key)] = value


# In[36]:


# Sort the dictionary by keys in ascending order
sorted_items = sorted(idx2word.items())
sorted_dict = {k: v for k, v in sorted_items}
sorted_dict


# In[6]:


def show_images(x, y, title_str='Label'):
    for c in range(1, 10):
        plt.subplot(3, 3, c)
        i = np.random.randint(len(x)) # Generate random integer number
        im = x[i] # get i-th image
        plt.axis("off")
        index = int(y[i]) # get i-th label
        label = idx2word.get(index)
        plt.title("{} = {}".format(title_str, label))
        plt.imshow(im, cmap="Greys")
show_images(data, labels)


# # Data Preprocessing

# In[7]:


# Labels
label = np.unique(labels)
num_classes = len(label)
print("Number of classes:", num_classes)
print("Unique Labels:", label)


# In[8]:


# Create a data augmentation stage with rotations, zooms
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.1), # Adding random rotation by 10 %
        tf.keras.layers.RandomZoom(0.1), # Adding random zoom by 10 %
        tf.keras.layers.RandomBrightness(0.1),  # Adding brightness adjustment
        tf.keras.layers.GaussianNoise(0.01),  # Adding random noise (adjust the scale as needed)
    ]
)


# In[9]:


# Create tensorflow Dataset object to represents a potentially large set of elements
dataset = tf.data.Dataset.from_tensor_slices((data, labels))

# shuffle the dataset
dataset = dataset.shuffle(data.shape[0], reshuffle_each_iteration=False)

batch_size = 32

# get training, validation and testing dataset
data_test = dataset.take(1000).batch(batch_size) # take the first 1000 images from dataset
data_valid = dataset.skip(1000).take(1000).batch(batch_size) # take the second 1000 images from dataset
data_train = dataset.skip(2000).batch(batch_size).map(lambda x, y: (data_augmentation(x), y)) # take the rest and apply augmentation


# In[10]:


# Custom preprocessing layer with Normalization
class CustomPreprocessNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomPreprocessNormalizationLayer, self).__init__()

    def call(self, inputs):
        # Normalize pixel values to [0, 1]
        normalized_images = inputs / 255.0

        return normalized_images


# # Building the model

# In[11]:


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# In[12]:


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


# In[13]:


model = create_model()
model.build(input_shape=(None, 64, 64, 1)) # late variable creation
model.summary()


# In[14]:


# Configures the model for training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[15]:


# use TensorBoard, princess Aurora!
callbacks = [
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]


# In[16]:


# Train the model
epoch = 80
history = model.fit(data_train,
                    epochs=epoch,
                    validation_data=data_valid,
                    callbacks=callbacks
                   )


# In[17]:


test_loss, test_acc = model.evaluate(data_test)

print('Test accuracy: {:2.2f}%'.format(test_acc*100))


# In[18]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['train','validation'],loc='lower right')
plt.show()


# In[19]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(['train','validation'],loc='upper right')
plt.show()


# In[33]:


# Generate a random number between 11 and 990
random_number = np.random.randint(11, 991)
ds = data_test.unbatch().skip(random_number).take(10)

fig = plt.figure(figsize=(15, 7))
for j, (example, label) in enumerate(ds):
    ax = fig.add_subplot(2, 5, j+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(example.numpy(), cmap="Greys")
    pred = model.predict(tf.expand_dims(example, axis=0)) # prediction probs
    ans = idx2word[np.argmax(pred)] # convert prediction index to word
    actual = idx2word[int(label)] # actual label
    ax.set_title("Prediction: "+ ans +
                 "\n Actual: "+ actual)

plt.show()


# # Testing a real-world image

# In[21]:


image_path = "../images/taWanPu.png"
image = tf.keras.preprocessing.image.load_img(image_path) # load image
image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY) # convert to Grayscale
image = cv2.resize(image, (30, 30), interpolation=cv2.INTER_LINEAR) # rescale the image
# Apply thresholding to remove noise and enhance contrast
threshold_value = 140  # Adjust this threshold as needed
_, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY) # apply threshold to each pixels


# In[22]:


image = tf.expand_dims(image, axis=-1) # add dim to the last axis
image = tf.cast(image, tf.float32) # cast pixel values to float
image = 255 - np.array(image)
np.array(image).shape


# In[23]:


plt.imshow(image, cmap="Greys")


# In[24]:


# Apply padding to the image to create a border, and then resize it to 64x64 pixels.
import cv2
import numpy as np

# Define the padding values (top, bottom, left, right)
top = 50
bottom = 50
left = 50
right = 50

# Define the padding color (white)
padding_color = [0, 0, 0]

# Add padding to the image
image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR) # rescale the image
plt.imshow(image, cmap="Greys")


# In[25]:


image = tf.expand_dims(image, axis=0) # add dims to the first axis
img_input = tf.expand_dims(image, axis=-1) # add dims to the first axis
img_input.shape


# In[26]:


y_input_pred = model.predict(img_input)
print(idx2word.get(np.argmax(y_input_pred)))
print("Achieved a {:.2%} accuracy rate.".format(np.max(y_input_pred)))


# # Save Model

# In[27]:


# save model using "SavedModel" format
model.save('../model')


# In[28]:


# save the model using "HDF5" format
model.save('../model.h5')


# In[29]:


loaded_model = tf.keras.models.load_model("../model")

# Use the loaded model for predictions
predictions = loaded_model.predict(img_input)


# In[32]:


print(idx2word.get(np.argmax(predictions)))

