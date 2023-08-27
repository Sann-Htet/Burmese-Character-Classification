import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

data_dir = '../dataset/township_alphabet_data'
idx2word = {}

for folder in os.listdir(data_dir):
    key, value = folder.split('_')
    idx2word[int(key)] = value

# Load model via "saved_model" format
model = tf.keras.models.load_model('../model')

image_path = "../images/taWanPu.png"
image = tf.keras.preprocessing.image.load_img(image_path) # load image
image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY) # convert to Grayscale
image = cv2.resize(image, (30, 30), interpolation=cv2.INTER_LINEAR) # rescale the image
# Apply thresholding to remove noise and enhance contrast
threshold_value = 140  # Adjust this threshold as needed
_, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY) # apply threshold to each pixels

image = tf.expand_dims(image, axis=-1) # add dim to the last axis
image = tf.cast(image, tf.float32) # cast pixel values to float
image = 255 - np.array(image)

# Apply padding to the image to create a border, and then resize it to 64x64 pixels.

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

image = tf.expand_dims(image, axis=0) # add dims to the first axis
img_input = tf.expand_dims(image, axis=-1) # add dims to the first axis

y_input_pred = model.predict(img_input)
print(idx2word.get(np.argmax(y_input_pred)))
print("Achieved a {:.2%} accuracy rate.".format(np.max(y_input_pred)))
