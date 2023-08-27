import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = '../dataset/township_alphabet_data'
idx2word = {}

for folder in os.listdir(data_dir):
    key, value = folder.split('_')
    idx2word[int(key)] = value

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