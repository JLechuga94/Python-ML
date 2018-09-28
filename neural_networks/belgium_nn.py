from skimage import transform
from skimage.color import rgb2gray
import tensorflow as tf
import os
import skimage.data as imd
import numpy as np
import matplotlib.pyplot as plt
import random

main_dir = "../../datasets/belgian/"
train_data_dir = os.path.join(main_dir, "Training")
test_data_dir = os.path.join(main_dir, "Testing")

def load_ml_data(data_directory):
    dirs = [d for d in os.listdir(data_directory)
            if os.path.isdir(os.path.join(data_directory,d))]
    labels = []
    images = []
    for d in dirs:
        label_dir = os.path.join(data_directory, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        for f in file_names:
            images.append(imd.imread(f))
            labels.append(int(d))
    return images, labels


images, labels = load_ml_data(train_data_dir)

images = np.array(images)
labels = np.array(labels)

print(len(set(labels))) # Number of output neurons for labels
# plt.hist(labels, len(set(labels))) # Shows the amount of images per label
#
def show_images(images):
    rand_signs = random.sample(range(0,len(labels)), 6)
    for i in range(len(rand_signs)):
        temp_im = images[rand_signs[i]]
        plt.subplot(1, 6, i+1)
        plt.axis("off")
        plt.imshow(temp_im, cmap="gray")
        plt.subplots_adjust(wspace = 0.5)
        print("Forma:{}, min:{}, max:{}".format(temp_im.shape, temp_im.min(), temp_im.max()))
    plt.show()

# unique_labels = set(labels)
# plt.figure(figsize=(16,16))
# i = 1
# for label in unique_labels:
#     temp_im = images[list(labels).index(label)]
#     plt.subplot(8,8, i)
#     plt.axis("off")
#     plt.title("Clase{} ({})".format(label, list(labels).count(label)))
#     i += 1
#     plt.imshow(temp_im)
# plt.show()

# Code for obtaining the smalles values for height and width
w = 9999
h = 9999
for image in images:
    if image.shape[0] < h:
        h = image.shape[0]
    if image.shape[1] < w:
        w = image.shape[1]
print("Tamaño mínimo: ",h,"x",w)

images30 = rgb2gray(np.array([transform.resize(image, (30,30)) for image in images]))
# print(images30[0]) # Show the first element of resized images
show_images(images30)
