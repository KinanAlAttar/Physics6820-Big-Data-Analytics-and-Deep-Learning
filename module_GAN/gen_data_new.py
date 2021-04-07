# example of loading the generator model and generating images
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)

from tensorflow.keras.utils import to_categorical
from collections import defaultdict



from tensorflow.keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
import argparse
import time
import numpy as np
from collections import defaultdict
from functools import partial

arg_parser = argparse.ArgumentParser() 
arg_parser.add_argument('--model ',
                            dest='model',
                            action="store",
                            type=str,
                            default='',
                            help='model file path')


arg_parser.add_argument('--num_samples ',
                            dest='num_samples',
                            action="store",
                            type=int,
                            default=100,
                            help='samples to generate')


arg_parser.add_argument('--latent_dim ',
                            dest='latent_dim',
                            action="store",
                            type=int,
                            default=100,
                            help='latent dimension')

arg_parser.add_argument('--digit ',
                            dest='digit',
                            action="store",
                            type=int,
                            default=-1,
                            help='digit to select (-1 if all digits 0..9)')

args = arg_parser.parse_args()

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
#    pyplot.show()
    filename = 'images/gen_test.png'
    pyplot.savefig(filename)
    pyplot.close()


# load model
print("Loading model: ",args.model)
model = load_model(args.model)
#
# generate images
latent_points = generate_latent_points(args.latent_dim, args.num_samples)
# generate images
test_images = model.predict(latent_points)
test_classes = np.argmax(test_images, axis=1)
test_labels = np.full((args.num_samples,1), args.digit)
print(latent_points.shape)

print(type(test_images),test_images.shape)
print(test_classes.shape)
# save the result
#  save_plot(test_images, args.digit)


#
# Your code here
#
# Load the CNN to classify MNIST images
#
# Run the predict method on the images you generated above
#
# Get the predicted class values (use np.argmax on the above predictions)
#
# Loop over the predictions and count the predicted classes: 
#.  Are they mostly correct?  
#.  Which is the highest count incorrect?
model_cnn = load_model("fully_trained_model_mnist_cnn.h5")
pred = model_cnn.predict(test_images)
classes = np.argmax(pred, axis=1)

d = dict.fromkeys(range(10),0)

for cl in classes:
  d[cl] += 1

correct = d[args.digit]
acc = correct/sum(d.values())
d[args.digit] = 0
most_incorrect = max(d, key=d.get)

print('Correct:', correct, 'Most Incorrect:', most_incorrect)
print('Accuracy:', acc)
print(d)
