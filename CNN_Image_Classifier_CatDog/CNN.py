pip install numpy==1.23.5

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow as tf

#Visual Domain Decathlon

class_names = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010']
img_size = 112 # CIFAR images are of size 32.
num_epochs = 15
batch_size = 160

# Download and prepare the CIFAR10 dataset
# The CIFAR10 dataset contains 60,000 color images in 10 classes, with 6,000 images in each class. The dataset is divided into
# 50,000 training images and 10,000 testing images. The classes are mutually exclusive and there is no overlap between them.

# If you get an error about unverified certification when loading the data, add the two following lines to the top of your code
'''import ssl
ssl._create_default_https_context = ssl._create_unverified_context 
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()'''


# Path to your local dataset
train_dir = r"./Visual Domain Decathlon (Subset)/train"
test_dir = r"./Visual Domain Decathlon (Subset)/test"

# Load training and testing datasets
train_dataset = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',  
    class_names=class_names,  
    image_size=(img_size, img_size),  
    batch_size=None 
)

test_dataset = image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    class_names=class_names,
    image_size=(img_size, img_size),
    batch_size=None
)

# Convert the dataset to numpy arrays to match CIFAR-10 structure
train_images, train_labels = zip(*[(image.numpy(), label.numpy()) for image, label in train_dataset])
test_images, test_labels = zip(*[(image.numpy(), label.numpy()) for image, label in test_dataset])

train_images = tf.convert_to_tensor(train_images)
train_labels = tf.convert_to_tensor(train_labels)
test_images = tf.convert_to_tensor(test_images)
test_labels = tf.convert_to_tensor(test_labels)
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# To verify that the dataset looks correct, let's plot the first 25 images from the training set:
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Create the convolutional base
# The 6 lines of code below define the convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.
# As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size.
# In this example, you will configure your CNN to process inputs of shape (32, 32, 3), which is the format of CIFAR images.
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=32, # The number of output filters in the convolution layer, e.g. 32 or 64
                              kernel_size=(3, 3), # The height and width of the 2D convolution, e.g.(3, 3)
                              strides=(1, 1), # The steps of the convolution or pooling window along the height and width
                              padding='same', # {'valid', 'same'}.
                              activation='relu', # {'relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential'} https://keras.io/api/layers/activations/
                              # relu(x) = max(x, 0)
                              input_shape=(img_size, img_size, 3))) # The dimensions of the input images: (height, width, number of channels). Input images should be of the same size.
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')) # Typically, as the width and height shrink, you can afford (computationally) to add more output channels in each Conv2D layer.
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

# Let's display the architecture of your model so far:
model.summary()
# Above, you can see that the output of every Conv2D and MaxPooling2D layer is a 3D tensor of shape (height, width, channels).
# The width and height dimensions tend to shrink as you go deeper in the network.

# First, you will flatten (or unroll) the 3D output to 1D, then add one or more Dense layers on top.
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=10))

# Here's the complete architecture of your model:
model.summary()

# Compile and train the model
model.compile(optimizer='adam', # You can find different optimizers here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # You can find different loss functions here: https://www.tensorflow.org/api_docs/python/tf/keras/losses
              metrics=['accuracy']) # List of metrics to be evaluated by the model during training and testing.
history = model.fit(x=train_images, y=train_labels, batch_size=batch_size, # batch_size is the number of samples per gradient update. default: 32.
                    epochs=num_epochs, # The number of training iterations
                    validation_data=(test_images, test_labels),
                    verbose=1) # {'auto', 0, 1, or 2} default='auto'.

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('Test accuracy is: ', test_acc)

# Create a graph of the training and validation accuracy over the training epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

#Cat-Dog

class_names = ['dog', 'cat']
img_size = 112 # CIFAR images are of size 32.
num_epochs = 15
batch_size = 160

# Download and prepare the CIFAR10 dataset
# The CIFAR10 dataset contains 60,000 color images in 10 classes, with 6,000 images in each class. The dataset is divided into
# 50,000 training images and 10,000 testing images. The classes are mutually exclusive and there is no overlap between them.

# If you get an error about unverified certification when loading the data, add the two following lines to the top of your code
'''import ssl
ssl._create_default_https_context = ssl._create_unverified_context 
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()'''


# Path to your local dataset
train_dir = r"./Cat-Dog/train"
test_dir = r"./Cat-Dog/test"

# Load training and testing datasets
train_dataset = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',  # Labels will be integers
    class_names=class_names,  # Use predefined class names
    image_size=(img_size, img_size),  # Resize images to 32x32
    batch_size=None  # Load all images into memory (matches CIFAR-10 behavior)
)

test_dataset = image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    class_names=class_names,
    image_size=(img_size, img_size),
    batch_size=None
)

# Convert the dataset to numpy arrays to match CIFAR-10 structure
train_images, train_labels = zip(*[(image.numpy(), label.numpy()) for image, label in train_dataset])
test_images, test_labels = zip(*[(image.numpy(), label.numpy()) for image, label in test_dataset])

train_images = tf.convert_to_tensor(train_images)
train_labels = tf.convert_to_tensor(train_labels)
test_images = tf.convert_to_tensor(test_images)
test_labels = tf.convert_to_tensor(test_labels)
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# To verify that the dataset looks correct, let's plot the first 25 images from the training set:
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Create the convolutional base
# The 6 lines of code below define the convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.
# As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size.
# In this example, you will configure your CNN to process inputs of shape (32, 32, 3), which is the format of CIFAR images.
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=32, # The number of output filters in the convolution layer, e.g. 32 or 64
                              kernel_size=(3, 3), # The height and width of the 2D convolution, e.g.(3, 3)
                              strides=(1, 1), # The steps of the convolution or pooling window along the height and width
                              padding='same', # {'valid', 'same'}.
                              activation='relu', # {'relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential'} https://keras.io/api/layers/activations/
                              # relu(x) = max(x, 0)
                              input_shape=(img_size, img_size, 3))) # The dimensions of the input images: (height, width, number of channels). Input images should be of the same size.
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')) # Typically, as the width and height shrink, you can afford (computationally) to add more output channels in each Conv2D layer.
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

# Let's display the architecture of your model so far:
model.summary()
# Above, you can see that the output of every Conv2D and MaxPooling2D layer is a 3D tensor of shape (height, width, channels).
# The width and height dimensions tend to shrink as you go deeper in the network.

# First, you will flatten (or unroll) the 3D output to 1D, then add one or more Dense layers on top.
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=2))

# Here's the complete architecture of your model:
model.summary()

# Compile and train the model
model.compile(optimizer='adam', # You can find different optimizers here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # You can find different loss functions here: https://www.tensorflow.org/api_docs/python/tf/keras/losses
              metrics=['accuracy']) # List of metrics to be evaluated by the model during training and testing.
history = model.fit(x=train_images, y=train_labels, batch_size=batch_size, # batch_size is the number of samples per gradient update. default: 32.
                    epochs=num_epochs, # The number of training iterations
                    validation_data=(test_images, test_labels),
                    verbose=1) # {'auto', 0, 1, or 2} default='auto'.

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('Test accuracy is: ', test_acc)

# Create a graph of the training and validation accuracy over the training epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
