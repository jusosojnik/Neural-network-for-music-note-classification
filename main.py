import tensorflow as tf
from keras.utils.np_utils import to_categorical
from tensorflow.python import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.transform as trf
from skimage import io
from skimage.color import rgb2gray
import time
import random
import scipy
import matplotlib
import glob
import os
import filetype


# mnist = tf.keras.datasets.cifar10  # 28x28 images of hand-written digits 0-9
#
# (x_train2, y_train2), (x_test2, y_test2) = mnist.load_data()

x_train, x_validate, x_test, y_train, y_validate, y_test = [], [], [], [], [], []
# for filename in glob.glob('datasets/training/Notes/Eight/*.jpg'): #assuming gif
#     img = io.imread(filename)
#     x_train.append(img)

# img = io.imread('./datasets-augmented/test/Notes/Eight/Eight_original_e851.jpg_01e33a25-9c90-4e44-97f1-5cfa422e8964.jpg')
# plt.imshow(img)
# plt.show()

category = -2
for subdir, dirs, files in os.walk('datasets/training/Notes'):
    category += 1
    for file in files:
        if filetype.is_image(os.path.join(subdir, file)):
            img = io.imread(os.path.join(subdir, file))
            x_train.append(rgb2gray(img))
            y_train.append([category])

category = -2
for subdir, dirs, files in os.walk('datasets/validation/Notes'):
    category += 1
    for file in files:
        if filetype.is_image(os.path.join(subdir, file)):
            img = io.imread(os.path.join(subdir, file))
            x_validate.append(rgb2gray(img))
            y_validate.append([category])

category = -2
for subdir, dirs, files in os.walk('datasets/test/Notes'):
    category += 1
    for file in files:
        if filetype.is_image(os.path.join(subdir, file)):
            img = io.imread(os.path.join(subdir, file))
            x_test.append(rgb2gray(img))
            y_test.append([category])


x_train, y_train, x_validate, y_validate, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_validate), np.array(
    y_validate), np.array(x_test), np.array(y_test)

# print(x_train2)
# print(x_train)

# plt.imshow(x_train[0])
# plt.imshow(x_train[3000])
# plt.imshow(x_train[500])
# plt.imshow(x_train[750])
# plt.imshow(x_train[1000])

# print(x_train[0])
# plt.imshow(x_train[0])
# plt.show()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_validate = tf.keras.utils.normalize(x_validate, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# print(x_train)

x_train = x_train.reshape(x_train.shape[0], 64, 64, 1)
x_validate = x_validate.reshape(x_validate.shape[0], 64, 64, 1)
x_test = x_test.reshape(x_test.shape[0], 64, 64, 1)

y_train = to_categorical(y_train, 5)
y_validate = to_categorical(y_validate, 5)
y_test = to_categorical(y_test, 5)

# print(x_train)

# print(x_train[0])
# plt.imshow(x_train[0])
# plt.show()

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=100)
#
#
# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)
#
# model.save('num_reader.model')

kernel_size = 5

model = keras.Sequential()
model.add(keras.layers.Convolution2D(32, (kernel_size, kernel_size), activation='relu',
                                     input_shape=(64, 64, 1)))
model.add(keras.layers.Convolution2D(16, (kernel_size, kernel_size), activation='relu'))
model.add(keras.layers.Convolution2D(16, (kernel_size, kernel_size), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(5, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cbs = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2,
        verbose=1)
]

hist = model.fit(x_train, y_train, epochs=9001, validation_data=(x_validate, y_validate),
                 callbacks=cbs)

new_model = tf.keras.models.load_model('num_reader.model')

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

plt.close('all')
plt.figure()
plt.plot(hist.history['val_loss'])
plt.figure()
plt.plot(hist.history['val_accuracy'])
plt.show()

# predictions = new_model.predict([x_test])
# plt.imshow(x_test[0])
# plt.imshow(x_test[250])
# plt.imshow(x_test[500])
# plt.imshow(x_test[750])
# plt.imshow(x_test[1000])
# plt.show()
# predictions = new_model.predict([x_test])
#
# for prediction in predictions:
#     print(prediction)
#
#
# plt.imshow(x_train[1])
# plt.show()
# print(len(y_train))
