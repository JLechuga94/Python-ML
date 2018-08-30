from tensorflow import keras
from termcolor import colored, cprint
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas


patient_data = pandas.read_csv("../project/audio_files/COMPLETE_ABCDF.csv")
print(patient_data.head())
print(patient_data.shape)

train, test = train_test_split(patient_data, test_size=0.3)

train_labels = train.pop("patient").values
test_labels = test.pop("patient").values
numeric_train_labels = np.asarray([0 if label == "normal" else 1 for label in train_labels])
numeric_test_labels = np.asarray([0 if label == "normal" else 1 for label in test_labels])

train = train.values
test = test.values

model = keras.Sequential([
    # keras.layers.Flatten(input_shape=(1, 7502)),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])
#
# Metrics for accuracy of the model and optimization of training
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
#             loss=keras.losses.categorical_crossentropy,
#             metrics=[keras.metrics.categorical_accuracy])

model.fit(train, numeric_train_labels, epochs=10, steps_per_epoch=30)
test_loss, test_acc = model.evaluate(test, numeric_test_labels)

print('Test accuracy:', test_acc)

# Array of accuracy for the two values 'normal' or 'abnormal'
predictions = model.predict(test)
for index in range(10):
    prediction = ""
    if np.argmax(predictions[index]) == 0:
        prediction = "normal"
    else:
        prediction = "abnormal"
    if prediction == test_labels[index]:
        color = 'green'
    else:
        color = 'red'
    print("\nTest value index:", index)
    cprint("Prediction made: " + prediction, color)
    cprint("Real value: "+test_labels[index], color)
# print(predictions[index]) # Prediction label for the 1st element
# print(np.argmax(predictions[index])) # Obtain the highest % label for the first element
# print(test_labels[index])
#
#
