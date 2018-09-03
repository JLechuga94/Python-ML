from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas
import numpy as np

patient_data = pandas.read_csv("../project/audio_files/COMPLETE_ABCDF.csv")
print(patient_data.head())
print(patient_data.shape)

train, test = train_test_split(patient_data, test_size=0.2)

train_labels = train.pop("patient").values
test_labels = test.pop("patient").values
numeric_train_labels = np.asarray([0 if label == "normal" else 1 for label in train_labels])
numeric_test_labels = np.asarray([0 if label == "normal" else 1 for label in test_labels])

train = train.values
test = test.values

model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)

])

# # Create a sigmoid layer:
# layers.Dense(64, activation='sigmoid')
# # Or:
# layers.Dense(64, activation=tf.sigmoid)
#
# # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
# layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))
# # A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
# layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01))
#
# # A linear layer with a kernel initialized to a random orthogonal matrix:
# layers.Dense(64, kernel_initializer='orthogonal')
# # A linear layer with a bias vector initialized to 2.0s:
# layers.Dense(64, bias_initializer=keras.initializers.constant(2.0))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Configure a model for categorical classification.
# model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
#               loss=keras.losses.categorical_crossentropy,
#               metrics=[keras.metrics.categorical_accuracy])

# Instantiates a toy dataset instance:
# dataset = tf.data.Dataset.from_tensor_slices((train, numeric_train_labels))
# dataset = dataset.batch(32)
# dataset = dataset.repeat()

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(train, numeric_train_labels, epochs=10, steps_per_epoch=30)

test_loss, test_acc = model.evaluate(test, numeric_test_labels)
print('Test accuracy:', test_acc)


# # Save entire model to a HDF5 file
# model.save('modelo_keras_1.h5')
#
# # Recreate the exact same model, including weights and optimizer.
# model = keras.models.load_model('modelo_keras_1.h5')
