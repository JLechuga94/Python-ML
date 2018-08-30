# Imports
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import logging
import pandas


tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 7501, 2])
  # input_layer = features["x"]

  # Convolutional Layer #1 and # Pooling Layer #1
  conv1 = tf.layers.conv1d(
      inputs=input_layer,
      filters=32,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=1, padding="same")

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv1d(
      inputs=pool1,
      filters=64,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=1, padding="same")

  # Dense Layer
  # pool2_flat = tf.reshape(pool2, [-1, 1875 * 64])
  dense = tf.layers.dense(inputs=pool2, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=2),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  print(labels.shape)
  print(logits.shape)
  print(labels)
  print(logits)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data

    patient_data = pandas.read_csv("../project/audio_files/COMPLETE_FORMATTED.csv")
    print(patient_data.head())
    print(patient_data.shape)

    train, test = train_test_split(patient_data, test_size=0.2)

    train_txt_labels = train.pop("patient").values
    test_txt_labels = test.pop("patient").values
    #
    train_data = train.values
    eval_data = test.values
    train_labels = np.asarray([0 if label == "normal" else 1 for label in train_txt_labels], dtype=np.int32)
    eval_labels = np.asarray([0 if label == "normal" else 1 for label in test_txt_labels], dtype=np.int32)


    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    print("------------------------------------")
    print(train_data.shape)
    print(train_labels)
    print(train_labels[0])
    print("------------------------------------")
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/heart_convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=5000,
        hooks=[logging_hook])

        # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    evaluation = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(evaluation)

    eval_results = list(mnist_classifier.predict(input_fn=eval_input_fn))
    print("------------------------------------")
    for index in range(10):
        element = eval_results[index]
        max_prob = np.argmax(element["probabilities"])
        clase = element["classes"]
        print("Index", index, "Real digit", test_txt_labels[index], "Predicted digit", clase)

    print("------------------------------------")

if __name__ == "__main__":
    tf.app.run()
