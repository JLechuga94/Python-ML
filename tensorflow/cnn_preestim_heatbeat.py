# Imports
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import logging
import pandas


tf.logging.set_verbosity(tf.logging.INFO)


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
    def input_evaluation_set():
        features = {'intensity': train_data}
        labels = train_labels
        return features, labels
    # Create the Estimator
    # mnist_classifier = tf.estimator.DNNClassifier(
    # feature_columns=my_feature_columns
    # model_fn=cnn_model_fn, model_dir="/tmp/heart_convnet_model")

    # Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.
    mnist_classifier = tf.estimator.DNNClassifier(
        feature_columns=train_data,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=2, model_dir="/tmp/heart_convnet_model")

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

    # eval_results = list(mnist_classifier.predict(input_fn=eval_input_fn))
    # print("------------------------------------")
    # for index in range(10):
    #     element = eval_results[index]
    #     max_prob = np.argmax(element["probabilities"])
    #     clase = element["classes"]
    #     print("Index", index, "Real digit", test_txt_labels[index], "Predicted digit", clase)
    #
    # print("------------------------------------")

if __name__ == "__main__":
    tf.app.run()
