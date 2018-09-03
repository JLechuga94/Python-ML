from sklearn.model_selection import train_test_split
from termcolor import colored, cprint
import argparse
import tensorflow as tf
import pandas
import numpy as np


def main(argv):
    # Fetch the data
    print("\nLoading information from csv...")
    patient_data = pandas.read_csv("../project/audio_files/COMPLETE_ABCDF.csv")
    train, test = train_test_split(patient_data, test_size=0.3)

    print("\nCreating numeric labels for features...")
    train_labels = train.pop("patient").values
    test_labels = test.pop("patient").values
    numeric_train_labels = np.asarray([0 if label == "normal" else 1 for label in train_labels])
    numeric_test_labels = np.asarray([0 if label == "normal" else 1 for label in test_labels])
    print("\nProcessing finished")

    train_features = {"Values": train.values}
    test_features = {"Values": test.values}

    # Feature columns describe how to use the input.
    my_feature_columns = [tf.feature_column.numeric_column(key="Values", shape=(7501,))]

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[1024, 512, 256],
        # The model must choose between 3 classes.
        n_classes=2,

        model_dir="heartbeat_dnn_model2")

    # Set up logging for predictions
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    # tensors=tensors_to_log, every_n_iter=50)

    # Training function
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_features,
        y=numeric_train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # Train the Model.
    classifier.train(input_fn=train_input_fn, steps=10000)

    # Evaluate function
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_features,
        y=numeric_train_labels,
        num_epochs=1,
        shuffle=False)

    # Evaluate the model.
    evaluation = classifier.evaluate(input_fn=eval_input_fn)
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**evaluation))

    # Make predictions
    prediction_results = list(classifier.predict(input_fn=eval_input_fn))
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    print("------------------------------------")
    for index in range(10):
        print("\n")
        prediction_dict = prediction_results[index]
        real_value = test_labels[index]
        max_prob = np.amax(prediction_dict["probabilities"]) * 100
        prediction = prediction_dict["classes"][0].decode("UTF-8")
        if prediction == '0':
            prediction = 'normal'
        else:
            prediction = 'abnormal'
        if prediction == real_value:
            color = 'green'
        else:
            color = 'red'
        cprint(template.format(prediction, max_prob, real_value), color)
    print("------------------------------------")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
