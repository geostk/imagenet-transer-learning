import os
import pickle
import re
import warnings

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.platform import gfile

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

model_dir = "../models/imagenet"
dataset_dir = "../datasets/product-image-cat/"

images_dir = os.path.join(dataset_dir, "images")
image_list = [os.path.join(images_dir, f) for f in os.listdir(images_dir)
              if re.search(r"jpg|JPG", f)]

data_dir = os.path.join(dataset_dir, "data")
features_file = os.path.join(data_dir, "features")
labels_file = os.path.join(data_dir, "labels")


def create_graph():
    with gfile.FastGFile(
                os.path.join(model_dir, "classify_image_graph_def.pb"),
                "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="")


def extract_features(list_images):
    nb_features = 2048
    _features = np.empty((len(list_images), nb_features))
    _labels = []

    create_graph()

    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name("pool_3:0")
        for i, image in enumerate(list_images):
            print("Processing: {}".format(image))
            if not gfile.Exists(image):
                tf.logging.fatal("File does not exist %s", image)
            image_data = gfile.FastGFile(image, "rb").read()
            predictions = sess.run(next_to_last_tensor,
                                   {"DecodeJpeg/contents:0": image_data})
            _features[i, :] = np.squeeze(predictions)
            _labels.append(re.split('_\d+', image.split('/')[1])[0])
    return _features, _labels


features, labels = extract_features(image_list)

# Save features and labels...
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

pickle.dump(features, open(features_file, "wb"))
pickle.dump(labels, open(labels_file, "wb"))

features = pickle.load(open(features_file, "rb"))
labels = pickle.load(open(labels_file, "rb"))

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    test_size=0.1,
                                                    random_state=42)

print("X_train =", len(X_train), "- y_train =", len(y_train))
print("X_test =", len(X_test), "- y_test =", len(y_test))

clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(y_test)


def plot_confusion_matrix(y_true, _y_pred):
    cm_array = confusion_matrix(y_true, _y_pred)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(_y_pred)

    plt.imshow(cm_array[:-1, :-1], interpolation='nearest', cmap=plt.cm.Blue)
    plt.title("Confusion matrix", fontsize=16)

    color_bar = plt.colorbar(fraction=0.046, pad=0.04)
    color_bar.set_label('Number of images', rotation=270,
                        labelpad=30, fontsize=12)

    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks, pred_labels)

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)

    plt.tight_layout()

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12

    plt.rcParams["figure.figsize"] = fig_size


print("Accuracy: {:.2%}".format(accuracy_score(y_test, y_pred)))
plot_confusion_matrix(y_test, y_pred)
