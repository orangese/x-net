"""

"data_parsing.py"

Parses SIXray data.

"""

import os
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np


CLASS_INDEXES = {
    "gun": 0,
    "knife": 1,
    "wrench": 2,
    "pliers": 3,
    "scissors": 4
}


# ---------------- DATA PARSING ----------------

# ANNOTATIONS
def parse_annotations(path_to_sixray):
    """Parses SIXray annotations.

    :param path_to_sixray: path to SIXray. Assumes annotations stored in "annotations" directory
    :return: dict with image paths mapped to a dict of one-hot encodings mapped to bounding boxes
    """

    os.chdir(os.path.join(path_to_sixray, "annotations"))

    parse_text = lambda element_list: element_list[0].text.lower()
    parse_bounding_box = lambda element: np.array([round(float(coord.text)) for coord in element])

    annotations = {}
    num_files = 0
    num_objs = 0

    for annotation_file in os.listdir(os.getcwd()):
        root = ET.parse(os.path.join(os.getcwd(), annotation_file))
        filename = parse_text(root.findall("./filename")).upper().replace("JPG", "jpg")

        annotations[filename] = {}

        objects = root.findall("./object")
        for obj in objects:
            try:
                class_vector = CLASS_INDEXES[parse_text(obj.findall("name"))]
                bounding_box = parse_bounding_box(obj.find("bndbox"))

                annotations[filename][class_vector] = bounding_box
                num_objs += len(annotations[filename])
            except IndexError:
                print("Ignoring empty <object></object> tag at {}".format(annotation_file))

        num_files += 1

    print("Parsed {} images and {} objects".format(num_files, num_objs))
    return annotations


def write_annotations(annotations, filename):
    with open(filename, "w") as file:
        for img in annotations:
            line = img + " "

            for obj in annotations[img]:
                bounding_box = annotations[img][obj]
                for coord in bounding_box:
                    line += str(coord) + ","
                line += str(obj) + " "

            file.write(line + "\n")

    print("Wrote annotations to {}".format(filename))


# LABELS
def parse_labels(path_to_sixray, sixray_set=10, label_type="train"):
    os.chdir(os.path.join(path_to_sixray, "labels", str(sixray_set)))
    parse_index = lambda index: int(index) if int(index) == 1 else 0

    labels = {}
    num_parsed, num_objects = 0, 0
    with open(label_type + ".csv") as label_file:
        for line in label_file:
            try:
                split = line.rstrip().split(",")

                img_path = split[0] + ".jpg"
                img_label = [parse_index(index) for index in split[1:]]

                labels[img_path] = img_label

                if 1 in img_label:
                    num_objects += 1
                num_parsed += 1
            except ValueError:
                pass

    print("Parsed {} labels, {} total objects".format(num_parsed, num_objects))
    return labels


# ---------------- DATA CONFIGURATION ----------------
def find_img(img_path, path_to_sixray):
    os.chdir(os.path.join(path_to_sixray, "images", img_path))
    for img_dir in os.listdir(os.getcwd()):
        if img_path in os.listdir(os.path.join(os.getcwd(), img_dir)):
            return os.listdir(os.path.join(os.getcwd(), img_dir, img_path))
    return -1


def reorganize_data(path_to_sixray, *args, **kwargs):
    labels = parse_labels(path_to_sixray, *args, **kwargs)

    for img in labels:
        full_img_path = find_img(img, path_to_sixray)
        print(full_img_path)


# ---------------- DATA VISUALIZATION ----------------
def show_bounding_boxes(path_to_sixray, img_dir):
    """Shows bounding boxes on SIXray annotated data.

    :param path_to_sixray: path to SIXray dataset
    :param img_dir: name of directory of images through which to iterate + display
    """

    annotations = parse_annotations(path_to_sixray)

    for img_name in os.listdir(os.path.join(path_to_sixray, img_dir)):
        img = cv2.imread(os.path.join(path_to_sixray, img_dir, img_name))
        img_name = img_name.upper().replace("JPG", "jpg")

        for obj in annotations[img_name]:
            x_min, y_min, x_max, y_max = annotations[img_name][obj]
            cv2.rectangle(
                img,
                (x_min, y_min),
                (x_max, y_max),
                color=(255, 0, 0),
                thickness=3
            )

        plt.gcf().canvas.set_window_title("SIXray visualization")

        plt.imshow(img, cmap="gray")
        plt.axis("off")

        plt.show()

        if input("'q' to quit:") == "q":
            break


if __name__ == "__main__":
    sixray = {
        "power": "/media/ryan/Data/x-ray-datasets/SIXray",
        "air": "/Users/ryan/Documents/Coding/Datasets/SIXray"
    }

    # parse_labels(sixray["power"], sixray_set=10)
    # show_bounding_boxes(sixray["power"], "images/20")
    # write_annotations(
        # parse_annotations(sixray["power"]),
        # os.path.join(sixray["power"], "annotations.csv")
    # )
