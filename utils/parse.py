"""

"utils/parse.py"

Parses SIXray data.

Data must be in the below structure:

./
├── annotations
├───── P00001.xml
├── images
├───── P00001.jpg
├── labels
├───── 10
├─────── train.csv
├─────── test.csv
├───── 100
└───── 1000

"""

import os
import functools
import json
import shutil
from xml.etree import ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from utils.decorators import restore_cwd


# ---------------- HELPERS AND SETUP ----------------

# CONSTANTS
CLASSES = ["gun", "knife", "wrench", "pliers", "scissors"]


# ---------------- DATA PARSING ----------------

# ANNOTATIONS
@restore_cwd
def parse_annotations(path_to_sixray):
    """Parses SIXray annotations

    :param path_to_sixray: path to SIXray-- assumes annotations stored in "annotations" directory

    :return: Annotations dictionary as filepath mapped to dictionary of objects mapped to bounding boxes

    """

    os.chdir(os.path.join(path_to_sixray, "annotations"))

    parse_text = lambda element_list: element_list[0].text.lower()
    parse_bounding_box = lambda element: np.array([round(float(coord.text)) for coord in element])

    annotations = {}
    num_files = 0
    num_objs = 0

    for annotation_file in os.listdir(os.getcwd()):
        root = ET.parse(os.path.abspath(annotation_file))
        filename = parse_text(root.findall("./filename")).upper().replace("JPG", "jpg")

        annotations[filename] = {}

        objects = root.findall("./object")
        for obj in objects:
            try:
                class_id = CLASSES.index([parse_text(obj.findall("name"))])
                bounding_box = parse_bounding_box(obj.find("bndbox"))

                if class_id in annotations[filename]:
                    annotations[filename][class_id].append(bounding_box)
                else:
                    annotations[filename][class_id] = [bounding_box]

                num_objs += len(annotations[filename])
            except IndexError:
                print("Ignoring empty <object></object> tag at {}".format(annotation_file))

        num_files += 1

    print("Parsed {} images and {} objects".format(num_files, num_objs))
    return annotations


def retrieve_annotations(filename):
    """Retrieves parsed annotations from filename

    :param filename: filename at which annotations reside

    :return: Annotations dictionary as filepath mapped to dictionary of objects mapped to bounding boxes

    """

    annotations = {}
    with open(filename, "r") as file:
        for line in file:
            split = line.rstrip().split(" ")
            path = split[0]
            annotations[path] = {}

            for box in split[1:]:
                box = box.split(",")
                label = box[-1]

                bounding_box = np.array([round(float(coord)) for coord in box[:-1]])
                if label in annotations[path]:
                    annotations[path][label].append(bounding_box)
                else:
                    annotations[path][label] = [bounding_box]

    return annotations


def filter_by_class(annotations, class_id):
    """Retrieves all files that have a specific class_id

    :param annotations: annotations dict
    :param class_id: id of class
    :returns: list of files containing specified class

    """

    results = []
    for file in annotations:
        if str(class_id) in annotations[file]:
            results.append(file)

    return results


def sort_by_class(annotations):
    """Sorts files in annotations by class id

    :param annotations: annotations dict
    :returns: dict mapping class id to list of files containing that class

    """

    sorted_files = {}
    for class_id in CLASSES:
        results = filter_by_class(annotations, CLASSES.index(class_id))
        sorted_files[class_id] = results
        print("Found {} files with '{}'".format(len(results), class_id))

    return sorted_files


def write_annotations(annotations, filename):
    """Writes annotations to filename

    :param annotations: annotations to write to file (generated using parse_annotations)
    :param filename: filename to write annotations to

    """

    with open(filename, "w") as file:
        for img in annotations:
            line = img + " "

            for obj in annotations[img]:
                for bounding_box in annotations[img][obj]:
                    for coord in bounding_box:
                        line += str(int(coord)) + ","
                    line += str(obj) + " "

            file.write(line + "\n")

    print("Wrote annotations to {}".format(filename))


# LABELS
@restore_cwd
def parse_labels(path_to_sixray, sixray_set=10, label_type="train", full_path=True):
    """Parses SIXray image labels

    :param path_to_sixray: path to SIXray dataset
    :param sixray_set: SIXray set (default is 10)
    :param label_type: either "train" or "test" (default is "train")
    :param full_path: dict keys as full or relative path to image

    :return: Labels as dictionary of object mapped to one-hot encoding

    """

    os.chdir(os.path.join(path_to_sixray, "images"))
    parse_index = lambda index: int(index) if int(index) == 1 else 0

    labels = {}
    num_parsed, num_objects = 0, 0
    label_file_path = os.path.join(path_to_sixray, "labels", str(sixray_set), label_type + ".csv")
    with open(label_file_path) as label_file:
        for line in label_file:
            try:
                split = line.rstrip().split(",")

                img_label = [parse_index(index) for index in split[1:]]
                img_path = os.path.abspath(split[0] + ".jpg")

                if not full_path:
                    img_path = img_path[img_path.rfind("/") + 1:]

                labels[img_path] = img_label

                if 1 in img_label:
                    num_objects += 1
                num_parsed += 1
            except ValueError:
                pass

    print("Parsed {} labels, {} total objects".format(num_parsed, num_objects))
    return labels


# ---------------- DATA CONFIGURATION ----------------
@restore_cwd
def copy(src, dest):
    """Copies png or jpg images from src to dest

    :param src: source directory
    :param dest: destination directory

    """

    os.chdir(src)
    print("Copying images from {} to {}".format(src, dest))

    with tqdm.trange(len(os.listdir(os.getcwd()))) as pbar:
        for file in os.listdir(os.getcwd()):
            if file.endswith(".jpg") or file.endswith(".png"):
                shutil.copyfile(file, os.path.join(dest, file))
                pbar.update()


def resize_imgs(img_dir, annotations, target_shape=(416, 416), annotation_file="annotations.csv"):
    """Resizes all images in target directory, along with their bounding boxes

    :param img_dir: target directory
    :param annotations: dict of annotations (bounding boxes)
    :param target_shape: new shape of images-- doesn't include channels (default: (416, 416))
    :param annotation_file: name of annotation file to write to (default: "annotations.csv")

    """

    print("Resizing images in {} to {}".format(img_dir, target_shape))

    with tqdm.trange(len(os.listdir(img_dir))) as pbar:
        for img_path in os.listdir(img_dir):
            if img_path.endswith(".jpg") or img_path.endswith(".png"):
                img = cv2.imread(os.path.join(img_dir, img_path))

                x_scale = target_shape[0] / img.shape[1]
                y_scale = target_shape[1] / img.shape[0]

                for obj in annotations[img_path]:
                    bounding_boxes = []

                    for bounding_box in annotations[img_path][obj]:
                        x_min, y_min, x_max, y_max = bounding_box

                        x_min = round(x_min * x_scale)
                        y_min = round(y_min * y_scale)
                        x_max = round(x_max * x_scale)
                        y_max = round(y_max * y_scale)

                        bounding_boxes.append(np.array([x_min, y_min, x_max, y_max]))

                    annotations[img_path][obj] = bounding_boxes

                cv2.imwrite(os.path.join(img_dir, img_path), cv2.resize(img, target_shape))

            pbar.update()

    write_annotations(annotations, os.path.join(img_dir, annotation_file))


# ---------------- DATA VISUALIZATION ----------------
def show_bounding_boxes(img_dir, annotations, color=(255, 0, 0), filter_fn=lambda f: True):
    """Shows bounding boxes on annotated data.

    :param img_dir: name of directory of images through which to iterate + display
    :param annotations: annotation set
    :param color: color of bounding box in RGB (default: red (255, 0, 0))
    :param filter_fn: filter to apply to images in img_dir

    """

    for img_name in os.listdir(img_dir):
        if filter_fn is None or filter_fn(img_name):
            img = cv2.imread(os.path.join(img_dir, img_name))
            img_name = img_name.upper().replace("JPG", "jpg")

            for obj in annotations[img_name]:
                for bounding_box in annotations[img_name][obj]:
                    x_min, y_min, x_max, y_max = bounding_box
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
                    cv2.putText(img, img_name, (0, 10), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 0), 1)

            plt.gcf().canvas.set_window_title("SIXray visualization")

            plt.imshow(img)
            plt.axis("off")

            plt.show()


# ---------------- DATA PREPARATIONS ----------------
def prepare_for_eval(net, annotations, labels, dump_paths):
    """Prepares annotations/detections for mAP evaluation

    :param net: YOLO net object
    :param annotations: annotations dict
    :param labels: labels dict (full_path=True)
    :param dump_paths: [truth dumping path, annotation dumping path]. Either can be None

    """

    truth_path, detection_path = dump_paths

    # annotation dump
    if truth_path is not None:
        print("Dumping annotations at {}".format(truth_path))
        for img_path in annotations:
            with open(os.path.join(truth_path, img_path.replace("jpg", "txt")), "w+") as truth_file:
                for cls_id in annotations[img_path]:
                    cls = CLASSES[int(cls_id)]

                    for bounding_box in annotations[img_path][cls_id]:
                        line = cls + " "

                        for coord in bounding_box:
                            line += str(coord) + " "

                        truth_file.write(line + "\n")

    # detection dump
    if detection_path is not None:
        print("Dumping predictions at {}".format(detection_path))
        labels = [img_path for img_path in labels if "P" in img_path]

        with tqdm.trange(len(labels)) as pbar:
            for img_path in labels:
                # open and predict
                try:
                    img = cv2.imread(img_path)
                except ValueError:
                    print("Could not open {}".format(img_path))
                    continue

                bounding_boxes, scores, classes = net.detect(img)

                # create and write to file
                img_predictions = ""
                for cls_id, bounding_box, score in zip(classes, bounding_boxes, scores):
                    prediction = CLASSES[cls_id]
                    bounding_box = [int(round(coord)) for coord in bounding_box]

                    img_predictions += "{} {} {} {} {} {}\n".format(prediction, round(score, 3), *bounding_box)

                path = img_path[img_path.rfind("/") + 1:].replace("jpg", "txt")
                with open(os.path.join(detection_path, path), "w+") as prediction_file:
                    prediction_file.write(img_predictions)

                pbar.update()


# ---------------- RESULTS ----------------

# PARSE MODEL EVAL
def parse_model_results(filepath):
    """Parses classification or localization results from a json file

    :param filepath: path to json results file
    :returns: dict of results

    """

    results = json.load(open(filepath))
    for model in results:
        time, acc = results[model].split(", ")
        results[model] = (float(time), float(acc))

    return results


# ---------------- TESTING ----------------
if __name__ == "__main__":
    def yolo_benchmark_format(src, dest, src_annotations):
        # formatting for benchmark YOLO training
        copy(src, dest)
        resize_imgs(dest, retrieve_annotations(src_annotations))

    annotated_imgs = "/media/ryan/Data/x-ray-datasets/sixray/images/"

    anns = retrieve_annotations(annotated_imgs + "annotations.csv")

    ct = 0
    for file in anns:
        for obj in anns[file]:
             ct += len(obj)
    print("Total obj ct: {}".format(ct))

    false_positive = ['P03585.jpg', 'P08758.jpg', 'P04435.jpg', 'P08701.jpg', 'P08833.jpg', 'P08902.jpg', 'P08888.jpg', 'P00673.jpg', 'P07425.jpg', 'P03862.jpg', 'P04107.jpg', 'P08008.jpg', 'P08893.jpg', 'P08025.jpg', 'P02574.jpg', 'P05605.jpg', 'P05528.jpg', 'P05325.jpg', 'P06240.jpg', 'P08466.jpg', 'P07701.jpg', 'P05286.jpg', 'P06239.jpg', 'P08882.jpg', 'P08670.jpg', 'P04504.jpg', 'P08884.jpg', 'P06838.jpg', 'P05493.jpg', 'P08758.jpg', 'P07586.jpg', 'P06409.jpg', 'P08135.jpg', 'P08851.jpg', 'P07687.jpg', 'P07855.jpg', 'P04774.jpg', 'P08857.jpg', 'P08132.jpg', 'P07585.jpg', 'P04578.jpg', 'P07526.jpg', 'P00763.jpg', 'P07219.jpg', 'P04039.jpg', 'P07217.jpg', 'P01920.jpg', 'P06141.jpg', 'P07673.jpg', 'P08788.jpg', 'P06240.jpg', 'P08296.jpg', 'P03470.jpg', 'P08219.jpg', 'P08857.jpg', 'P04361.jpg', 'P08908.jpg', 'P08559.jpg', 'P06156.jpg', 'P00276.jpg', 'P06725.jpg', 'P07218.jpg']

    show_bounding_boxes(
        annotated_imgs,
        retrieve_annotations(annotated_imgs + "/annotations.csv"),
        filter_fn=lambda f: f in false_positive
    )
