"""

"data_parsing.py"

Parses SIXray data.

"""

import os
import xml.etree.ElementTree as ET

import numpy as np


# CONSTANTS
CLASS_INDEXES = {
    "gun": 0,
    "knife": 1,
    "wrench": 2,
    "pliers": 3,
    "scissors": 4
}


# DATA PARSING
def parse_localization_data(path_to_sixray):
    """Parses SIXray annotations.

    :param path_to_sixray: path to SIXray. Assumes annotations stored in Annotations directory

    :return: dict with image paths mapped to a dict of one-hot encodings mapped to bounding boxes
    """

    os.chdir(os.path.join(path_to_sixray, "Annotation"))

    parse_text = lambda element_list: element_list[0].text.lower()
    parse_bounding_box = lambda element: np.array([float(coord.text) for coord in element])

    annotations = {}
    num_files = 0
    num_objs = 0

    for annotation_file in os.listdir(os.getcwd()):
        root = ET.parse(os.path.join(os.getcwd(), annotation_file))
        filename = parse_text(root.findall("./filename"))

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


if __name__ == "__main__":
    print(parse_localization_data("/media/ryan/Data/x-ray-datasets/SIXray"))
