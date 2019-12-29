"""

"utils/anchor.py"

Contains anchor generator functions for custom data.

"""

import numpy as np
from sklearn.cluster import KMeans


# ---------------- ANCHOR GENERATION ----------------
def generate_anchors(boxes_or_file, num_anchors):
    """Generates anchors given bounding boxes or a bounding box file using K-means

    :param boxes_or_file: either array of bounding boxes (width, height) or an bounding box annotations file
    :param num_anchors: number of anchors (i.e., clusters to be generated)
    :returns: generated anchors

    """

    def _generate_anchors(bounding_boxes, num_anchors):
        return KMeans(n_clusters=num_anchors, n_init=1).fit(bounding_boxes).cluster_centers_.astype(np.uint8)

    try:
        return _generate_anchors(boxes_or_file, num_clusters)
    except ValueError:
        return _generate_anchors(parse_annotations(boxes_or_file), num_anchors)


# ---------------- FILE IO ----------------
def parse_annotations(filename):
    """Retrieves and parses annotations already parsed by parse.py

    :param filename: annotations filename
    :return: array of bounding boxes with shape (num_boxes, 2)-- each box is just height and width

    """

    bounding_boxes = []
    with open(filename, "r") as file:
        for line in file:
            for box in line.rstrip().split(" ")[1:]:
                x_min, y_min, x_max, y_max = [float(coord) for coord in box.split(",")[:-1]]
                width = round(x_max - x_min)
                height = round(y_max - y_min)

                bounding_boxes.append((width, height))

    return np.array(bounding_boxes)


def write_anchors(anchor_file, annotation_file, num_anchors):
    """Writes anchors to a file

    :param anchor_file: file to write anchors
    :param annotation_file: bounding box annotations file
    :param num_anchors: number of anchors to be generated

    """

    anchors = generate_anchors(parse_annotations(annotation_file), num_anchors)
    with open(anchor_file, "w") as file:
        written_anchors = ""

        for idx, anchor in enumerate(anchors):
            x, y = anchor

            written_anchors += str(int(x)) + "," + str(int(y))
            if idx != len(anchors) - 1:
                written_anchors += ",  "

        file.write(written_anchors)

    print("Wrote {} anchors to {}".format(num_anchors, anchor_file))


# ---------------- TESTING ----------------
if __name__ == "__main__":
    num_clusters = 9  # 6 for tiny yolo
    filename = "/media/ryan/Data/x-ray-datasets/sixray/images/annotations.csv"
    print(generate_anchors(filename, num_clusters))
