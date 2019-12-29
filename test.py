"""

"test.py"

Various tests and benchmarking for YOLO-based nets.

"""

import os

import cv2
import matplotlib.pyplot as plt

from utils.parse import CLASSES, parse_labels, retrieve_annotations
from utils.visuals import Draw
from yolo.yolo import YOLO


# ---------------- TESTING ----------------
if __name__ == "__main__":
    defaults = {
        "model_path": "/home/ryan/models/sixray/x-net/models/first/final/trained_weights_stage_1.h5",
        "anchors_path": "/home/ryan/models/sixray/x-net/anchors/sixray_anchors.txt",
        "classes_path": "/media/ryan/Data/x-ray-datasets/sixray/classes.txt"
    }

    net = YOLO(**defaults)

    labels = parse_labels("/media/ryan/Data/x-ray-datasets/sixray", sixray_set=10, label_type="test", full_path=False)
    annotations = retrieve_annotations("/home/ryan/scratchpad/sixray/sixray/annotations.csv")

    # prepare_for_eval(
    #    net,
    #    "/home/ryan/scratchpad/sixray/sixray2",
    #    retrieve_annotations("/media/ryan/Data/x-ray-datasets/SIXray/annotations.csv"),
    #    ["/home/ryan/scratchpad/mAP/input/ground-truth-letterbox", "/home/ryan/scratchpad/mAP/input/detection-results-letterbox"]
    # )

    img_dir_path = "/home/ryan/scratchpad/sixray/sixray/"
    for img_path in os.listdir(img_dir_path):
        if img_path.endswith(".jpg") or img_path.endswith(".png"):
            img = cv2.imread(os.path.join(img_dir_path, img_path))

            bounding_boxes, scores, classes = net.detect(img)
            print(annotations[img_path])

            plt.imshow(Draw.draw_on_img(img, bounding_boxes, scores, classes, CLASSES))
            plt.show()
