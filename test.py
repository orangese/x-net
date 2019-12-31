"""

"test.py"

Various tests and benchmarking for YOLO-based nets.

"""

import os

import cv2
import matplotlib.pyplot as plt

from utils.parse import CLASSES, parse_labels, retrieve_annotations, prepare_for_eval
from utils.visuals import Draw
from yolo.yolo import YOLO


# ---------------- TESTING ----------------
if __name__ == "__main__":
    defaults = {
        "model_path": "/home/ryan/models/sixray/x-net/models/first/stage_1_v2/train_weights_v1.h5", # "/home/ryan/models/sixray/x-net/models/first/final/trained_weights_stage_1.h5",
        "anchors_path": "/home/ryan/models/sixray/x-net/anchors/sixray_anchors.txt",
        "classes_path": "/media/ryan/Data/x-ray-datasets/sixray/classes.txt"
    }

    net = YOLO(**defaults)
    annotations = retrieve_annotations("/media/ryan/Data/x-ray-datasets/sixray/images/annotations.csv")

    # prepare_for_eval(
    #     net=net,
    #     annotations=annotations,
    #     labels=parse_labels("/media/ryan/Data/x-ray-datasets/sixray/", label_type="test"),
    #     dump_paths=[None, "/home/ryan/scratchpad/mAP/input/detection-results-neg"],
    # )

    img_dir_path = "/media/ryan/Data/x-ray-datasets/sixray/images"
    for img_path in os.listdir(img_dir_path):
        if "P" in img_path and img_path.endswith(".jpg") or img_path.endswith(".png"):
            if "2" in annotations[img_path]:
                img = cv2.imread(os.path.join(img_dir_path, img_path))

                bounding_boxes, scores, classes = net.detect(img)
                print(annotations[img_path])

                plt.imshow(Draw.draw_on_img(img, bounding_boxes, scores, classes, CLASSES))
                plt.axis("off")
                plt.show()
