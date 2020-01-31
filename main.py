"""

"main.py"

Various tests and benchmarking for YOLO-based nets.

"""

import os
from timeit import default_timer as timer

import cv2
import matplotlib.pyplot as plt

from utils.parse import CLASSES, retrieve_annotations
from utils.visuals import Draw
from yolo.yolo import YOLO


# ---------------- CONFIG ----------------
DEFAULTS = {
    "model_path": "/home/ryan/models/sixray/x-net/models/v2/stage_1/trained.h5",
    "anchors_path": "/home/ryan/models/sixray/x-net/anchors/sixray_anchors.txt",
    "classes_path": "/media/ryan/Data/x-ray-datasets/sixray/classes.txt"
}

# ---------------- DEMONSTRATION ----------------
def test(net, path_to_sixray, num_imgs=None, display=True):
    times = []

    for img_path in os.listdir(path_to_sixray):
        if "P" in img_path and img_path.endswith(".jpg") or img_path.endswith(".png"):
            img = cv2.imread(os.path.join(path_to_sixray, img_path))

            start = timer()
            bounding_boxes, scores, classes = net.detect(img)
            times.append(timer() - start)

            if display:
                plt.imshow(Draw.draw_on_img(img, bounding_boxes, scores, classes, CLASSES, annotation=img_path))
                plt.axis("off")
                plt.show()

            if num_imgs and len(times) >= num_imgs:
                break

    print("Average time (s): {}".format(sum(times) / len(times)))
    print("Total time (s): {}".format(sum(times)))


# ---------------- TESTING ----------------
if __name__ == "__main__":
    net = YOLO(**DEFAULTS)
    annotations = retrieve_annotations("/media/ryan/Data/x-ray-datasets/sixray/images/annotations.csv")
    path_to_sixray = "/media/ryan/Data/x-ray-datasets/sixray/images"

    test(net, path_to_sixray, num_imgs=50, display=False)
