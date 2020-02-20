"""

"main.py"

Various tests and benchmarking for YOLO-based nets.

"""

import json
import os
from itertools import cycle
from timeit import default_timer as timer

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.parse import CLASSES
from utils.visuals import Draw
from yolo.yolo import YOLO


# ---------------- CONFIG ----------------
DEFAULTS = {
    "model_path": "/home/ryan/models/sixray/x-net/models/v2/stage_1/trained.h5",
    "anchors_path": "/home/ryan/models/sixray/x-net/anchors/sixray_anchors.txt",
    "classes_path": "/media/ryan/Data/x-ray-datasets/sixray/classes.txt",
}


# ---------------- DEMONSTRATION ----------------
def test_on_img(net, img_path, display=True):
    """Detects and draws YOLO predictions on an image

    :param net: YOLO net object
    :param img_path: path to test image
    :param display: whether or not to display the image with bounding box outputs (default: True)
    :returns: elapsed time

    """

    results = {}

    img = cv2.imread(img_path)

    start = timer()
    bounding_boxes, scores, classes = net.detect(img)
    elapsed = timer() - start

    print(img_path + ":")
    results[img_path] = []
    for cls, score, box in zip(classes.tolist(), scores.tolist(), np.round(bounding_boxes).tolist()):
        print(" - ({}s) Detected {} with {}% confidence at {}".format(
            round(elapsed, 3), CLASSES[cls], round(score * 100, 2), box)
        )
        results[img_path].append({"id": cls, "score": score, "box": box})

    if display:
        plt.imshow(Draw.draw_on_img(img, bounding_boxes, scores, classes, annotation=img_path))
        plt.axis("off")
        plt.show()

    return results, elapsed


def test(net, imgs, num_imgs=None, display=True, write_to=None):
    """Test YOLO on SIXRay

    :param net: YOLO net
    :param imgs: path to sixray dataset or list of images to test on
    :param num_imgs: number of images to test on
    :param display: whether or not to display images with bounding boxes (default: True)
    :param write_to: path to write predictions to (default: None)
    :returns: times to feed-forward each image

    """

    times = []
    results = {}

    if isinstance(imgs, str):
        imgs = [path for path in os.listdir(path_to_sixray) if path.endswith(".jpg") or path.endswith(".png")]

    for img_path in imgs:
        if "P" in img_path and img_path.endswith(".jpg") or img_path.endswith(".png"):
            result, elapsed = test_on_img(net, os.path.join(path_to_sixray, img_path), display=display)

            results.update(result)
            times.append(elapsed)

            if num_imgs and len(times) >= num_imgs:
                break

    total_time = sum(times)

    print("Average time (s): {}".format(total_time / len(times)))
    print("Total time (s): {}".format(total_time))

    if write_to:
        with open(write_to, "a+") as file:
            json.dump(results, file, indent=4)

    return times


# ---------------- INTERACTIVE SLIDESHOW ----------------
def slideshow(imgs, path_to_detections, sec_per_img=1):
    """Slideshow of SIXRay. It will cycle through set images and pauses to show annotations on a key press

    :param imgs: images to cycle through
    :param path_to_detections: path to detections for imgs
    :param sec_per_img: seconds per slide (default: 1)

    """

    img_dict = {}
    for img_path in imgs:
        img_dict[img_path] = cv2.imread(img_path)
    img_cycle = cycle(img_dict)

    results = json.load(open(path_to_detections))

    while True:
        img_path = next(img_cycle)

        fig = plt.gcf()

        fig.canvas.set_window_title(img_path)
        plt.axis("off")

        img = img_dict[img_path]
        objs = results[img_path]

        ids = []
        scores = []
        boxes = []
        for obj in objs:
            ids.append(obj["id"])
            scores.append(obj["score"])
            boxes.append(np.array(obj["box"]))

        annotated_img = Draw.draw_on_img(img, boxes, scores, ids)

        plt.imshow(img)
        plt.pause(1e-6)
        plt.show(block=False)

        print("Current image: {}".format(img_path))

        if plt.waitforbuttonpress(sec_per_img) is not None:
            print("Pause at {}".format(img_path))

            plt.waitforbuttonpress(0)
            print("[Annotations show here]")
            plt.imshow(annotated_img)
            plt.pause(1e-6)
            plt.show(block=False)

            plt.waitforbuttonpress(0)
            print("Continuing slideshow")


# ---------------- TESTING ----------------
if __name__ == "__main__":
    if "/home/ryan" in os.path.expanduser("~"):
        path_to_sixray = "/media/ryan/Data/x-ray-datasets/sixray/images"
    elif "/Users/ryan" in os.path.expanduser("~"):
        path_to_sixray = "/Users/ryan/data/sixray"
    else:
        raise ValueError("cannot run tests on this computer")

    imgs = ["P03879.jpg", "P06792.jpg", "P08109.jpg", "P06241.jpg"]
    imgs = [os.path.join(path_to_sixray, img) for img in imgs]

    detections = "results/examples/detection.txt"
    slideshow(imgs, detections, sec_per_img=1)

    # net = YOLO(**DEFAULTS)
    # test(net, imgs, write_to="results/examples/detection.txt")
e