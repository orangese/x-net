"""

"main.py"

Various tests and benchmarking for YOLO-based nets.

"""

import os
from itertools import cycle
from timeit import default_timer as timer

import cv2
import matplotlib.pyplot as plt

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

    start = timer()

    img = cv2.imread(img_path)
    bounding_boxes, scores, classes = net.detect(img)

    print("{}: {} {} {}".format(img_path, bounding_boxes, scores, classes))

    if display:
        plt.imshow(Draw.draw_on_img(img, bounding_boxes, scores, classes, CLASSES, annotation=img_path))
        plt.axis("off")
        plt.show()

    return timer() - start


def test(net, path_to_sixray, num_imgs=None, display=True):
    """Test YOLO on SIXRay

    :param net: YOLO net
    :param path_to_sixray: path to sixray dataset
    :param num_imgs: number of images to test on
    :param display: whether or not to display images with bounding boxes (default: True)
    :returns: times to feed-forward each image

    """

    times = []

    for img_path in os.listdir(path_to_sixray):
        if "P" in img_path and img_path.endswith(".jpg") or img_path.endswith(".png"):
            times.append(test_on_img(net, os.path.join(path_to_sixray, img_path), display=display))

            if num_imgs and len(times) >= num_imgs:
                break

    total_time = sum(times)

    print("Average time (s): {}".format(total_time / len(times)))
    print("Total time (s): {}".format(total_time))

    return times


# ---------------- INTERACTIVE SLIDESHOW ----------------
def get_dict_and_cycle(path_to_sixray, imgs):
    """Retrieves dictionary of images and its cycle for slideshow

    :param path_to_sixray: path to sixray dataset
    :param imgs: relative path to images from sixray dataset
    :returns: dict mapping path to image array, cycle of that dict

    """

    img_dict = {}
    for img in imgs:
        path = os.path.join(path_to_sixray, img)
        img_dict[path] = cv2.imread(path)

    return img_dict, cycle(img_dict)


def slideshow(path_to_sixray, imgs, time_per_img):
    """Slideshow of SIXRay. It will cycle through set images and pauses to show annotations on a key press

    :param path_to_sixray: path to sixray dataset
    :param imgs: images to cycle through as relative paths from sixray
    :param time_per_img: time per slide

    """

    img_dict, img_cycle = get_dict_and_cycle(path_to_sixray, imgs)

    while True:
        img_path = next(img_cycle)

        fig = plt.gcf()

        fig.canvas.set_window_title(img_path)
        plt.axis("off")

        plt.imshow(img_dict[img_path])
        plt.pause(1e-6)
        plt.show(block=False)

        print("Current image: {}".format(img_path))

        if plt.waitforbuttonpress(time_per_img) is not None:
            print("Pause at {}".format(img_path))
            plt.waitforbuttonpress(0)
            print("[Annotations show here]")
            plt.waitforbuttonpress(0)


# ---------------- TESTING ----------------
if __name__ == "__main__":
    if "/home/ryan" in os.path.expanduser("~"):
        path_to_sixray = "/media/ryan/Data/x-ray-datasets/sixray/images"
    elif "/Users/ryan" in os.path.expanduser("~"):
        path_to_sixray = "/Users/ryan/data/sixray"
    else:
        raise ValueError("cannot run tests on this computer")

    imgs = ["P03879.jpg", "P06792.jpg", "P08109.jpg", "P06241.jpg"]

    slideshow(path_to_sixray, imgs, time_per_img=1)

    net = YOLO(**DEFAULTS)
    for img_path in [os.path.join(path_to_sixray, img) for img in imgs]:
        test_on_img(net, img_path)
