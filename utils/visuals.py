"""

"utils/visuals.py"

Visuals utils.

"""


import colorsys

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from utils.parse import parse_results
from utils.rand import shuffle_with_seed


# ---------------- DRAW ----------------
class Draw:
    """Draws bounding box and label"""

    # INITS
    def __init__(self, img, classes):
        """Initializes Draw object

        :param img: img to draw on
        :param classes: list of all possible classes

        """

        self.img = img
        self.classes = classes

        self.color_init()
        self.font_init()

    def color_init(self):
        """Initializes random colors. Taken from https://github.com/qqwweee/keras-yolo3"""
        hsv_tuples = [(x / len(self.classes), 1., 1.) for x in range(len(self.classes))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        self.colors = shuffle_with_seed(colors).tolist()

    def font_init(self):
        """Initializes font and line settings"""
        self.font_size = 0.3
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.thickness = (self.img.shape[0] + self.img.shape[1]) // 300
        self.font_thickness = 1


    # DRAWING
    def draw(self, bounding_boxes, scores, classes, annotation=None):
        """Draws bounding box, score, and prediction on image

        :param bounding_boxes: bounding boxes as array
        :param scores: confidence scores as list or array
        :param classes: predicted classes as list or array
        :param annotation: annotation message to be put in top left corner (default: None)

        """

        black = (0, 0, 0)

        for bounding_box, score, class_id in zip(bounding_boxes, scores, classes):
            predicted_class = self.classes[class_id]

            y_max, x_min, y_min, x_max = bounding_box
            y_max = max(0, int(y_max.item() + 0.5))
            x_min = max(0, int(x_min.item() + 0.5))
            y_min = min(self.img.shape[1], int(y_min.item() + 0.5))
            x_max = min(self.img.shape[0], int(x_max.item() + 0.5))

            color = self.colors[class_id]
            label = "{} {:.2f}".format(predicted_class, score)

            cv2.rectangle(self.img, (x_min, y_min), (x_max, y_max), color, self.thickness)

            (width, height), _ = cv2.getTextSize(label, self.font, self.font_size, self.font_thickness)

            cv2.rectangle(self.img, (x_min - 1, y_max - 20), (x_min + width + 5, y_max), color, cv2.FILLED)
            cv2.putText(self.img, label, (x_min, y_max - 6), self.font, self.font_size, black, self.font_thickness)

        if annotation:
            cv2.putText(self.img, annotation, (0, 10), self.font, self.font_size, black, self.font_thickness)


    # CLASSMETHODS
    @classmethod
    def draw_on_img(cls, img, bounding_boxes, scores, classes, all_classes, annotation=None):
        """Draws on image without creating an instance of Draw

        :param img: image to draw on
        :param bounding_boxes: bounding boxes for image
        :param scores: confidence scores
        :param classes: predicted classes
        :param all_classes: full class list
        :param annotation: annotation for img (default: None)
        :returns: drawn-on image

        """

        drawer = cls(img, all_classes)
        drawer.draw(bounding_boxes, scores, classes, annotation=annotation)
        return drawer.img



# ---------------- PLOT RESULTS ----------------

# PLOT FUNC
def plot(results, mode, save_path=None, with_citations=False):
    """Plot results of X-Net vs. TSA vs. other models

    :param results: results dict
    :param mode: either 'localization' or 'classification', case-insensitive
    :param save_path: save path for plot (default: None)
    :param with_citations: include citations in plot or not (default: False)

    """

    def without_outliers():
        adj = results.copy()

        adj.pop("X-Net")
        adj.pop("TSA [5], [12], [16], [21]")

        return adj

    def draw_brackets():
        adj = {model: (time + 0.2, acc) for model, (time, acc) in without_outliers().items()}

        vertical_endpts = (adj[min(adj, key=lambda key: adj[key][-1])], adj[max(adj, key=lambda key: adj[key][-1])])
        vertical_endpts = list(zip(*vertical_endpts)) + ["black"]
        vertical_endpts[0] = (vertical_endpts[0][0], vertical_endpts[0][0])  # make sure the line isn't slanted

        (x_top, y_top), (x_bottom, y_bottom), _ = vertical_endpts

        horizontal_endpts = [
            (x_top - 0.075, y_top), (x_bottom, x_bottom), "black",
            (x_top - 0.075, y_top), (y_bottom, y_bottom), "black"
        ]

        plt.plot(*vertical_endpts)
        plt.plot(*horizontal_endpts)

        return tuple(zip(*vertical_endpts)), tuple(zip(*horizontal_endpts))

    results = {model: stats for model, stats in sorted(results.items(), key=lambda stat: stat[1][1], reverse=True)}
    times, accuracies = zip(*results.values())

    # plot points
    color = shuffle_with_seed(cm.rainbow(np.linspace(0, 1, len(accuracies))), seed=1234)
    plt.scatter(times, accuracies, c=color)

    # annotate outliers
    for model in results:
        if "X-Net" in model:
            plt.annotate(model, (results[model][0] + 0.1, results[model][1] + 0.25), weight="bold", fontsize=13)
        elif "TSA" in model:
            if with_citations:
                plt.annotate("TSA Officer", (results[model][0] - 1.75, results[model][1] + 2.7), weight="bold", fontsize=13)
                plt.annotate(model.replace("TSA", " "), (results[model][0], results[model][1] + 2.9))
            else:
                plt.annotate("TSA Officer", (results[model][0] - 0.875, results[model][1] + 2.7), weight="bold", fontsize=13)

    # annotate other points
    vertical_endpts, horizontal_endpts = draw_brackets()

    for idx, model in enumerate(without_outliers().keys()):
        if not with_citations:
            model = model[:model.find(" ")]
        if mode.lower() == "classification":
            plt.annotate(model, (vertical_endpts[0][0] + 0.1, vertical_endpts[0][1] - (idx * 4) + 1), fontsize=10)
        elif mode.lower() == "localization":
            plt.annotate(model, (vertical_endpts[0][0] + 0.1, vertical_endpts[1][1] - (idx * 4)), fontsize=10)
        else:
            raise ValueError("supported modes are 'classification' and 'localization'")

    # set up grid and plot
    plt.grid(which="both", linestyle=":")

    plt.title("{} mAP vs. Baggage Analysis Time".format(mode))
    plt.xlabel("Time to analyze an X-ray baggage scan (s)")
    plt.ylabel("{} mAP (%)".format(mode))

    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2 + 2, y1, 100))

    if save_path:
        plt.savefig(save_path)

    plt.show()


# ---------------- TESTING ----------------
if __name__ == "__main__":
    classification_results = parse_results("../results/text/classification_results.json")
    localization_results = parse_results("../results/text/localization_results.json")

    plot(classification_results, mode="Classification", save_path="../results/plots/classification_map.png")
    plot(localization_results, mode="Localization", save_path="../results/plots/localization_map.png")
