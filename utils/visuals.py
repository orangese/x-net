"""

"utils/visuals.py"

Visuals utils.

"""


import colorsys

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from utils.random import shuffle_with_seed


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
    def draw(self, bounding_boxes, scores, classes):
        """Draws bounding box, score, and prediction on image

        :param bounding_boxes: bounding boxes as array
        :param scores: confidence scores as list or array
        :param classes: predicted classes as list or array

        """

        for bounding_box, score, class_id in zip(bounding_boxes, scores, classes):
            predicted_class = self.classes[class_id]

            y_max, x_min, y_min, x_max = bounding_box
            y_max = max(0, int(y_max.item() + 0.5))
            x_min = max(0, int(x_min.item() + 0.5))
            y_min = min(self.img.shape[1], int(y_min.item() + 0.5))
            x_max = min(self.img.shape[0], int(x_max.item() + 0.5))

            color = self.colors[class_id]
            label = '{} {:.2f}'.format(predicted_class, score)
            black = (0, 0, 0)

            cv2.rectangle(self.img, (x_min, y_min), (x_max, y_max), color, self.thickness)

            text_size, _ = cv2.getTextSize(label, self.font, self.font_size, self.font_thickness)
            width, height = text_size

            cv2.rectangle(self.img, (x_min - 1, y_max - 20), (x_min + width + 5, y_max), color, cv2.FILLED)
            cv2.putText(self.img, label, (x_min, y_max - 6), self.font, self.font_size, black, self.font_thickness)


    # CLASSMETHODS
    @classmethod
    def draw_on_img(cls, img, bounding_boxes, scores, classes, all_classes):
        """Draws on image without creating an instance of Draw

        :param img: image to draw on
        :param bounding_boxes: bounding boxes for image
        :param scores: confidence scores
        :param classes: predicted classes
        :param all_classes: full class list
        :returns: drawn-on image

        """

        drawer = cls(img, all_classes)
        drawer.draw(bounding_boxes, scores, classes)
        return drawer.img



# ---------------- PLOT RESULTS ----------------

# RESULTS
classification_results = {
    "ResNet101 [17]": (0.045, 77.38),
    "ResNet101 + CHR [17]": (0.05, 79.37),
    "Inception-v3 [17]": (0.07 / 6, 77.01),
    "Inception-v3 + CHR [17]": (0.07 / 6, 79.49),
    "YOLOv3 [17]": (0.07, 78.70),
    "X-Net": (0.1, 83.45),
    "TSA [5], [12], [16], [21]": (5., 17.5)
}

localization_results = {
    "ResNet101 [17]": (0.045, 50.10),
    "ResNet101 + CHR [17]": (0.05, 51.35),
    "Inception-v3 [17]": (0.07 / 6, 62.92),
    "Inception-v3 + CHR [17]": (0.07 / 6, 63.54),
    "YOLOv3": (0.07, 53.68),
    "X-Net": (0.1, 74.93),
    "TSA [5], [12], [16], [21]": (5., 17.5)
}


# PLOT FUNC
def plot(results, mode):
    """Plot results of X-Net vs. TSA vs. other models

    :param results: results dict
    :param mode: either 'localization' or 'classification', case-insensitive

    """

    def draw_brackets():
        adj = {model: (time + 0.2, acc) for model, [time, acc] in results.items()}
        adj.pop("X-Net")
        adj.pop("TSA [5], [12], [16], [21]")

        vertical_endpts = (adj[min(adj, key=lambda key: adj[key][-1])], adj[max(adj, key=lambda key: adj[key][-1])])
        vertical_endpts = list(zip(*vertical_endpts))
        vertical_endpts[0] = (vertical_endpts[0][0], vertical_endpts[0][0])  # make sure the line isn't slanted

        horizontal_endpts = [
            (vertical_endpts[0][0] - 0.075, vertical_endpts[0][1]), (vertical_endpts[1][0], vertical_endpts[1][0]),
            "black",
            (vertical_endpts[0][0] - 0.075, vertical_endpts[0][1]), (vertical_endpts[1][1], vertical_endpts[1][1]),
            "black",
        ]

        plt.plot(*vertical_endpts, "black")
        plt.plot(*horizontal_endpts)

        return tuple(zip(*vertical_endpts)), tuple(zip(*horizontal_endpts))

    results = {model: stats for model, stats in sorted(results.items(), key=lambda stat: stat[1][1], reverse=True)}
    times, accuracies = zip(*results.values())

    # plot points
    color = cm.rainbow(np.linspace(0, 1, len(accuracies)))
    np.random.seed(1234)
    np.random.shuffle(color)
    np.random.seed(None)
    plt.scatter(times, accuracies, c=color)

    # annotate outliers
    for model in results:
        if "X-Net" in model:
            plt.annotate(model, (results[model][0] + 0.1, results[model][1] + 0.25), weight="bold", fontsize=13)
        elif "TSA" in model:
            plt.annotate("TSA Officer", (results[model][0] - 1.5, results[model][1] + 2.7), weight="bold", fontsize=13)
            plt.annotate(model.replace("TSA", " "), (results[model][0] - 0.0, results[model][1] + 2.9))

    # annotate other points
    vertical_endpts, horizontal_endpts = draw_brackets()

    middle_models = results.copy()
    middle_models.pop("TSA [5], [12], [16], [21]")
    middle_models.pop("X-Net")

    for idx, model in enumerate(middle_models.keys()):
        if mode.lower() == "classification":
            plt.annotate(model, (vertical_endpts[0][0] + 0.1, vertical_endpts[0][1] - (idx * 4) + 1), fontsize=10)
        elif mode.lower() == "localization":
            plt.annotate(model, (vertical_endpts[0][0] + 0.1, vertical_endpts[1][1] - (idx * 4)), fontsize=10)
        else:
            raise ValueError("supported modes are 'classification' and 'localization'")

    # set up grid and plot
    plt.grid(which="both", linestyle=":")

    plt.title("{} mAP vs. Time".format(mode))
    plt.xlabel("Time (s)")
    plt.ylabel("{} mAP (%)".format(mode))

    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2 + 2, y1, 100))

    plt.show()


# ---------------- TESTING ----------------
if __name__ == "__main__":
    plot(classification_results, mode="Classification")
    plot(localization_results, mode="Localization")
