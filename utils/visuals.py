"""

"utils/visuals.py"

Visuals utils.

"""


import colorsys

import cv2
import numpy as np


class Draw:
    """Draws bounding box and label"""

    # INITS
    def __init__(self, img, classes):
        self.img = img
        self.classes = classes

        self.color_init()
        self.font_init()

    def color_init(self):
        # TAKEN FROM https://github.com/qqwweee/keras-yolo3
        hsv_tuples = [(x / len(self.classes), 1., 1.) for x in range(len(self.classes))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

    def font_init(self):
        self.font_size = 0.3
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.thickness = (self.img.shape[0] + self.img.shape[1]) // 300


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

            cv2.rectangle(self.img, (x_min, y_min), (x_max, y_max), color, self.thickness)

            cv2.rectangle(self.img, (x_min, y_max - 20), (x_max, y_max), color, cv2.FILLED)
            cv2.putText(self.img, label, (x_min + 6, y_max - 6), self.font, self.font_size, (0, 0, 0), thickness=1)


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
