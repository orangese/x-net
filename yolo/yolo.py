"""

"yolo/yolo.py"

Class definition of YOLO_v3 style detection model on image and video

"""

import cv2
import keras
from keras import backend as K
import numpy as np
import tensorflow as tf

from yolo.head.backend import yolo_eval
from yolo.head.join import yolo


# ---------------- YOLO! ----------------
class YOLO:
    """YOLO as a class"""

    HYPERPARAMS = {
        "img_size": (416, 416),
        "score": 0.3,
        "iou": 0.45,
    }

    BACKBONES = [
        "x-net",
        "darknet",
    ]


    # INITS
    def __init__(self, model_path, anchors_path, classes_path, backbone="x-net", **kwargs):
        self.HYPERPARAMS.update(kwargs)

        self.model_path = model_path
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.backbone = backbone

        self.anchor_and_cls_init()
        self.model_init()

    def anchor_and_cls_init(self):
        with open(self.anchors_path) as anchor_file:
            self.anchors = np.array(anchor_file.readline().split(","), dtype=np.float32).reshape(-1, 2)
        with open(self.classes_path) as classes_file:
            self.classes = [cls.strip() for cls in classes_file]

    def model_init(self):
        assert self.model_path.endswith(".h5"), "only keras .h5 files supported"
        assert self.backbone in self.BACKBONES, "supported backbones are {}".format(self.BACKBONES)

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        K.set_session(self.sess)

        inputs = keras.layers.Input((*self.HYPERPARAMS["img_size"], 3))
        self.yolo = yolo(inputs, len(self.anchors) // 3, len(self.classes), backbone_type=self.backbone)
        self.yolo.load_weights(self.model_path)
        print("{} loaded".format(self.model_path))

        self.img_size_tensor = K.placeholder(shape=(2,))
        self.bounding_boxes, self.scores, self.classes = yolo_eval(
            yolo_outputs=self.yolo.output,
            anchors=self.anchors,
            num_classes=len(self.classes),
            image_shape=self.img_size_tensor,
            score_threshold=self.HYPERPARAMS["score"],
            iou_threshold=self.HYPERPARAMS["iou"]
        )


    # DETECTION
    def detect(self, img):
        """Detects objects in image

        :param img: image as array with shape (h, w, 3)
        :returns: boxes, scores, classes
        """

        original_shape = img.shape[:2]

        img = img.astype(np.float32) / 255.
        img = cv2.resize(img, self.HYPERPARAMS["img_size"])
        img = np.expand_dims(img, axis=0)

        return self.sess.run(
            [self.bounding_boxes, self.scores, self.classes],
            feed_dict={
                self.yolo.input: img,
                self.img_size_tensor: original_shape,
                K.learning_phase(): 0
            }
        )
