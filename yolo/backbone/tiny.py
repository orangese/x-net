"""

"yolo/backbone/tiny.py"

Tiny YOLOv3 backbone.

"""

import keras

from yolo.backbone.utils import conv_bn_leaky


# ---------------- MAKING TINY MODEL ----------------
def tiny_body(x):
    """
    Tiny YOLOv3 body.

    1. Repeat 5 times:
        a. Darknet convolutional (no bias) + batch normalization + leaky ReLU
        b. Max pooling
    2.

    :param x: symbolic tensor input
    :returns: symbolic tensor output

    """

    def tiny_yolo_block(x, filters, identifier, return_both=False, **kwargs):
        config = {"pool_size": (2, 2), "strides": (2, 2), "padding": "same"}
        config.update(kwargs)

        x = conv_bn_leaky(x, "{}".format(identifier), filters, (3, 3))
        y = keras.layers.MaxPooling2D(**config, name="{}_max_pooling".format(identifier))(x)

        if return_both:
            return x, y
        return y

    block_1 = tiny_yolo_block(x, 16, "block_1_16")
    block_1 = tiny_yolo_block(block_1, 32, "block_1_32")
    block_1 = tiny_yolo_block(block_1, 64, "block_1_64")
    block_1 = tiny_yolo_block(block_1, 128, "block_1_128")
    block_1, block_2 = tiny_yolo_block(block_1, 256, "block_1_256", return_both=True)

    block_2 = tiny_yolo_block(block_2, 512, "block_2_512", strides=(1, 1))
    block_2 = conv_bn_leaky(block_2, "block_2_1024", 1024, (3, 3))
    block_2 = conv_bn_leaky(block_2, "block_2_256", 256, (1, 1))

    return block_1, block_2
