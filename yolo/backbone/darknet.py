"""

"yolo/backbone/darknet.py"

Darknet backbone.

"""

from yolo.backbone.utils import conv_bn_leaky, residual_blocks


# ---------------- MAKING DARKNET ----------------
def darknet_body(x):
    """Darknet body

    1. Darknet convolutional (no bias) + batch normalization + leaky ReLU
    2. Darknet stacked residual block (5x)

    :param x: symbolic tensor input
    :returns: symbolic tensor after applying darknet body

    """

    x = conv_bn_leaky(x, "stage_1", 32, (3, 3))

    x = residual_blocks(x, 64, 1, identifier="stage_2")
    x = residual_blocks(x, 128, 2, identifier="stage_3")
    x = residual_blocks(x, 256, 8, identifier="stage_4")
    x = residual_blocks(x, 512, 8, identifier="stage_5")
    x = residual_blocks(x, 1024, 4, identifier="stage_6")

    return x
