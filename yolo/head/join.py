"""

"utils/join.py"

Contains YOLO and Tiny YOLO factories.

"""

import keras

from yolo.backbone.utils import backbone_head, conv_bn_leaky, DarknetConv2D
from yolo.backbone.darknet import darknet_body
from yolo.backbone.tiny import tiny_body
from yolo.backbone.xnet import xnet_body


# ---------------- MODEL CREATION ----------------
def yolo(inputs, num_anchors, num_classes, backbone_type="x-net"):
    """Backbone + YOLO head

    :param inputs: inputs to YOLO as symbolic tensors
    :param num_anchors: number of anchors to be used
    :param num_classes: number of classes to be predicted by YOLO
    :param backbone_type: backbone, either 'x-net' or 'darknet' (default: 'x-net')
    :returns: YOLO as a keras Model, with three output layers
    :raise: ValueError if backbone isn't supported

    """

    def yolo_block(x, num_filters, identifier, concat_layer=None, upsample=True):
        if upsample:
            assert concat_layer is not None, "concat_layer must be provided in upsample mode"
            x = conv_bn_leaky(x, "{}_yolo_block".format(identifier), num_filters, (1, 1))
            x = keras.layers.UpSampling2D(2, name="{}_upsampling".format(identifier))(x)
            x = keras.layers.Concatenate(name="{}_concat".format(identifier))([x, concat_layer])
        x, y = backbone_head(x, num_filters, num_anchors * (num_classes + 5), identifier)

        return x, y

    if backbone_type == "x-net":
        backbone_body = xnet_body
    elif backbone_type == "darknet":
        backbone_body = darknet_body
    else:
        raise ValueError("only 'x-net' and 'darknet' supported as backbones")

    backbone = keras.Model(inputs, backbone_body(inputs), name="backbone")

    x, yolo_512 = yolo_block(backbone.output, 512, "yolo_512", upsample=False)
    x, yolo_256 = yolo_block(x, 256, "yolo_256", backbone.get_layer("stage_5_res_add_8_512_7").output)
    x, yolo_128 = yolo_block(x, 128, "yolo_128", backbone.get_layer("stage_4_res_add_8_256_7").output)

    yolo = keras.Model(inputs, [yolo_512, yolo_256, yolo_128], name="yolo")

    return yolo


def tiny_yolo(inputs, num_anchors, num_classes):
    """Tiny YOLO

    :param inputs: inputs to YOLO as symbolic tensors
    :param num_anchors: number of anchors to be used
    :param num_classes: number of classes to be predicted by YOLO
    :returns: YOLO as a keras Model, with two output layers

    """

    yolo_block_1, yolo_block_2 = tiny_body(inputs)

    yolo_512 = conv_bn_leaky(yolo_block_2, "yolo_512_1", 512, (3, 3))
    yolo_512 = DarknetConv2D("yolo_512_2", num_anchors * (num_classes + 5), (1, 1))(yolo_512)

    yolo_block_2 = conv_bn_leaky(yolo_block_2, "yolo_block_2", 128, (1, 1))
    yolo_block_2 = keras.layers.UpSampling2D(2, name="yolo_block_2_upsampling")(yolo_block_2)

    yolo_concat = keras.layers.Concatenate(name="yolo_concat")([yolo_block_2, yolo_block_1])
    yolo_concat = conv_bn_leaky(yolo_concat, "yolo_concat_1", 256, (3, 3))
    yolo_concat = DarknetConv2D("yolo_concat_2", num_anchors * (num_classes + 5), (1, 1))(yolo_concat)

    return keras.Model(inputs, [yolo_512, yolo_concat], name="tiny_yolo")
