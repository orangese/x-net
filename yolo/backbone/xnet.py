"""

"yolo/backbone/xnet.py"

X-Net backbone.

"""

import keras

from yolo.backbone.utils import conv_bn_leaky, conv_gate, residual_blocks, residual_transpose_blocks


# ---------------- MAKING X-NET ----------------
def xnet_body(x):
    """X-Net body

    Description of structure: see paper

    :param x: symbolic tensor input
    :returns: symbolic tensor after applying x-net body

    """

    # HORIZONTAL LAYERS

    # stage 1: (n, n, 3) -> (n, n, 32)
    x = conv_bn_leaky(x, "stage_1", 32, (3, 3))
    # stage 2: (n, n, 32) -> (n / 2, n / 2, 64)
    x = residual_blocks(x, 64, 1, identifier="stage_2")
    # stage 3: (n / 2, n / 2, 64) -> (n / 4, n / 4, 128)
    x = residual_blocks(x, 128, 2, identifier="stage_3")
    # stage 4: (n / 4, n / 4, 128) -> (n / 8, n / 8, 256)
    x_branch_8 = residual_blocks(x, 256, 8, identifier="stage_4")
    # stage 5: (n / 8, n / 8, 256) -> (n / 16, n / 16, 512)
    x_branch_16 = residual_blocks(x_branch_8, 512, 8, identifier="stage_5")
    # stage 5: (n / 16, n / 16, 256) -> (n / 32, n / 32, 1024)
    x_branch_32 = residual_blocks(x_branch_16, 1024, 4, identifier="stage_6")


    # VERTICAL LAYERS

    # third branch: (n / 8, n / 8, 256)
    branch_32 = residual_transpose_blocks(x_branch_32, 512, 1, identifier="branch_32_512")
    branch_32 = residual_transpose_blocks(branch_32, 256, 1, identifier="branch_32_256")
    branch_32 = residual_transpose_blocks(branch_32, 128, 1, identifier="branch_32_128")

    # second branch: (n / 8, n / 8, 256)
    branch_16 = residual_transpose_blocks(x_branch_16, 256, 1, identifier="branch_16_256")
    branch_16 = residual_transpose_blocks(branch_16, 128, 1, identifier="branch_16_128")
    branch_16 = keras.layers.Add(name="branch_16_add")([branch_16, branch_32])

    # first branch: (n / 8, n / 8, 256)
    branch_8 = residual_transpose_blocks(x_branch_8, 128, 1, identifier="branch_8_128")
    branch_8 = keras.layers.Add(name="branch_8_add")([branch_8, branch_16, branch_32])

    # branch-off: (n / 8, n / 8, 1024)
    branch = keras.layers.Concatenate(name="final_branch_concat")([branch_8, branch_16, branch_32])
    branch = keras.layers.ZeroPadding2D(((1, 0), (1, 0)), name="final_branch_1_zero_padding")(branch)
    branch = conv_bn_leaky(branch, "final_branch_1", 512, (3, 3), strides=(2, 2))
    branch = keras.layers.ZeroPadding2D(((1, 0), (1, 0)), name="final_branch_2_zero_padding")(branch)
    branch = conv_bn_leaky(branch, "final_branch_2", 512, (3, 3), strides=(2, 2))

    branch = residual_blocks(branch, 1024, 1, identifier="final_branch_3_1024")

    # output: (n / 8, n / 8, 1024)
    # gated_output = conv_gate(x_branch_32, branch, kernel_regularizer=keras.regularizers.l2(5e-4))
    gated_output = keras.layers.Add()([x_branch_32, branch])

    return gated_output
