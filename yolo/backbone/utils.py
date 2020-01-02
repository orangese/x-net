"""

"yolo/backbone/utils.py"

Backbone (darknet or x-net) layers and tools.

"""

import functools

import keras
from keras import backend as K


# ---------------- SETUP ----------------

# CONSTANTS
CONFIGS = {
    "darknet_conv": {
        "kernel_regularizer": keras.regularizers.l2(5e-4),
        "padding": "same"
    },
    "darknet_conv_transpose": {
        "kernel_regularizer": keras.regularizers.l2(5e-4),
        "padding": "valid"
    },
    "conv_bn_leaky": {
        "use_bias": False
    }

}


# RETRIEVER
def get_config(key):
    """Retrieves an editable CONFIG dictionary

    :param key: config key
    :returns: copy of CONFIG[model]
    :raise: KeyError if key is not found

    """

    try:
        return CONFIGS[key].copy()
    except KeyError:
        raise KeyError("{} not found. Available configs are {}".format(key, list(CONFIGS.keys())))


# DECORATORS
def add_config(config_name, wraps=None):
    """Decorator that adds custom configuration to layer

    :param config_name: name of config dictionary (contained in CONFIGS)
    :param wraps: function to wrap

    """

    def _add_config(func):
        @functools.wraps(func if wraps is None else wraps)
        def _func(*args, **kwargs):
            config = get_config(config_name)
            config.update(kwargs)
            return func(*args, **config)

        return _func

    return _add_config


# ---------------- WRAPPER LAYERS ----------------
@add_config("darknet_conv", wraps=keras.layers.Conv2D)
def DarknetConv2D(identifier, *args, **kwargs):
    """Conv2D class wrapper with Darknet parameters. See CONFIGS for more information

    :param identifier: identifier for layer
    :param args: additional layer configuration
    :param kwargs: additional named layer configuration
    :returns: wrapper for keras.layers.Conv2D (not symbolic tensor)

    """

    return keras.layers.Conv2D(*args, name="{}_conv2d".format(identifier), **kwargs)


@add_config("darknet_conv_transpose", wraps=keras.layers.Conv2DTranspose)
def DarknetConv2DTranspose(identifier, *args, **kwargs):
    """Conv2DTranspose class wrapper with Darknet parameters. See CONFIGS for more information

    :param identifier: identifier for layer
    :param args: additional layer configuration
    :param kwargs: additional named layer configuration
    :returns: wrapper for keras.layers.Conv2DTranspose (not symbolic tensor)

    """

    return keras.layers.Conv2DTranspose(*args, name="{}_conv2d_transpose".format(identifier), **kwargs)


# ---------------- FUNCTIONAL-STYLE LAYERS ----------------
def conv_bn_leaky(x, identifier, *args, **kwargs):
    """Darknet convolutional (without bias) + batch normalization + leaky ReLU

    :param x: symbolic tensor input
    :param identifier: identifier for layer name
    :param args: additional layer configuration
    :param kwargs: additional named layer configuration
    :returns: symbolic tensor output after applying convolutional + batch norm + leaky

    """

    @add_config("conv_bn_leaky", wraps=DarknetConv2D)
    def LeakyConv2D(identifier, *args, **kwargs):
        return DarknetConv2D(identifier, *args, **kwargs)

    x = LeakyConv2D(identifier, *args, **kwargs)(x)
    x = keras.layers.normalization.BatchNormalization(name="{}_bn".format(identifier))(x)
    x = keras.layers.advanced_activations.LeakyReLU(0.1, name="{}_leaky".format(identifier))(x)

    return x


def conv_transpose_bn_leaky(x, identifier, *args, **kwargs):
    """Darknet transposed convolutional (without bias) + batch normalization + leaky ReLU

    :param identifier: identifier for layer
    :param x: symbolic tensor input
    :param args: additional layer configuration
    :param kwargs: additional named layer configuration
    :returns: symbolic tensor output after applying convolutional + batch norm + leaky

    """

    @add_config("conv_bn_leaky", wraps=DarknetConv2DTranspose)
    def LeakyConv2DTranspose(identifier, *args, **kwargs):
        return DarknetConv2DTranspose(identifier, *args, **kwargs)

    x = LeakyConv2DTranspose(identifier, *args, **kwargs)(x)
    x = keras.layers.normalization.BatchNormalization(name="{}_bn_transpose".format(identifier))(x)
    x = keras.layers.advanced_activations.LeakyReLU(0.1, name="{}_leaky_transpose".format(identifier))(x)

    return x


def residual_blocks(x, filters, blocks, identifier):
    """Darknet residual blocks (includes non-residual convolutional layer + zero padding)

    1. Zero padding
    2. Darknet convolutional (no bias) + batch normalization + leaky ReLU
    3. Repeat `blocks` times:
        a. Darknet convolutional (no bias) + batch normalization + leaky ReLU
        b. Darknet convolutional (no bias) + batch normalization + leaky ReLU
        c. Add input to a and b

    :param x: symbolic tensor input
    :param filters: number of filters for conv_bn_leaky modules
    :param blocks: number of residual blocks to chain together
    :param identifier: id to append to beginning of residual layer names
    :returns: symbolic tensor output

    """

    def residual_block(x, filters, name=None):
        y = conv_bn_leaky(x, name.replace("res", "res_1x1"), filters // 2, (1, 1))
        y = conv_bn_leaky(y, name.replace("res", "res_3x3"), filters, (3, 3))

        return keras.layers.Add(name=name)([x, y])

    x = keras.layers.ZeroPadding2D(((1, 0), (1, 0)), name="{}_zero_padding".format(identifier))(x)

    name = "{}_3x3_{}_{}".format(identifier, blocks, filters)
    x = conv_bn_leaky(x, name, filters, (3, 3), strides=(2, 2), padding="valid")

    for block in range(blocks):
        name = "{}_res_add_{}_{}_{}".format(identifier, blocks, filters, block)
        x = residual_block(x, filters, name=name)

    return x


def residual_transpose_blocks(x, filters, blocks, identifier):
    """Darknet residual blocks (includes non-residual convolutional layer + zero padding)

    1. Darknet convolutional (no bias) + batch normalization + leaky ReLU
    2. Repeat `blocks` times:
        a. Darknet convolutional (no bias) + batch normalization + leaky ReLU
        b. Darknet convolutional (no bias, same padding) + batch normalization + leaky ReLU
        c. Add input to a and b

    :param x: symbolic tensor input
    :param filters: number of filters for conv_bn_leaky modules
    :param blocks: number of residual blocks to chain together
    :param identifier: id to append to beginning of residual layer names
    :returns: symbolic tensor output

    """

    def residual_transpose_block(x, filters, name=None):
        y = conv_transpose_bn_leaky(x, name.replace("res", "res_1x1_transpose"), filters // 2, (1, 1))
        y = conv_transpose_bn_leaky(y, name.replace("res", "res_3x3_transpose"), filters, (3, 3), padding="same")

        return keras.layers.Add(name=name)([x, y])

    name = "{}_3x3_{}_{}".format(identifier, blocks, filters)
    x = conv_transpose_bn_leaky(x, name, filters, (3, 3), strides=(2, 2), padding="same")

    for block in range(blocks):
        name = "transpose_{}_res_add_{}_{}_{}".format(identifier, blocks, filters, block)
        x = residual_transpose_block(x, filters, name=name)

    return x


def backbone_head(x, filters, out_filters, identifier):
    """Backbone head (same as Darknet head)

    1. Repeat three times:
         a. Darknet convolutional (no bias) + batch normalization + leaky ReLU
         b. Darknet convolutional (no bias) + batch normalization + leaky ReLU
    2. 3rd 1a, darknet convolutional on 3rd 1b (will be concatenated later)

    :param x: symbolic tensor input
    :param filters: number of filters for darknet convolutional layers
    :param out_filters: number of filters for last darknet convolutional layer
    :param identifier: layer name identifier
    :returns: symbolic tensor output after applying darknet head

    """

    def darknet_block(x, filters, identifier, return_both=False):
        x = conv_bn_leaky(x, "{}_1x1_head".format(identifier), filters, (1, 1))
        y = conv_bn_leaky(x, "{}_3x3_head".format(identifier), filters * 2, (3, 3))

        if return_both:
            return x, y
        return y

    x = darknet_block(x, filters, "{}_1".format(identifier))
    x = darknet_block(x, filters, "{}_2".format(identifier))

    x, y = darknet_block(x, filters, "{}_3".format(identifier), return_both=True)
    y = DarknetConv2D(identifier, out_filters, (1, 1))(y)

    return x, y


def conv_gate(a, b, *args, **kwargs):
    """Convolutional update gate (similar to LSTM)

    Equations:

    `out = u_a ⊙ a + u_b ⊙ b`,

    `u_a = sigmoid(w_aa * a + w_ab * b + b_a)`,

    `u_b = sigmoid(w_ba * a + w_bb * b + b_b)`

    ⊙ denotes the Hadamard product (element-wise multiplication)

    * denotes a 1 x 1 x k convolution, k = filters in a == filters in b

    :param a: first tensor
    :param b: second tensor (must have same shape as a)
    :param args: additional layer configuration info
    :param kwargs: additional named layer configuration info
    :returns: symbolic tensor after applying update gate
    :raise: AssertionError if kwargs has illegal arguments ('activation', 'use_bias')

    """

    assert "activation" not in kwargs or kwargs["activation"] is "linear", "activation cannot be overriden"
    assert "use_bias" not in kwargs or not kwargs["use_bias"], "use_bias cannot be overriden"

    def get_update_term(a, b, which="a"):
        def mul(x, use_bias, name):
            return keras.layers.Conv2D(K.int_shape(a)[-1], (1, 1), use_bias=use_bias, name=name, *args, **kwargs)(x)

        update_part_a = mul(a, use_bias=False, name="gate_{}a_conv".format(which))
        update_part_b = mul(b, use_bias=True, name="gate_{}b_conv".format(which))
        raw_update = keras.layers.Add(name="gate_{}_add".format(which))([update_part_a, update_part_b])
        update = keras.layers.Activation("sigmoid", name="gate_{}_sigmoid".format(which))(raw_update)

        return keras.layers.Multiply(name="gate_{}_mul".format(which))([update, a if which == "a" else b])

    # update gate for a
    update_a = get_update_term(a, b, which="a")
    # update gate for b
    update_b = get_update_term(a, b, which="b")

    # add the gates
    out = keras.layers.Add(name="gate_add")([update_a, update_b])

    return out
