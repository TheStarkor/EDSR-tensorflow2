from tensorflow.keras.layers import Add, Conv2D, Input, Lambda, Activation
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def SubpixelConv2D(scale, **kwargs):
    return Lambda(lambda x: tf.nn.depth_to_space(x, scale), **kwargs)


def Normalization(rgb_mean=DIV2K_RGB_MEAN, **kwargs):
    return Lambda(lambda x: (x - rgb_mean) / 127.5, **kwargs)


def Denormalization(rgb_mean=DIV2K_RGB_MEAN, **kwargs):
    return Lambda(lambda x: x * 127.5 + rgb_mean, **kwargs)


def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))

    x = b = Conv2D(num_filters, 3, padding="same")(x_in)
    for _ in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding="same")(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding="same")(x)

    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding="same")(x_in)
    x = Activation("relu")(x)
    x = Conv2D(filters, 3, padding="same")(x)
    x = Add()([x_in, x])
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding="same", **kwargs)(x)
        return SubpixelConv2D(factor)(x)

    if scale == 2:
        x = upsample_1(x, 2, name="conv2d_1_scale_2")
    elif scale == 3:
        x = upsample_1(x, 3, name="conv2d_1_scale_3")
    elif scale == 4:
        x = upsample_1(x, 2, name="conv2d_1_scale_2")
        x = upsample_1(x, 2, name="conv2d_2_scale_2")

    return x
