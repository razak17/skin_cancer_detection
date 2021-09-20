# Tensorflow v.2.3.1

from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
)
import tensorflow as tf
from tensorflow import keras
import typing


@tf.function
def block(X: tf.Tensor,
          kernel_size: int,
          filters: typing.List[int],
          stage_no: int,
          block_name: str,
          is_conv_layer: bool = False,
          stride: int = 2) -> tf.Tensor:
    """
    Block for residual network.

    Arguments:
    X             -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size   -- integer, specifying the shape of the middle CONV's window for the main path
    filters       -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage_no      -- integer, used to name the layers, depending on their position in the network
    block_name    -- string/character, used to name the layers, depending on their position in the network
    is_conv_layer -- to identiy if identity downsample is needed
    stride        -- integer specifying the stride to be used
    
    Returns:
    X             -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # names
    conv_name_base = "res" + str(stage_no) + block_name + "_branch"
    bn_name_base = "bn" + str(stage_no) + block_name + "_branch"

    # filters
    F1, F2, F3 = filters

    # save the input value for shortcut.
    X_shortcut = X

    #  First component
    # NOTE: if conv_layer, you need to do downsampling
    X = Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        strides=(stride, stride) if is_conv_layer else (1, 1),
        padding="valid",
        name=conv_name_base + "2a",
        kernel_initializer="glorot_uniform",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    # Second component
    X = Conv2D(
        filters=F2,
        kernel_size=(kernel_size, kernel_size),
        strides=(1, 1),
        padding="same",
        name=conv_name_base + "2b",
        kernel_initializer="glorot_uniform",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    # Third component
    X = Conv2D(
        filters=F3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=conv_name_base + "2c",
        kernel_initializer="glorot_uniform",
    )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    # NOTE: if is_conv_layer, you need to do downsampling the X_shortcut to match the output (X) channel
    #       so it can be added together
    if is_conv_layer:
        X_shortcut = Conv2D(
            filters=F3,
            kernel_size=(1, 1),
            strides=(stride, stride),
            padding="valid",
            name=conv_name_base + "1",
            kernel_initializer="glorot_uniform",
        )(X_shortcut)
        X_shortcut = BatchNormalization(axis=3,
                                        name=bn_name_base + "1")(X_shortcut)

    # Shortcut value
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


# Create a convolution block
def conv_block(input, kernel_size, filters, stage, block, strides=(2, 2)):
    # Variables
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Create block layers
    output = keras.layers.Conv2D(filters1, (1, 1),
                                 strides=strides,
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '2a')(input)
    output = keras.layers.BatchNormalization(axis=3,
                                             name=bn_name_base + '2a')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters2,
                                 kernel_size,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '2b')(output)
    output = keras.layers.BatchNormalization(axis=3,
                                             name=bn_name_base + '2b')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters3, (1, 1),
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '2c')(output)
    output = keras.layers.BatchNormalization(axis=3,
                                             name=bn_name_base + '2c')(output)
    shortcut = keras.layers.Conv2D(filters3, (1, 1),
                                   strides=strides,
                                   kernel_initializer='he_normal',
                                   name=conv_name_base + '1')(input)
    shortcut = keras.layers.BatchNormalization(axis=3, name=bn_name_base +
                                               '1')(shortcut)
    output = keras.layers.add([output, shortcut])
    output = keras.layers.Activation('relu')(output)
    # Return a block
    return output


# Create an identity block
def identity_block(input, kernel_size, filters, stage, block):

    # Variables
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Create layers
    output = keras.layers.Conv2D(filters1, (1, 1),
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '2a')(input)
    output = keras.layers.BatchNormalization(axis=3,
                                             name=bn_name_base + '2a')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters2,
                                 kernel_size,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '2b')(output)
    output = keras.layers.BatchNormalization(axis=3,
                                             name=bn_name_base + '2b')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters3, (1, 1),
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '2c')(output)
    output = keras.layers.BatchNormalization(axis=3,
                                             name=bn_name_base + '2c')(output)
    output = keras.layers.add([output, input])
    output = keras.layers.Activation('relu')(output)
    # Return a block
    return output