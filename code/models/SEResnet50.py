
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


# Keras imports
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D,Permute ,Conv2D, add, multiply

from keras.regularizers import l2
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


def squeeze_excite_block(input, ratio=16):

    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def build_SEResNet50(img_shape=(3, 224, 224), n_classes=1000, weight_decay =1e-4,
                 freeze_layers_from='base_model', SE = True):
    
    initial_conv_filters=64
    depth=[3, 4, 6, 3]
    filters=[64, 128, 256, 512]
    width=1
    bottleneck=False
    

    img_input = Input(shape=img_shape)
    
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    N = list(depth)

    # block 1 (initial conv block)
    x = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # block 2 (projection block)
    for i in range(N[0]):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[0], width, SE=SE)
        else:
            x = _resnet_block(x, filters[0], width, SE=SE)

    # block 3 - N
    for k in range(1, len(N)):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[k], width, strides=(2, 2), SE=SE)
        else:
            x = _resnet_block(x, filters[k], width, strides=(2, 2), SE=SE)

        for i in range(N[k] - 1):
            if bottleneck:
                x = _resnet_bottleneck_block(x, filters[k], width, SE=SE)
            else:
                x = _resnet_block(x, filters[k], width, SE=SE)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
              activation='softmax')(x)

            
    inputs = img_input
    
    # Create model.
    model = Model(inputs, x, name='seresnet50')

    return model




    
    
def _resnet_block(input, filters, k=1, strides=(1, 1), SE = False):
    ''' Adds a pre-activation resnet block without bottleneck layers
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)

    if strides != (1, 1) or init._keras_shape[channel_axis] != filters * k:
        init = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    if SE:
        x = squeeze_excite_block(x)

    m = add([x, init])
    return m


def _resnet_bottleneck_block(input, filters, k=1, strides=(1, 1), SE = False):
    ''' Adds a pre-activation resnet block with bottleneck layers
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    bottleneck_expand = 4

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)

    if strides != (1, 1) or init._keras_shape[channel_axis] != bottleneck_expand * filters * k:
        init = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    if SE:
        x = squeeze_excite_block(x)

    m = add([x, init])
    return m
