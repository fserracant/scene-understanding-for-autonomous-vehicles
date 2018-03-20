# Keras imports
from keras.models import *
from keras.layers import *

IMAGE_ORDERING = 'channels_last' 

# Paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
# Original caffe code: https://github.com/shelhamer/fcn.berkeleyvision.org


def build_segnet(img_shape=(416, 608, 3), nclasses=8, l2_reg=0.,
               init='glorot_uniform', path_weights=None,
               load_pretrained=False, freeze_layers_from=None):


	kernel = 3
	filter_size = 64
	pad = 1
	pool_size = 2

	model = models.Sequential()
	model.add(Layer(input_shape=(3, input_height , input_width )))

	# encoder
	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Convolution2D(128, kernel, kernel, border_mode='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Convolution2D(256, kernel, kernel, border_mode='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Convolution2D(512, kernel, kernel, border_mode='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))


	# decoder
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Convolution2D(512, kernel, kernel, border_mode='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(pool_size,pool_size)))
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Convolution2D(256, kernel, kernel, border_mode='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(pool_size,pool_size)))
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Convolution2D(128, kernel, kernel, border_mode='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(pool_size,pool_size)))
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
	model.add( BatchNormalization())


	model.add(Convolution2D( nClasses , 1, 1, border_mode='valid',))

	model.outputHeight = model.output_shape[-2]
	model.outputWidth = model.output_shape[-1]


	model.add(Reshape(( nClasses ,  model.output_shape[-2]*model.output_shape[-1]   ), input_shape=( nClasses , model.output_shape[-2], model.output_shape[-1]  )))
	
	model.add(Permute((2, 1)))
	model.add(Activation('softmax'))

    # Freeze some layers
    if freeze_layers_from is not None:
        freeze_layers(model, freeze_layers_from)

    return model



# Freeze layers for finetunning
def freeze_layers(model, freeze_layers_from):
    # Freeze the VGG part only
    if freeze_layers_from == 'base_model':
        print ('   Freezing base model layers')
        freeze_layers_from = 23

    # Show layers (Debug pruposes)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print ('   Freezing from layer 0 to ' + str(freeze_layers_from))

    # Freeze layers
    for layer in model.layers[:freeze_layers_from]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_from:]:
        layer.trainable = True


if __name__ == '__main__':
    input_shape = [224, 224, 3]
    print (' > Building')
    model = build_fcn8(input_shape, 11, 0.)
    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()