# Keras imports
from keras.models import *
from keras.layers import *
from keras import models
from keras.applications.vgg16 import VGG16

IMAGE_ORDERING = 'channels_last' 

# Paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
# Original caffe code: https://github.com/shelhamer/fcn.berkeleyvision.org


from keras import backend as K
from keras.layers import Layer
import keras.backend

class DePool2D(UpSampling2D):


    def __init__(self, pool2d_layer , *args, **kwargs):
        self._pool2d_layer = pool2d_layer
        super(DePool2D, self).__init__(*args, **kwargs)

    def get_output(self, train=False)  :
        X = self.get_input(train)
        if self.dim_ordering == 'th':
            output = repeat_elements(X, self.size[0], axis=2)
            output = repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = repeat_elements(X, self.size[0], axis=1)
            output = repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        return gradients(
            sum(
                self._pool2d_layer.get_output(train)
            ),
            self._pool2d_layer.get_input(train)
        ) * output


def build_segnet(img_shape=(416, 608, 3), nclasses=8, l2_reg=0.,
               init='glorot_uniform', path_weights=None,
               load_pretrained=False, freeze_layers_from=None):


	kernel = 3
	filter_size = 64
	pad = 1
	pool_size = 2

	model = Sequential()
	model.add(Layer(input_shape=img_shape))

	# encoder
	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Conv2D(filter_size, (kernel, kernel), padding='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Conv2D(128, (kernel, kernel), padding='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Conv2D(256, (kernel, kernel), padding='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Conv2D(512, (kernel, kernel), padding='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))


	# decoder
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Conv2D(512, (kernel, kernel), padding='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(pool_size,pool_size)))
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Conv2D(256, (kernel, kernel), padding='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(pool_size,pool_size)))
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Conv2D(128, (kernel, kernel), padding='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(pool_size,pool_size)))
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Conv2D(filter_size, (kernel, kernel), padding='valid'))
	model.add( BatchNormalization())


	model.add(Conv2D( nclasses , (1, 1), padding='valid',))

	model.outputHeight = model.output_shape[1]
	model.outputWidth = model.output_shape[2]
	print (model.output_shape )


	model.add(Reshape(( nclasses ,  model.output_shape[1]*model.output_shape[2]   ), input_shape=( nclasses , model.output_shape[1], model.output_shape[2]  )))
	
	model.add(Permute((2, 1)))
	model.add(Activation('softmax'))

    # Freeze some layers
    	if freeze_layers_from is not None:
        	freeze_layers(model, freeze_layers_from)

    	return model

        
        
        

    
def build_segnet_PI(img_shape=(416, 608, 3), nclasses=8, l2_reg=0.,
               ker_init='glorot_uniform',  freeze_layers_from=None, indices = True):

    input_tensor = Input(shape=img_shape) # type: object
    encoder = VGG16(
        include_top=False, 
        weights='imagenet', 
        input_tensor=input_tensor,
        input_shape=img_shape,
        pooling="valid" ) # type: tModel
    #encoder.summary()

    L = [layer for i, layer in enumerate(encoder.layers) ] # type: List[Layer]
    #for layer in L: layer.trainable = False # freeze VGG16
    L.reverse()

    x = encoder.output
    x = Dropout(0.5)(x)
    # Block 5
    if indices: x = DePool2D(L[0], size=L[0].pool_size, input_shape=encoder.output_shape[1:])(x)
    else:       x = UpSampling2D(  size=L[0].pool_size, input_shape=encoder.output_shape[1:])(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[1].filters, L[1].kernel_size, padding=L[1].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[2].filters, L[2].kernel_size, padding=L[2].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[3].filters, L[3].kernel_size, padding=L[3].padding, kernel_initializer=ker_init)(x)))
    x = Dropout(0.5)(x)
    # Block 4
    if indices: x = DePool2D(L[4], size=L[4].pool_size)(x)
    else:       x = UpSampling2D(  size=L[4].pool_size)(x)
    x = ZeroPadding2D(padding=((1, 0),(0,0)))(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[5].filters, L[5].kernel_size, padding=L[5].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[6].filters, L[6].kernel_size, padding=L[6].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[7].filters, L[7].kernel_size, padding=L[7].padding, kernel_initializer=ker_init)(x)))
    x = Dropout(0.5)(x)
    # Block 3
    if indices: x = DePool2D(L[8], size=L[8].pool_size)(x)
    else:       x = UpSampling2D(  size=L[8].pool_size)(x)

    x = Activation('relu')(BatchNormalization()(Conv2D(L[10].filters, L[10].kernel_size, padding=L[10].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[11].filters, L[11].kernel_size, padding=L[11].padding, kernel_initializer=ker_init)(x)))
    x = Dropout(0.5)(x)
    # Block 2
    if indices: x = DePool2D(L[12], size=L[12].pool_size)(x)
    else:       x = UpSampling2D(   size=L[12].pool_size)(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[13].filters, L[13].kernel_size, padding=L[13].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[14].filters, L[14].kernel_size, padding=L[14].padding, kernel_initializer=ker_init)(x)))
    # Block 1
    if indices: x = DePool2D(L[15], size=L[15].pool_size)(x)
    else:       x = UpSampling2D(   size=L[15].pool_size)(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[16].filters, L[16].kernel_size, padding=L[16].padding, kernel_initializer=ker_init)(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[17].filters, L[17].kernel_size, padding=L[17].padding, kernel_initializer=ker_init)(x)))

    x = Conv2D(nclasses, (1, 1), padding='valid', kernel_initializer=ker_init)(x)
    model = Model(inputs=encoder.inputs, outputs=x)
    model.summary()
    x = Reshape((img_shape[0] * img_shape[1],nclasses ), input_shape=( img_shape[0], img_shape[1],nclasses))(x)
    
    
    x = Activation('softmax')(x)
    
    predictions = x

    model = Model(inputs=encoder.inputs, outputs=predictions) # type: tModel

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
