from keras.applications.resnet50 import ResNet50
from keras.utils import plot_model

base_model = ResNet50(include_top=True, weights=None,
                      input_tensor=None, input_shape=(224, 224, 3))

plot_model(base_model, to_file='resnet50.png',
           show_shapes=True, show_layer_names=True)
