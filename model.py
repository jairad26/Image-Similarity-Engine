
import numpy as np
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing import image

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

base_model = VGG16(weights='imagenet', include_top=False)

model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

VECTOR_DIMENSION = 2048


def extract_features(img):
    img = img.astype('float32')
    img = preprocess_input(img)
    features = model.predict(img[np.newaxis])
    return features.flatten()
    
features = []
for i in range(len(x_train)):
    features.append(extract_features(x_train[i]))

def get_image_vector(file):
    # Load image from file
    img = image.load_img(file, target_size=(224, 224))

    # Preprocess image for ResNet50
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Generate image features using ResNet50
    features = model.predict(x)

    # Return image features as a vector
    vector = np.reshape(features, (VECTOR_DIMENSION,))
    return vector


