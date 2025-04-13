from tensorflow.keras.applications import MobileNetV2  # Lighter than VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np

class FeatureExtractor:
    def __init__(self):
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                               pooling='avg', input_shape=(224, 224, 3))
        self.model = Model(inputs=base_model.input, outputs=base_model.output)
        self.model._make_predict_function()  # For thread safety

    def extract(self, img):
        img = img.resize((224, 224)).convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 127.5 - 1.0  # MobileNetV2 preprocessing
        feature = self.model.predict(x, verbose=0)[0]
        return feature / np.linalg.norm(feature)
