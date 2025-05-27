from keras.models import model_from_json
from app.core.config import JSON_MODEL_PATH, WEIGHTS_MODEL_PATH

class EmotionModel:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        """Load the trained model from files"""
        with open(JSON_MODEL_PATH, 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(WEIGHTS_MODEL_PATH)
        return model

    def predict(self, features):
        """Make prediction using the loaded model"""
        return self.model.predict(features) 