from abstracts.AbstractPredictor import AbstractPredictor
from fetch_ec_improved import fetch_ec_async
import pandas as pd

class DiamondPredictor(AbstractPredictor):
    def __init__(self, predict_path, model_params=None):
        super().__init__(model_params)
        self.predict_path = predict_path

    def build_model(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, feature_path):
        fetch_ec_async(feature_path, self.predict_path)
        return pd.read_csv(self.predict_path)

