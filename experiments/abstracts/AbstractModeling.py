from abc import ABC, abstractmethod

class AbstractModeling(ABC):
    def __init__(self, model_params=None):
        """
        Initialize the modeling with optional parameters.
        
        :param model_params: A dictionary of model parameters.
        """
        self.model_params = model_params if model_params else {}
        self.model = None  # This will store the defined model

    @abstractmethod
    def build_model(self):
        """
        Define and build the model architecture.
        """
        raise NotImplementedError()

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the provided training data.
        
        :param X: Feature matrix for training.
        :param y: Labels corresponding to the training data.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        :param X: Feature matrix for prediction.
        :return: Predictions (e.g., as an array or list).
        """
        raise NotImplementedError()

    def evaluate(self, X, y, metric):
        """
        Evaluate the model using a specified metric.
        
        :param X: Feature matrix for evaluation.
        :param y: True labels for evaluation.
        :param metric: A function that computes the evaluation metric (e.g., accuracy).
        :return: The computed metric value.
        """
        predictions = self.predict(X)
        return metric(y, predictions)