import cv2
import pickle

class Detection:
    def __init__(self):
        self.model = None
        self.labels = None

    def load_model(self, model_path):
        """
        Loads a trained model and labels from a saved file.

        Args:
        model_path: Path to the pickled model and labels file.

        Returns:
        None
        """

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            self.labels = pickle.load(f)

    def classify(self, image):
        """
        Classifies a new image using the loaded model.

        Args:
        image: A NumPy array representing the image.

        Returns:
        Predicted category of the image as an integer.
        """

        # Feature extraction with SURF
        surf = cv2.xfeatures2d.SURF.create()
        kp, des = surf.detectAndCompute(image, None)
        features = des

        # Normalize features
        features = normalize_features([features])

        # Prediction using loaded model
        prediction = self.model.predict(features)
        return int(prediction[0])