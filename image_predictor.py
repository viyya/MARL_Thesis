import cv2
import numpy as np
import pandas as pd

class ImagePredictor:
    def __init__(self, test_img_dir, loaded_model, label_encoder):
        self.test_img_dir = test_img_dir
        self.loaded_model = loaded_model
        self.label_encoder = label_encoder

    def predict_labels(self, test_img, target_label):
        predicted_labels = []
        # Load the test image
        for img, tar_label in zip(test_img, target_label):
            test_image_path = self.test_img_dir + img
            test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
            test_image = cv2.resize(test_image, (128, 128)) # Resize to match the model's input shape
            test_image = np.expand_dims(test_image, axis=0) # Add batch dimension
            test_image = test_image / 255.0 # Normalize pixel values

            # Make prediction
            predicted_probs = self.loaded_model.predict(test_image) # Predict probabilities for all classes
            predicted_label = np.argmax(predicted_probs) # Get the class label with the highest probability
            predicted_label = self.label_encoder.inverse_transform([predicted_label])[0] # Convert label index back to the original label
            predicted_labels.append(predicted_label)

        return predicted_labels

