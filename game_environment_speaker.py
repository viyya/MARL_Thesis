import pandas as pd
import cv2
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('averaged_perceptron_tagger')

class GameEnvironmentSpeaker:
    def __init__(self, file_path, test_img_dir, loaded_model, label_encoder):
        self.file_path = file_path
        self.test_img_dir = test_img_dir
        self.loaded_model = loaded_model
        self.label_encoder = label_encoder

    def preprocess_data(self):
        # Load the test data
        test_data = pd.read_excel(self.file_path)
        test_img = test_data['Filepath']
        target_label = test_data['shapes']
        true_labels = target_label.tolist()

        return test_img, target_label, true_labels

    def play_game(self, test_img, target_label):
        speaker_reward = 0
        sin_plu_distinct_reward = 0
        predicted_labels = []

        for img, tar_label in zip(test_img, target_label):
            test_image_path = self.test_img_dir + img
            test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
            test_image = cv2.resize(test_image, (128, 128)) # Resize to match the model's input shape
            test_image = np.expand_dims(test_image, axis=0) # Add batch dimension
            test_image = test_image / 255.0 # Normalize pixel values

            # Make prediction
            predicted_probs = self.loaded_model.predict(test_image) # Predict probabilities for all classes
            predicted_label = np.argmax(predicted_probs) # Get the class label with highest probability
            predicted_label = self.label_encoder.inverse_transform([predicted_label])[0] # Convert label index back to original label
            predicted_labels.append(predicted_label)
            if predicted_label == tar_label:
                speaker_reward += 1
            else:
                speaker_reward += 0

            # Use nltk to tag the word's part of speech
            tagged_word = nltk.pos_tag([predicted_label])
            tagged_word1 = nltk.pos_tag([tar_label])

            # Check if the word is singular or plural based on its POS tag
            if tagged_word[0][1] == 'NNS' and tagged_word1[0][1] == 'NNS':
                sin_plu_distinct_reward += 1
            elif tagged_word[0][1] == 'NN' and tagged_word1[0][1] == 'NN':
                sin_plu_distinct_reward += 1
            else:
                sin_plu_distinct_reward += 0

        return predicted_labels, speaker_reward, sin_plu_distinct_reward


