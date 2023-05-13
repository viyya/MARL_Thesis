from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

class DataPreprocessor:
    def __init__(self, img_data, labels):
        self.img_data = img_data
        self.labels = labels

    def preprocess_data(self):
        # Perform label encoding
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(self.labels)

        # Split data into training and validation sets
        train_images, val_images, train_labels, val_labels = train_test_split(
            self.img_data, labels_encoded, test_size=0.2, random_state=42
        )

        # Normalize pixel values to [0, 1]
        train_images = train_images / 255.0
        val_images = val_images / 255.0

        # Reshape the input data to match the expected input shape of the model
        train_images = tf.reshape(train_images, (-1, 128, 128, 1))
        val_images = tf.reshape(val_images, (-1, 128, 128, 1))

        # Convert target labels to one-hot encoded vectors
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=160)
        val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=160)

        return train_images, val_images, train_labels, val_labels
