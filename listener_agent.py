import cv2
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier

class ListenerAgent:
    def __init__(self, data_path, filenames, shapes):
        self.data_path = data_path
        self.filenames = filenames
        self.shapes = shapes

    def predict_label(self, test_image_path, predicted_label):
        # Get list of images with the same label as the predicted label
        similar_images = [img for img, label in zip(self.filenames, self.shapes) if label == predicted_label]

        # Load query image
        query_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        query_image = cv2.resize(query_image, (128, 128))

        # Compute HOG features for query image
        query_features = hog(query_image, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), feature_vector=True)

        # Reshape query_features to have a shape of (1, num_features)
        query_features = query_features.reshape(1, -1)

        # Compute HOG features for similar images
        feature_vectors = []
        labels = []
        for img_path, label in zip(self.filenames, self.shapes):
            if label == predicted_label:
                img = cv2.imread(self.data_path + img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (128, 128))
                hog_features = hog(img, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), feature_vector=True)
                feature_vectors.append(hog_features)
                labels.append(label)

        # Train a K-Nearest Neighbors classifier on the feature vectors
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(feature_vectors, labels)

        # Predict the label of the query image using the trained KNN classifier
        predicted_label_listener = knn.predict(query_features)

        # Get list of images with the predicted label
        predicted_images = [img for img, label in zip(self.filenames, self.shapes) if label == predicted_label_listener]

        # Load the first image in the list
        matched_image_path = predicted_images[0]
        image = cv2.imread(self.data_path + matched_image_path)

        return image, predicted_label_listener
