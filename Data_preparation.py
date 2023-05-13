import pandas as pd
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, csv_path, image_folder):
        self.csv_path = csv_path
        self.image_folder = image_folder

    def process_images(self):
        # Load the CSV file into a pandas DataFrame
        data = pd.read_csv(self.csv_path)

        # Extract the filenames and labels into separate arrays
        filenames = data['Filenames'].values
        labels = data['labels'].values

        # Set up a list to hold the image data
        images = []

        # Loop over the filenames and read each image into the images list
        for filename in filenames:
            # Use OpenCV to read the image file
            image = cv2.imread(self.image_folder + filename)

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize the image to a standard size
            resized = cv2.resize(gray, (128, 128))

            # Convert the image to a numerical format and add it to the images list
            images.append(np.array(resized).reshape(-1))

        # Convert the images list to a numpy array
        img = np.array(images)

        # Convert the labels to a numpy array
        labels = np.array(labels)

        return img, labels
