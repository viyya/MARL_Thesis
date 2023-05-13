from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SpeakerAgent:
    def __init__(self, train_images, train_labels, val_images, val_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.val_images = val_images
        self.val_labels = val_labels

    def create_model(self):
        # Create data augmentation pipeline
        datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                                     shear_range=0.1, zoom_range=0.1, horizontal_flip=True)

        # Define the CNN model
        model = Sequential()

        # Add Convolutional Layers
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        # Add two more Convolutional Layers
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        # Add Dropout layer
        model.add(Dropout(0.25))

        # Add Fully Connected Layers
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(160, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(datagen.flow(self.train_images, self.train_labels, batch_size=32),
                  steps_per_epoch=len(self.train_images) // 32, epochs=30,
                  validation_data=(self.val_images, self.val_labels))

        return model


