import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ModelUpdater:
    def __init__(self, weights_path):
        self.weights_path = weights_path
    
    def update_model(self, model, reward, train_images, train_labels, val_images, val_labels, lr=0.001, max_epochs=10):
        datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                              shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
        if reward >= 0.5: # If reward is greater than or equal to 0.5, save the weights
            model.save_weights(self.weights_path)
        
        # Update learning rate based on reward
        if reward >= 0.8:
            lr *= 1.5
        elif reward >= 0.6:
            lr *= 1.2
        elif reward <= 0.3:
            lr *= 0.5
        elif reward <= 0.1:
            lr *= 0.1
        
        # Compile model with new learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # Fit the model with a maximum number of epochs
        model.fit(datagen.flow(train_images, train_labels, batch_size=32), 
                  steps_per_epoch=len(train_images) // 32, epochs=max_epochs, 
                  validation_data=(val_images, val_labels))
        
        return model, lr
