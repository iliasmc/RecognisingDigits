import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from PIL import Image, ImageOps

# Image location of the image we want to predict
# When editting image, a 3px black marker in paint 3D is ideal, drawn in the center
img_location = ""

# Bool value to determine whether to display from self.test_images or the above image
show_testing = False

# Bool value to select whether to display only the above image or if to show 25 examples of
display_only_image = True


class Digits(object):
    """
    Class that builds a CNN capable of identifying handritten images, trained
    on the MNIST database
    """

    # Neural Network Architecture
    def setup(self):
        """
        Setting up the testing and training data
        """
        # Download images of handwritten digits and their labels
        handwritten_digits = tf.keras.datasets.mnist
        (self.train_images, self.train_labels), [self.test_images, self.test_labels] = handwritten_digits.load_data()

        # Normalising all values to be between 0 and 1
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

        # Showing the 25 first digits (if self.show_testing = False)
        if not display_only_image:
            plt.figure(figsize=(10, 10))
            for i in range(25):
                plt.subplot(5, 5, i + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(self.train_images[i], cmap=plt.cm.binary)
                plt.xlabel(self.train_labels[i])
            plt.show()

    def model(self):
        """
        Building the neural network
        """
        # Building the CNN
        self.model = tf.keras.Sequential([
            # Transforms two dimensional pixel image to one dimensional array
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            # Fully connected layer with 128 nodes
            tf.keras.layers.Dense(128, activation='sigmoid'),
            # 10 node output layer
            tf.keras.layers.Dense(10)
        ])

        # Compiling the model
        self.model.compile(
            # Optimizer -> How the model is updated
            optimizer='adam',
            # Loss -> Measures how accurate the model is in training
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            # Metrics -> Monitors the training and testing steps
            metrics=['accuracy']
        )

        # Feeding the training data to the model
        self.model.fit(self.train_images, self.train_labels, epochs=10)

    def predict(self):
        # Use softmax activation to make a model for predictions
        self.probability_model = tf.keras.Sequential([
            self.model,
            tf.keras.layers.Softmax()
        ])
        # "self.predictions[i]" predicts what digit i is
        self.predictions = self.probability_model.predict(self.test_images)


class Plotting(Digits):
    """
    Class that uses the model of the parent class to predict and plot guesses
    from super().test_images and other images
    """

    # Displaying predictions
    @staticmethod
    def plot_image(predictions_array, img):
        # Generating an image and the model's predictions
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        plt.xlabel("({}) {:2.0f}%".format(predicted_label, 100 * np.max(predictions_array)), color="blue")

    @staticmethod
    def plot_output(predictions_array):
        # Drawing the value array to show the result of the output layer
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        for i in range(10):
            thisplot[i].set_color("cyan")
        thisplot[predicted_label].set_color("blue")

    def plot_prediction(self):
        # bool value set to display either testing images (if set to True) or another image (if set to False)
        global show_testing

        # Testing 4 images [i] from self.test_images to see if the model evaluates a correct output
        if show_testing:
            i = randint(0, len(self.test_images))
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            self.plot_image(self.predictions[i], self.test_images[i])
            plt.subplot(1, 2, 2)
            self.plot_output(self.predictions[i])
            plt.tight_layout()
            plt.show()
        else:
            # Fetching image, prepossessing it, predicting it
            global img_location
            img = Image.open(img_location).convert('L').resize((28, 28), Image.ANTIALIAS)
            img = ImageOps.exif_transpose(img)
            img = ImageOps.invert(img)
            img = np.array(img)
            prediction = self.probability_model.predict(img[None, :, :])

            # Displaying prediction
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            self.plot_image(prediction[0], img)
            plt.title("Image")
            plt.subplot(1, 2, 2)
            self.plot_output(prediction[0])
            plt.title("Predictions")
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    run = Plotting()
    run.setup()
    run.model()
    run.predict()

    while True:
        run.plot_prediction()
