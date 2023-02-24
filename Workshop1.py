import tensorflow as tf
from tensoerflow import keras
import numpy as np

import matplotlib.pyplot as plt

# Download data
(training_images, training_labels), (test_images,
                                     test_labels) = keras.datasets.fashion_mnist.load_data()
# Classify each label with an article of clothing
data_names = ["t-shirt", "pants", "pullover", "dress",
              "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# Normalize data
training_images = keras.utils.normalize(training_images, axis=1)
test_images = keras.utils.normalize(test_images, axis=1)

# Display a sample (not necessary for training)
SAMPLE_INDEX = 1203
print("Sample %d in training dataset:" % SAMPLE_INDEX)
plt.imshow(training_images[SAMPLE_INDEX])
plt.show()
print("Sample label: %s" % data_names[training_labels[SAMPLE_INDEX]])

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape))
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(units=256, activation='relu'))
# Output layer
model.add(keras.layers.Dense(10, activation='softmax'))
# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Training
model.fit(training_images, training_labels, epochs=5)

# Evaluate accuracy
eval_loss, eval_accuracy = model.evaluate(test_images, test_labels)
print("Test loss/cost: %.5f" % eval_loss)
print("Test accuracy: %.5f" % eval_accuracy)

predictions = model.predict(test_images)

INDEX = 50
# Display prediction vs actual
print("Sample %d in testing dataset:" % INDEX)
plt.imshow(test_images[INDEX])
plt.show()
print("Prediction: %s\nActual: %s\n" % (data_names[np.argmax(predictions[INDEX])], data_names[test_labels[INDEX]]))

# Probability bar graph
plt.title('Model Predictions')
plt.xticks(np.arange(10), data_names, rotation=90)
plt.yticks(np.arange(0, 1.1, .1))
plt.ylim(0, 1)
plot_bar = plt.bar(range(10), predictions[INDEX])
plot_bar[np.argmax(predictions[INDEX])].set_color('red')
plot_bar[test_labels[INDEX]].set_color('green')