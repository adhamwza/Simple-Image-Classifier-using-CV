#Imports
import tensorflow as tf
from PIL import ImageEnhance
from PIL.Image import Image
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import random
import numpy
import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from pathlib import Path
from tensorflow.python.keras import regularizers
import tensorflow as tf
import tensorflow as tf


# Check the number of available GPUs
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Check if TensorFlow is using GPU acceleration
print("TensorFlow is using GPU: ", tf.test.is_gpu_available())




# Load and preprocess the data
def load_and_preprocess_data(data_path):
    image_data_path = list(data_path.glob("**/*.jpg"))
    label_path = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], image_data_path))
    final_data = pd.DataFrame({"image_data": image_data_path, "label": label_path}).astype("str")
    final_data = final_data.sample(frac=1).reset_index(drop=True)
    return final_data


# Load and preprocess images
def load_and_preprocess_images(directory, num_samples=25):
    plt.figure(figsize=(15, 15))

    for i in range(num_samples):
        file = random.choice(os.listdir(directory))
        image_path = os.path.join(directory, file)

        # Load the image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)

        # Apply greyscaling
        img_array_gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

        # Apply noise reduction (Gaussian blur)
        img_array_blurred = cv.GaussianBlur(img_array_gray, (5, 5), 0)

        # Convert TensorFlow tensor to Python integer

        i_value = int(i)
        ax = plt.subplot(5, 5, i_value + 1)
        plt.imshow(img_array_blurred.astype("uint8"), cmap='gray')  # Display the grayscale image
        plt.axis("off")
    plt.show()

# Directory paths
food_images = r'\Users\adham\OneDrive\Desktop\Dataset3\Training\food'
building_images = r'\Users\adham\OneDrive\Desktop\Dataset3\Training\building'
landscape_images = r'\Users\adham\OneDrive\Desktop\Dataset3\Training\landscape'
people_images = r'\Users\adham\OneDrive\Desktop\Dataset3\Training\people'


# Load and preprocess images for each class
load_and_preprocess_images(food_images)
load_and_preprocess_images(building_images)
load_and_preprocess_images(landscape_images)
load_and_preprocess_images(people_images)


# Data paths
image_path = r'\Users\adham\OneDrive\Desktop\Dataset3\Training'
train_data_path = Path(image_path)
valid_data_path = Path(r'\Users\adham\OneDrive\Desktop\Dataset3\Validation')
test_data_path = Path(r'\Users\adham\OneDrive\Desktop\Dataset3\Testing')


# Load and preprocess data
final_train_data = load_and_preprocess_data(train_data_path)
final_valid_data = load_and_preprocess_data(valid_data_path)
final_test_data = load_and_preprocess_data(test_data_path)

label_name= ['Food', 'Landscape', 'Building', 'People']

image_size=(224,224)


class_names = os.listdir(image_path)
print(class_names)
print("Number of classes : {}".format(len(class_names)))


#Batch size
batch_size=128
#train data generator with image augmentation
traindata_generator= ImageDataGenerator(rescale=1./255,
                                        #zoom_range=0.2,
                                        #width_shift_range=0.2,
                                        #height_shift_range=0.2,
                                        #shear_range=0.2,
                                        #horizontal_flip=True,
                                        #validation_split=0.2,
                                        #rotation_range=20,  # Add rotation with a maximum angle of 20 degrees
                                        #brightness_range=[0.8, 1.2],  # Adjust brightness within the specified range
                                        #channel_shift_range=20,  # Randomly shift color channels
                                        #vertical_flip=True,  # Enable vertical flipping
                                        #fill_mode='nearest'
                                        )


#data generator for the validation and test data
validdata_generator=ImageDataGenerator(rescale=1./255)
testdata_generator=ImageDataGenerator(rescale=1./255)


train_data_generator=traindata_generator.flow_from_dataframe(dataframe=final_train_data,
                                                             x_col="image_data",
                                                             y_col="label",
                                                             batch_size=batch_size,
                                                             class_mode="categorical",
                                                             target_size=(224,224),
                                                             color_mode="rgb",
                                                             shuffle=True )

valid_data_generator=validdata_generator.flow_from_dataframe(dataframe=final_valid_data,
                                                             x_col="image_data",
                                                             y_col="label",
                                                             batch_size=batch_size,
                                                             class_mode="categorical",
                                                             target_size=(224,224),
                                                             color_mode="rgb",
                                                             shuffle=True )

test_data_generator=testdata_generator.flow_from_dataframe(dataframe=final_test_data,
                                                           x_col="image_data",
                                                           y_col="label",
                                                           batch_size=batch_size,
                                                           class_mode="categorical",
                                                           target_size=(224,224),
                                                           color_mode="rgb",
                                                           shuffle=False )

class_dict = train_data_generator.class_indices
class_list = list(class_dict.keys())
print(class_list)


#Model
#Contains 5 layers
#150 neurons each and goes down 30 at a time
#Has an L2 regularizer and a dropout in order
#to avoid overfitting
model = Sequential([
    # first layer:
    Flatten(input_shape=[224, 224, 3]),
    # middle layers/ hidden layers and the softmax output layer
    Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.001),
    Dense(120, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.001),
    Dense(90, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.001),
    Dense(60, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.001),
    Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.001),
    Dense(4, activation='softmax')
])
model.summary()


#custom_learning_rate = 0.1

#custom_optimizer = tf.keras.optimizers.Adamax(learning_rate=0.01)
custom_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.00006,
                                                                    decay_steps=5000,
                                                                    decay_rate=0.8)


custom_optimizer = tf.keras.optimizers.Adamax(learning_rate=custom_lr_schedule)

model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
              optimizer = custom_optimizer,
              metrics = ['accuracy']
              )


#epochs is how many times the model will train. in this case, 5 times, so he gets more accurate and loses less as the epochs go.
#model_history = model.fit(train_data_generator, epochs=12, validation_data=valid_data_generator )

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True)
model_history = model.fit(train_data_generator, epochs=32, validation_data=valid_data_generator, callbacks=[early_stopping])

#Evaluation and visualization
test_loss, test_accuracy = model.evaluate(test_data_generator)
print(f"Test Accuracy: {test_accuracy}")


prediction= model.predict(test_data_generator)
prediction=np.argmax(prediction,axis=1)
map_label=dict((m,n) for n,m in (test_data_generator.class_indices).items())
final_predict=pd.Series(prediction).map(map_label).values
y_test=list(final_test_data.label)


#Plotting the confusion matrix
plt.figure(figsize=(15, 15))
plt.style.use("classic")
number_images = (5, 5)
for i in range(1, (number_images[0] * number_images[1]) + 1):
    plt.subplot(number_images[0], number_images[1], i)
    plt.axis("off")

    color = "green"
    if final_test_data.label.iloc[i] != final_predict[i]:
        color = "red"
    plt.title(f"True:{final_test_data.label.iloc[i]}\nPredicted:{final_predict[i]}", color=color)
    plt.imshow(plt.imread(final_test_data['image_data'].iloc[i]))

plt.show()
