import tensorflow as tf
from tensorflow import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import matplotlib.image as mpimg



#PS C:\Users\adham> & C:/Users/adham/AppData/Local/Programs/Python/Python311/python.exe "c:/Users/adham/OneDrive/Desktop/Projects/CVAAI Project/Model1CVAAI.py"
#2023-12-05 19:58:19.869660: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
#WARNING:tensorflow:From C:\Users\adham\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

#2023-12-05 19:58:24.535947: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#Num GPUs Available:  0
#WARNING:tensorflow:From c:\Users\adham\OneDrive\Desktop\Projects\CVAAI Project\Model1CVAAI.py:21: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.
#Instructions for updating:
#Use `tf.config.list_physical_devices('GPU')` instead.
#TensorFlow is using GPU:  False


# Check the number of available GPUs
print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

# Check if TensorFlow is using GPU acceleration
print("TensorFlow is using GPU: ", tf.test.is_gpu_available())



# Load and preprocess the data
#Loads image path, extracts labels,
#creates dataframe.
#Parameter: Data Path
def load_and_preprocess_data(data_path):
    image_data_path = list(data_path.glob("**/*.jpg"))
    label_path = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], image_data_path))
    final_data = pd.DataFrame({"image_data": image_data_path, "label": label_path}).astype("str")
    final_data = final_data.sample(frac=1).reset_index(drop=True)
    return final_data


# Load and preprocess images

#This function is designed to load and 
#preprocess a set of images from a specified directory. 
#The preprocessing includes resizing the images to a standard size
#of 224x224 pixels, converting the images from a PIL Image format
#to an array format, and then visualizing a 5x5 grid of randomly
#selected images from the directory.
#Parameter: Directory and number of samples.
def load_and_preprocess_images(directory, num_samples=25):
    plt.figure(figsize=(15, 15))
    for i in range(num_samples):
        file = random.choice(os.listdir(directory))
        image_path = os.path.join(directory, file)
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        i_value = int(i.numpy())
        ax = plt.subplot(5, 5, i_value + 1)
        plt.imshow(img_array.astype("uint8"))
        plt.axis("off")
    plt.show()

#"C:\Users\adham\OneDrive\Desktop\Projects\Dataset3"
# Directory paths
food_images = r'\Users\adham\OneDrive\Desktop\Projects\Dataset3\Training\food'
building_images = r'\Users\adham\OneDrive\Desktop\Projects\Dataset3\Training\building'
landscape_images = r'\Users\adham\OneDrive\Desktop\Projects\Dataset3\Training\landscape'
people_images = r'\Users\adham\OneDrive\Desktop\Projects\Dataset3\Training\people'


# Load and preprocess images for each class
load_and_preprocess_images(food_images)
load_and_preprocess_images(building_images)
load_and_preprocess_images(landscape_images)
load_and_preprocess_images(people_images)


# Data paths
image_path = r'\\Users\\adham\\OneDrive\\Desktop\\Projects\\Dataset3\\Training\\'
train_data_path = Path(image_path)
valid_data_path = Path(r'\\Users\\adham\\OneDrive\\Desktop\\Projects\\Dataset3\\Validation\\')
test_data_path = Path(r'\\Users\\adham\\OneDrive\\Desktop\\Projects\\Dataset3\\Testing\\')


# Load and preprocess data
final_train_data = load_and_preprocess_data(train_data_path)
final_valid_data = load_and_preprocess_data(valid_data_path)
final_test_data = load_and_preprocess_data(test_data_path)

# Data generators

#Batch size
batch_size = 30

#Train Data Generator
#Augments training data in order to have
#more augmented data
traindata_generator = ImageDataGenerator(
    rescale=1./ 255,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    fill_mode='nearest'
)

# Validation Data Generator
validdata_generator = ImageDataGenerator(rescale=1. / 255)
testdata_generator = ImageDataGenerator(rescale=1. / 255)

# Model parameters
train_data_generator = traindata_generator.flow_from_dataframe(
    dataframe=final_train_data,
    x_col="image_data",
    y_col="label",
    batch_size=batch_size,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=True
)

valid_data_generator = validdata_generator.flow_from_dataframe(
    dataframe=final_valid_data,
    x_col="image_data",
    y_col="label",
    batch_size=batch_size,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=True
)

test_data_generator = testdata_generator.flow_from_dataframe(
    dataframe=final_test_data,
    x_col="image_data",
    y_col="label",
    batch_size=batch_size,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False
)

# Model
#Sequential model, utilizing the pre trained weights
#of image net as a base
#Contains 5 layers
#100 neurons, decreases by 30, 20, 30, 16
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = keras.models.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(input_shape = (224,224,3)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(70, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

model.summary()

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)


# Training
#And training times
model_history = model.fit(train_data_generator, epochs=1, validation_data=valid_data_generator)

# Evaluation
test_loss, test_accuracy = model.evaluate(test_data_generator)
print(f"Test Accuracy: {test_accuracy}")

# Predictions and visualization of only 25 images
prediction= model.predict(test_data_generator)
prediction=np.argmax(prediction,axis=1)
map_label=dict((m,n) for n,m in test_data_generator.class_indices.items())
final_predict=pd.Series(prediction).map(map_label).values
y_test=list(final_test_data.label)


plt.figure(figsize=(15, 15))
plt.style.use("classic")
number_images = (5, 5)
for i in range(1, (number_images[0] * number_images[1]) + 1):
    plt.subplot(number_images[0], number_images[1], int(i))  # Convert i to integer
    plt.axis("off")

    i = i.numpy() if isinstance(i, tf.Tensor) else i  # Convert to NumPy array if it's a TensorFlow tensor

    color = "green"
    if final_test_data.label.iloc[int(i)] != final_predict[int(i)]:
        color = "red"
    plt.title(f"True:{final_test_data.label.iloc[int(i)]}\nPredicted:{final_predict[int(i)]}", color=color)
    plt.imshow(plt.imread(final_test_data['image_data'].iloc[int(i)]))

plt.show()