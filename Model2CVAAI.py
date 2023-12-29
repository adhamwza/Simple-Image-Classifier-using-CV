#Imports
import tensorflow as tf
from tensorflow import keras
import random
import cv2 as c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


#Path for the food training data
food_images = '\\Users\\adham\\OneDrive\\Desktop\\Dataset3\\Training\\food'

#Visiualization of 25 images of the food
for i in range(25):
    file=random.choice(os.listdir(food_images))
    food_image_path=os.path.join(food_images,file)
    img=mpimg.imread(food_image_path)
    ax=plt.subplot(5,5,i+1)
    plt.imshow(img)
plt.show()

#Path for the building training data
building_images = '\\Users\\adham\\OneDrive\\Desktop\\Dataset3\\Training\\building'

#Visiualization of 25 images of the building
for i in range(25):
    file=random.choice(os.listdir(building_images))
    building_image_path=os.path.join(building_images,file)
    img=mpimg.imread(building_image_path)
    ax=plt.subplot(5,5,i+1)
    plt.imshow(img)
plt.show()

#Path for the landscape training data
landscape_images = '\\Users\\adham\\OneDrive\\Desktop\\Dataset3\\Training\\landscape'

#Visiualization of 25 images of the landscape
for i in range(25):
    file=random.choice(os.listdir(landscape_images))
    landscape_image_path=os.path.join(landscape_images,file)
    img=mpimg.imread(landscape_image_path)
    ax=plt.subplot(5,5,i+1)
    plt.imshow(img)
plt.show()


#Path for the food people data
people_images = '\\Users\\adham\\OneDrive\\Desktop\\Dataset3\\Training\\people'

#Visiualization of 25 images of the people
for i in range(25):
    file=random.choice(os.listdir(people_images))
    people_image_path=os.path.join(people_images,file)
    img=mpimg.imread(people_image_path)
    ax=plt.subplot(5,5,i+1)
    plt.imshow(img)
plt.show()


#Path for the full training dataset
image_path='\\Users\\adham\\OneDrive\\Desktop\\Dataset3\\Training\\'


#Label names
label_name= ['Food', 'Landscape', 'Building', 'People']


#Pre determined image size
image_size=(224,224)

#This part is just used to make sure that the
#code understands the amount of classes
class_names = os.listdir(image_path)
print(class_names)
print("Number of classes : {}".format(len(class_names)))

numberof_images={}
for class_name in class_names:
    numberof_images[class_name]=len(os.listdir(image_path+"/"+class_name))
images_each_class=pd.DataFrame(numberof_images.values(),index=numberof_images.keys(),columns=["Number of images"])
print(images_each_class)



#Determining the training, validation, and testing datasets and their shuffling
train_data_path=Path(r"\\Users\\adham\\OneDrive\\Desktop\\Dataset3\\Training\\")
image_data_path=list(train_data_path.glob(r"**/*.jpg"))
train_label_path=list(map(lambda x:os.path.split(os.path.split(x)[0])[1],image_data_path))
final_train_data=pd.DataFrame({"image_data":image_data_path,"label":train_label_path}).astype("str")
final_train_data=final_train_data.sample(frac=1).reset_index(drop=True)
print(final_train_data['image_data'])


valid_data_path=Path(r"\\Users\\adham\\OneDrive\\Desktop\\Dataset3\\Validation\\")
image_data_path=list(valid_data_path.glob(r"**/*.jpg"))
valid_label_path=list(map(lambda x:os.path.split(os.path.split(x)[0])[1],image_data_path))
final_valid_data=pd.DataFrame({"image_data":image_data_path,"label":valid_label_path}).astype("str")
final_valid_data=final_valid_data.sample(frac=1).reset_index(drop=True)
print(final_valid_data['image_data'])



test_data_path=Path(r"\\Users\\adham\\OneDrive\\Desktop\\Dataset3\\Testing\\")
image_data_path=list(test_data_path.glob(r"**/*.jpg"))
test_label_path=list(map(lambda x:os.path.split(os.path.split(x)[0])[1],image_data_path))
final_test_data=pd.DataFrame({"image_data":image_data_path,"label":test_label_path}).astype("str")
final_test_data=final_test_data.sample(frac=1).reset_index(drop=True)
print(final_test_data['image_data'])



#Batch size
batch_size=64
#Training data generator that generates augmented images for the model
#to continue training on
traindata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                    #rotation_range = 40,
                                                                    #zoom_range=0.4,
                                                                    #width_shift_range=0.2,
                                                                    #height_shift_range=0.2,
                                                                    #shear_range=0.2,
                                                                    #horizontal_flip=True,
                                                                    validation_split=0.2)
                                                                    #fill_mode='nearest')


#Data generator for both validation and testing
validdata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
testdata_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)



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
#Contains 8 layers, first containing 200 neurons, and goes down 30 neurons at a time
model = keras.models.Sequential([
    #first layer:
    keras.layers.Flatten(input_shape=[224, 224, 3]),
    #middle layers/ hidden layers and the softmax output layer (mutlitple layers. decrease neurons as u go down layers, until last output layer is the same as the number of classes in the dataset):
    keras.layers.Dense(200, activation = 'relu'),

    keras.layers.Dense(170, activation = 'relu'),

    keras.layers.Dense(140, activation='relu'),

    keras.layers.Dense(110, activation = 'relu') ,

    keras.layers.Dense(80, activation = 'relu') ,

    keras.layers.Dense(50, activation = 'relu'),

    keras.layers.Dense(20, activation='relu'),

    keras.layers.Dense(4, activation = 'softmax')
])

model.summary()

# Define your learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Create a learning rate scheduler callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)


model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00006),#last was .0025(acc(75 test,88 acc, 80 val)// 0.0015
              metrics = ['accuracy']
              )


#epochs is how many times the model will train. in this case, 5 times, so he gets more accurate and loses less as the epochs go.
early_stopping = tf.keras.callbacks.EarlyStopping( monitor='val_loss',patience=55,restore_best_weights=True)
model_history = model.fit(train_data_generator, epochs=50, validation_data=valid_data_generator, callbacks = [early_stopping])

test_loss, test_accuracy = model.evaluate(test_data_generator)
print(f"Test Accuracy: {test_accuracy}")


prediction= model.predict(test_data_generator)
prediction=np.argmax(prediction,axis=1)
map_label=dict((m,n) for n,m in (test_data_generator.class_indices).items())
final_predict=pd.Series(prediction).map(map_label).values
y_test=list(final_test_data.label)

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


# Generate predictions on the test set
predictions = model.predict(test_data_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Get true labels
true_labels = test_data_generator.classes

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Visualize confusion matrix using seaborn
class_names = list(test_data_generator.class_indices.keys())
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()