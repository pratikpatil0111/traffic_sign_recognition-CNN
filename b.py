import numpy as np
import streamlit as st
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os

# Function to preprocess the image
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

# Function to load and preprocess the dataset
def load_dataset(path):
    images = []
    classNo = []
    for count, folder in enumerate(os.listdir(path)):
        myPicList = os.listdir(os.path.join(path, folder))
        for y in myPicList:
            curImg = cv2.imread(os.path.join(path, folder, y))
            images.append(preprocessing(curImg))
            classNo.append(count)
    images = np.array(images)
    classNo = np.array(classNo)
    return images, classNo

# Load dataset
path = "Dataset" 
images, classNo = load_dataset(path)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Preprocess the data for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Data augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1,   
                             height_shift_range=0.1,
                             zoom_range=0.2,  
                             shear_range=0.1,  
                             rotation_range=10)  
dataGen.fit(X_train)

# One-hot encode the labels
y_train = to_categorical(y_train)
y_validation = to_categorical(y_validation)
y_test = to_categorical(y_test)

# Load or define the model
def myModel():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(np.unique(classNo)), activation='softmax')) 
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create Streamlit app
st.title('Traffic Sign Recognition')

# Display dataset information
st.write(f"Total Classes Detected: {len(np.unique(classNo))}")
st.write("Data Shapes")
st.write(f"Train: {X_train.shape}, {y_train.shape}")
st.write(f"Validation: {X_validation.shape}, {y_validation.shape}")
st.write(f"Test: {X_test.shape}, {y_test.shape}")

# Train the model
model = myModel()
history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=32),
                              steps_per_epoch=len(X_train) // 32,
                              epochs=10,
                              validation_data=(X_validation, y_validation),
                              shuffle=True)

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
st.write('Test Score:', score[0])
st.write('Test Accuracy:', score[1])

# Save the model
model.save("model.h5")
