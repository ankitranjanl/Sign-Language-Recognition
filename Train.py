import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

image_dir = "Data/"
input_shape = (224, 224, 3)  # updated input shape

images = []
labels = []
label_to_index = {}
index_to_label = {}
for idx, folder in enumerate(os.listdir(image_dir)):
    label_to_index[folder] = idx
    index_to_label[idx] = folder
    for filename in os.listdir(os.path.join(image_dir, folder)):
        img = cv2.imread(os.path.join(image_dir, folder, filename))
        img = cv2.resize(img, input_shape[:2])
        images.append(img)
        labels.append(idx)
images = np.array(images)
labels = np.array(labels)
num_classes = len(label_to_index)
labels = to_categorical(labels, num_classes)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

print("Image shape:", img.shape)

model.save("my_model.h5")

with open("labels.txt", "w") as f:
    for label in label_to_index:
        f.write(label + "\n")
