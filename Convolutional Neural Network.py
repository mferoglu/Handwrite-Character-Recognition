def calculate_mem_use(snapshot, key_type='lineno', func=""):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)
    total = 0
    for index, stat in enumerate(top_stats, 1):
        frame = stat.traceback[0]
        if func in frame.filename or "data" in frame.filename:
            total = total + stat.size
    return total

import tracemalloc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import time
from time import perf_counter
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten 
from sklearn.utils import shuffle
from keras.utils import to_categorical

#Reading the data from csv
print("SUCCESS!")
raw_data = pd.read_csv('data//A_Z Handwritten Data.csv')
print(raw_data.head(5))
alphabet = "abcdefghijklmnopqrstuvwxyz"
print(alphabet)
labels = raw_data['0']                             # Subtract labels from dataset
del raw_data['0']

x_labels =[]
for a in labels:
    x_labels.append(alphabet[a])
    
print(raw_data.head(5))
print(type(labels))

ax = plt.figure(figsize = (10,5))
sns.countplot(x=x_labels)

plt.show()

print("SUCCESS.")

print(raw_data.info())
X = raw_data
Y = labels


# Split the dataset into train-test

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 67% training and 33% test
X_train = np.reshape(X_train.values, (X_train.shape[0], 28,28))
X_test = np.reshape(X_test.values, (X_test.shape[0], 28,28))

print("Train data shape: ", X_train.shape)
print("Test data shape: ", X_test.shape)


# Shuffle and print the dataset as images.

shuff = shuffle(X_train[:100])
shuff = shuff.astype("float32")
fig, ax = plt.subplots(3,3, figsize = (10,10))
axes = ax.flatten()

for i in range(9):
    th, shu = cv2.threshold(shuff[i], 30, 200, cv2.THRESH_BINARY_INV)
    axes[i].imshow(np.reshape(shuff[i], (28,28)), cmap="Greys")
plt.show()

# Reshape the dataset

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
print("New shape of train data: ", X_train.shape)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],1)
print("New shape of train data: ", X_test.shape)

y_trainOHE = to_categorical(y_train, num_classes = 26, dtype='int')
print("New shape of train labels: ", y_trainOHE.shape)

y_testOHE = to_categorical(y_test, num_classes = 26, dtype='int')
print("New shape of test labels: ", y_testOHE.shape)

# ADD CNN some layers
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))

model.add(Dense(26,activation ="softmax"))
print("Success!")

# Train the model and measure metrics

tracemalloc.start()
start_timer= perf_counter()
model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_trainOHE, epochs=1,  validation_data = (X_test,y_testOHE))
snapshot = tracemalloc.take_snapshot()
mem_use = calculate_mem_use(snapshot, func="cnn")
print("Mem. use : %1.f B" %(mem_use))
end_timer= perf_counter()
print(f"Execution time of CNN {end_timer-start_timer:0.4f} seconds")
print("Success!")



model.summary()
model.save(r'model_hand.h5')

# Performance metrics
print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
fig, axes = plt.subplots(3,3, figsize=(8,9))
axes = axes.flatten()

# Do some predictions from dataset.

for i,ax in enumerate(axes):
    img = np.reshape(X_test[i], (28,28))
    ax.imshow(img, cmap="Greys")
    
    pred = word_dict[np.argmax(y_testOHE[i])]
    ax.set_title("Prediction: "+pred)
    ax.grid()
