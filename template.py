import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import utils as np_utils
import numpy as np

img = np.load('images.npy')
lbl = np.load('labels.npy')
iFlat = np.reshape(img, (6500, 784))
# print(iFlat.shape)
# print(lbl.shape)
# print(lbl[1])
lOH = keras.utils.to_categorical(lbl, num_classes=10)
# print(lOH.shape)
# print(lOH[1])

strat = [[] for x in range(0,10)]
count = 0
for i in lbl:
    strat[i].append(iFlat[count])
    count += 1

strat = np.array(strat)
print(strat.shape)
train = np.array([]).reshape(0,784)
validate = np.array([]).reshape(0,784)
test = np.array([]).reshape(0,784)
for i in range(0, 10):
    temp = np.split(strat[i], [int(.6 * len(strat[i])), int(.75 * len(strat[i]))])
    train = np.concatenate((train, np.array(temp[0])), axis=0)
    validate = np.concatenate((validate, np.array(temp[1])), axis=0)
    test = np.concatenate((test, np.array(temp[2])), axis=0)

print(len(strat[0]))

print(len(train))
print(len(validate))
print(len(test))

# Model Template

# model = Sequential() # declare model
# model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
# model.add(Activation('relu'))
# #
# #
# #
# # Fill in Model Here
# #
# #
# model.add(Dense(10, kernel_initializer='he_normal')) # last layer
# model.add(Activation('softmax'))


# # Compile Model
# model.compile(optimizer='sgd',
#               loss='categorical_crossentropy', 
#               metrics=['accuracy'])

# # Train Model
# history = model.fit(x_train, y_train, 
#                     validation_data = (x_val, y_val), 
#                     epochs=10, 
#                     batch_size=512)


# # Report Results

# print(history.history)
# model.predict()