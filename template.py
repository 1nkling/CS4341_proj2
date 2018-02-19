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
lblStrat = [[] for x in range(0,10)]
count = 0
for i in lbl:
    strat[i].append(iFlat[count])
    lblStrat[i].append(lOH[count])
    count += 1

strat = np.array(strat)
lblstrat = np.array(lblStrat)
print(strat.shape)
train = np.array([]).reshape(0,784)
validate = np.array([]).reshape(0,784)
test = np.array([]).reshape(0,784)
lbltrain = np.array([]).reshape(0,10)
lblvalidate = np.array([]).reshape(0,10)
lbltest = np.array([]).reshape(0,10)
for i in range(0, 10):
    temp = np.split(strat[i], [int(.6 * len(strat[i])), int(.75 * len(strat[i]))])
    lbltemp = np.split(lblstrat[i], [int(.6 * len(lblstrat[i])), int(.75 * len(lblstrat[i]))])
    lbltrain = np.concatenate((lbltrain, np.array(lbltemp[0])), axis=0)
    lblvalidate = np.concatenate((lblvalidate, np.array(lbltemp[1])), axis=0)
    lbltest = np.concatenate((lbltest, np.array(lbltemp[2])), axis=0)
    train = np.concatenate((train, np.array(temp[0])), axis=0)
    validate = np.concatenate((validate, np.array(temp[1])), axis=0)
    test = np.concatenate((test, np.array(temp[2])), axis=0)

# print("stratified labels:\n")
# print(len(lblStrat[0]))
# print(len(lblStrat[1]))
# print(len(lblStrat[2]))
# print(len(lblStrat[3]))
# print(len(lblStrat[4]))
# print(len(lblStrat[5]))
# print(len(lblStrat[6]))
# print(len(lblStrat[7]))
# print(len(lblStrat[8]))
# print(len(lblStrat[9]))
# print("\nstratified images:\n")
# print(len(strat[0]))
# print(len(strat[1]))
# print(len(strat[2]))
# print(len(strat[3]))
# print(len(strat[4]))
# print(len(strat[5]))
# print(len(strat[6]))
# print(len(strat[7]))
# print(len(strat[8]))
# print(len(strat[9]))

print(len(train))
print(len(validate))
print(len(test))

print(len(lbltrain))
print(len(lblvalidate))
print(len(lbltest))

# Model Template

model = Sequential() # declare model
model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))
#
#
#
# Fill in Model Here
#
#
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train Model
history = model.fit(train, lbltrain,
                    epochs=10, 
                    batch_size=512)


# # Report Results

print(history.history)
#model.predict()