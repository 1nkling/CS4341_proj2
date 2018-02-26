import numpy as np
from math import floor
from sklearn.neighbors import KNeighborsClassifier
from skimage import io

images_as_matrices = np.load('images.npy')
labels_as_integers = np.load('labels.npy')

images_as_vertices = np.reshape(images_as_matrices, (6500, 784))
#labels_as_vertices = utils.to_categorical(labels_as_integers, num_classes=10)

zeroes = []
ones = []
twos = []
threes = []
fours = []
fives = []
sixes = []
sevens = []
eights = []
nines = []

for i in range(len(labels_as_integers)):
    if labels_as_integers[i] == 0:
        zeroes.append(images_as_vertices[i])
    elif labels_as_integers[i] == 1:
        ones.append(images_as_vertices[i])
    elif labels_as_integers[i] == 2:
        twos.append(images_as_vertices[i])
    elif labels_as_integers[i] == 3:
        threes.append(images_as_vertices[i])
    elif labels_as_integers[i] == 4:
        fours.append(images_as_vertices[i])
    elif labels_as_integers[i] == 5:
        fives.append(images_as_vertices[i])
    elif labels_as_integers[i] == 6:
        sixes.append(images_as_vertices[i])
    elif labels_as_integers[i] == 7:
        sevens.append(images_as_vertices[i])
    elif labels_as_integers[i] == 8:
        eights.append(images_as_vertices[i])
    else: #if labels_as_integers[i] == 9:
        nines.append(images_as_vertices[i])

def split(clas):
    training_set = []
    validation_set = []
    test_set = []

    # TODO: rewrite
    i = 0
    while i < floor(len(clas) * 0.60):
        training_set.append(clas[i])
        i += 1
    while i < floor(len(clas) * (0.60 + 0.15)):
        validation_set.append(clas[i])
        i += 1
    while i < floor(len(clas) * (0.60 + 0.15 + 0.25)):
        test_set.append(clas[i])
        i += 1

    return {'training_set': training_set, 'validation_set': validation_set, 'test_set': test_set}

zero_sets = split(zeroes)
one_sets = split(ones)
two_sets = split(twos)
three_sets = split(threes)
four_sets = split(fours)
five_sets = split(fives)
six_sets = split(sixes)
seven_sets = split(sevens)
eight_sets = split(eights)
nine_sets = split(nines)

training_set = {'images': [], 'labels': []}
validation_set = {'images': [], 'labels': []}
test_set = {'images': [], 'labels': []}

training_set['images'] = training_set['images'] + zero_sets['training_set']
training_set['images'] = training_set['images'] + one_sets['training_set']
training_set['images'] = training_set['images'] + two_sets['training_set']
training_set['images'] = training_set['images'] + three_sets['training_set']
training_set['images'] = training_set['images'] + four_sets['training_set']
training_set['images'] = training_set['images'] + five_sets['training_set']
training_set['images'] = training_set['images'] + six_sets['training_set']
training_set['images'] = training_set['images'] + seven_sets['training_set']
training_set['images'] = training_set['images'] + eight_sets['training_set']
training_set['images'] = training_set['images'] + nine_sets['training_set']
training_set['labels'] = training_set['labels'] + [0 for i in range(len(zero_sets['training_set']))]
training_set['labels'] = training_set['labels'] + [1 for i in range(len(one_sets['training_set']))]
training_set['labels'] = training_set['labels'] + [2 for i in range(len(two_sets['training_set']))]
training_set['labels'] = training_set['labels'] + [3 for i in range(len(three_sets['training_set']))]
training_set['labels'] = training_set['labels'] + [4 for i in range(len(four_sets['training_set']))]
training_set['labels'] = training_set['labels'] + [5 for i in range(len(five_sets['training_set']))]
training_set['labels'] = training_set['labels'] + [6 for i in range(len(six_sets['training_set']))]
training_set['labels'] = training_set['labels'] + [7 for i in range(len(seven_sets['training_set']))]
training_set['labels'] = training_set['labels'] + [8 for i in range(len(eight_sets['training_set']))]
training_set['labels'] = training_set['labels'] + [9 for i in range(len(nine_sets['training_set']))]

validation_set['images'] = validation_set['images'] + zero_sets['validation_set']
validation_set['images'] = validation_set['images'] + one_sets['validation_set']
validation_set['images'] = validation_set['images'] + two_sets['validation_set']
validation_set['images'] = validation_set['images'] + three_sets['validation_set']
validation_set['images'] = validation_set['images'] + four_sets['validation_set']
validation_set['images'] = validation_set['images'] + five_sets['validation_set']
validation_set['images'] = validation_set['images'] + six_sets['validation_set']
validation_set['images'] = validation_set['images'] + seven_sets['validation_set']
validation_set['images'] = validation_set['images'] + eight_sets['validation_set']
validation_set['images'] = validation_set['images'] + nine_sets['validation_set']
validation_set['labels'] = validation_set['labels'] + [0 for i in range(len(zero_sets['validation_set']))]
validation_set['labels'] = validation_set['labels'] + [1 for i in range(len(one_sets['validation_set']))]
validation_set['labels'] = validation_set['labels'] + [2 for i in range(len(two_sets['validation_set']))]
validation_set['labels'] = validation_set['labels'] + [3 for i in range(len(three_sets['validation_set']))]
validation_set['labels'] = validation_set['labels'] + [4 for i in range(len(four_sets['validation_set']))]
validation_set['labels'] = validation_set['labels'] + [5 for i in range(len(five_sets['validation_set']))]
validation_set['labels'] = validation_set['labels'] + [6 for i in range(len(six_sets['validation_set']))]
validation_set['labels'] = validation_set['labels'] + [7 for i in range(len(seven_sets['validation_set']))]
validation_set['labels'] = validation_set['labels'] + [8 for i in range(len(eight_sets['validation_set']))]
validation_set['labels'] = validation_set['labels'] + [9 for i in range(len(nine_sets['validation_set']))]

test_set['images'] = test_set['images'] + zero_sets['test_set']
test_set['images'] = test_set['images'] + one_sets['test_set']
test_set['images'] = test_set['images'] + two_sets['test_set']
test_set['images'] = test_set['images'] + three_sets['test_set']
test_set['images'] = test_set['images'] + four_sets['test_set']
test_set['images'] = test_set['images'] + five_sets['test_set']
test_set['images'] = test_set['images'] + six_sets['test_set']
test_set['images'] = test_set['images'] + seven_sets['test_set']
test_set['images'] = test_set['images'] + eight_sets['test_set']
test_set['images'] = test_set['images'] + nine_sets['test_set']
test_set['labels'] = test_set['labels'] + [0 for i in range(len(zero_sets['test_set']))]
test_set['labels'] = test_set['labels'] + [1 for i in range(len(one_sets['test_set']))]
test_set['labels'] = test_set['labels'] + [2 for i in range(len(two_sets['test_set']))]
test_set['labels'] = test_set['labels'] + [3 for i in range(len(three_sets['test_set']))]
test_set['labels'] = test_set['labels'] + [4 for i in range(len(four_sets['test_set']))]
test_set['labels'] = test_set['labels'] + [5 for i in range(len(five_sets['test_set']))]
test_set['labels'] = test_set['labels'] + [6 for i in range(len(six_sets['test_set']))]
test_set['labels'] = test_set['labels'] + [7 for i in range(len(seven_sets['test_set']))]
test_set['labels'] = test_set['labels'] + [8 for i in range(len(eight_sets['test_set']))]
test_set['labels'] = test_set['labels'] + [9 for i in range(len(nine_sets['test_set']))]

for i in range(4,5):
    print('k-nearest neighbors, weighted, for k = {}'.format(i))
    classifier = KNeighborsClassifier(n_neighbors=i, weights='distance').fit(training_set['images'], training_set['labels'])
    test_set_classifications = classifier.predict(test_set['images'])
    
    #
    # Right/Wrong Classifications for the Validation Set
    #
    number_of_right_classifications = 0
    number_of_wrong_classifications = 0
    
    for i in range(len(test_set_classifications)):
        if test_set_classifications[i] == test_set['labels'][i]:
            number_of_right_classifications += 1
        else:
            number_of_wrong_classifications += 1
    
    print('number of right classifications: {}'.format(number_of_right_classifications))
    print('number of wrong classifications: {}'.format(number_of_wrong_classifications))
    print('accuracy: {}'.format(number_of_right_classifications/len(test_set_classifications)))
    
    #
    # Confusion Matrix for the Validation Set
    #
    confusion_matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    for i in range(len(test_set_classifications)):
        true_label = test_set['labels'][i]
        predicted_label = test_set_classifications[i]
        confusion_matrix[true_label][predicted_label] += 1

        #if (true_label == 4 and predicted_label == 1):
        #    test_set_image_as_matrix = np.reshape(test_set['images'][i], (28, 28))
        #    io.imsave('test_set_image_{}.bmp'.format(i), test_set_image_as_matrix)

    print(np.matrix(confusion_matrix))
    print()

    for i in range(10):
        true_positives = confusion_matrix[i][i]
        false_positives = 0
        for j in range(10):
            if j != i:
                false_positives += confusion_matrix[j][i]
        false_negatives = 0
        for j in range(10):
            if j != i:
                false_negatives += confusion_matrix[i][j]
        true_negatives = len(test_set_classifications) - (true_positives + false_positives + false_negatives)

        #print('number of true positives for {} is {}'.format(i, true_positives))
        #print('number of false positives for {} is {}'.format(i, false_positives))
        #print('number of false negatives for {} is {}'.format(i, false_negatives))
        #print('number of true negatives for {} is {}'.format(i, true_negatives))
        print('Precision for {} is {}'.format(i, true_positives/(true_positives + false_positives)))
        print('Recall for {} is {}'.format(i, true_positives/(true_positives + false_negatives)))
        print()

    #neighbors = classifier.kneighbors([test_set['images'][775]], 4)

    #print(neighbors[1][0])
    #for neighbor_index in neighbors[1][0]:
    #    print(training_set['labels'][neighbor_index])
    #print(neighbors[0][0])

    #for neighbor_index in neighbors[1][0]:
    #    neighbor_image_as_matrix = np.reshape(training_set['images'][neighbor_index], (28,28))
    #    io.imsave('neighbor_image_{}.bmp'.format(neighbor_index), neighbor_image_as_matrix)
