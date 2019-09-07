from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os.path
from keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.utils import plot_model

import cv2
import matplotlib.pyplot as plt
import sys
import shutil
import sklearn.metrics as metrics




def crop_imgs_suitable_for_colab_folder():
    """
    :return: this function crop the images rgb using images depth as a mask
    """
    print("reading depth")
    counter = 0
    try:
        os.makedirs("./images_cropped")
    except FileExistsError:
        # directory already exists
        pass
    for folder1 in os.listdir("./dataset5/"):  # A
        #print(folder1)

        if not folder1.startswith('.'):
            for folder2 in os.listdir("./dataset5/" + folder1):  # a
                print(counter)
                try:
                    os.makedirs("./images_cropped/" + folder2)
                except FileExistsError:
                    # directory already exists
                    pass
                #print(folder2)
                if not folder2.startswith('.'):
                    for filename1 in os.listdir("./dataset5/" + folder1 + "/" + folder2):
                        #print(filename1)
                        counter += 1
                        #print(counter)
                        if filename1.endswith(".png"):
                            ll = filename1
                            if "depth" in ll: # se è una depth
                                img = cv2.imread("./dataset5/" + folder1 + "/" + folder2 + "/" + ll,
                                                 cv2.IMREAD_UNCHANGED)  # img è la depth

                                # prendo la corrispondente immagine color
                                color_name = ll.replace("depth", "color")
                                img_color = cv2.imread("./dataset5/" + folder1 + "/" + folder2 + "/" + color_name)  # è l'immagine a colori
                                if (img_color is not None) and (not os.path.isfile("./images_cropped/" + folder2 + "/" +"_" + folder1 + color_name )):  # se l'immagine a colori corrispondente della depth esiste e se non c'è già

                                    for riga in range(img.shape[0]):

                                        for colonna in range(img.shape[1]):
                                            if 500 <= img[riga][colonna] <= 990:
                                                pass
                                            else:
                                                img_color[riga][colonna] = np.zeros(3)
                                    cv2.imwrite("./images_cropped/" + folder2 + "/" + "_" + folder1 + color_name, img_color)



def create_train_generator(train_data_dir):
    """
    :param train_data_dir: the path of the images to train the cnn
    :return: a train_generator
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(100, 100),
        color_mode="grayscale",
        batch_size=16,
        class_mode='categorical')
    return train_generator

def create_validation_generator(validation_data_dir):
    """
    :param validation_data_dir: the directory in which there are validation set
    :return: a validation_generator
    """
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(100, 100),
        color_mode="grayscale",
        batch_size=16,
        class_mode='categorical')
    return validation_generator


def cnn(train_flow, validation_flow, model_exist, tipo, num_training, num_validation):
    """
    :param train_flow: train_generator
    :param validation_flow: validation_generator
    :param model_exist: if the model is already in folder
    :param tipo: type of net colored (RGB) or depth
    :return: a cnn model trained with its weight
    """

    if tipo == "color" and model_exist:     # if model exist
        json_file = open('model_color.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        #load weights into new model
        model.load_weights("model_color.h5")
        #the model is loaded from disk
    if tipo=="depth" and model_exist:     # if model exist
        json_file = open('model_depth.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        #load weights into new model
        model.load_weights("model_depth.h5")
        #the model is loaded from disk

    if model_exist == False:       # if model doesn't exist
        if tipo == "color":
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(0.50))
            model.add(layers.Dense(24, activation='softmax'))

        if tipo == "depth":
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(0.25))

            model.add(layers.Dense(24, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history1 = model.fit_generator(
            train_flow,
            steps_per_epoch=num_training // 16,
            epochs=5,
            validation_data=validation_flow,
            validation_steps=num_validation // 16)
        plt.plot(history1.history['loss'])
        plt.plot(history1.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
        plt.clf()

        if tipo == "color":
            # Saving the model
            model_json = model.to_json()
            # serialize model to json
            with open("model_color.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("model_color.h5")

        if tipo == "depth":
            # Saving the model
            model_json = model.to_json()
            # serialize model to json
            with open("model_depth.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("model_depth.h5")
    plot_model(model, show_shapes=True, to_file=tipo + 'model.png')
    return model

def predict(model, path, batch_size, tipo, num_test):
    """
    :param model: the keras model
    :param path: the path of test set
    :param batch_size: number of example processed per batch
    :param tipo: type of net, colored or depth
    :param num_test: number of test images
    :return: the indices of the most probable prediction and also the prediction
    """
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        directory=path,
        target_size=(100, 100),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    prediction = model.predict_generator(test_generator, verbose=0, steps=num_test)    # numero di immagini / dimensione del batch
    predicted_class_indices = np.argmax(prediction, axis=1)
    if num_test != 1:
        true_classes = test_generator.classes

        class_labels = list(test_generator.class_indices.keys())
        try:  # in the case predicted_class_indices would be bigger than true_classes idk why this happens
            report = metrics.classification_report(true_classes, predicted_class_indices, target_names=class_labels)
        except:
            predicted_class_indices = np.resize(predicted_class_indices, len(true_classes))
            report = metrics.classification_report(true_classes, predicted_class_indices, target_names=class_labels)
        print(report)
        plot_classification_report(report, tipo)
    return predicted_class_indices, prediction





def plot_classification_report(cr, tipo):
    """
    :param cr: classification report with all the information about precision recall and f1_score
    :param tipo: the type of net, colored (RGB) or depth
    :return: print the plot of metrics
    """

    lines = cr.split('\n')

    classes = []
    plotMat = []

    for line in lines[2 : (len(lines) - 3)]:
        if line and not "micro" in line:  # to avoid errors using Theano
            #print(line)
            t = line.split()

            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]

            plotMat.append(v)



    cm = plotMat[:-1]
    labels = classes[:-1]
    precision = [x[0] for x in cm]

    recall = [x[1] for x in cm]
    f1_score = [x[2] for x in cm]

    x = np.arange(len(labels)*6, step=6)  # the label locations
    width = 1.5  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width , precision, width, label='precision')
    rects2 = ax.bar(x, recall, width, label='recall')
    rects3 = ax.bar(x + width , f1_score, width, label='f1-score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



    fig.tight_layout()

    plt.savefig(tipo + 'scores.png')
    plt.clf()
