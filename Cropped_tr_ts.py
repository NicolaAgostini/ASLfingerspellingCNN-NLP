import shutil
import os
import numpy as np
import math

def Generate_test_set_cropped(): # trasferisce il 10% delle immagini del training nel test set
    """
    :return: generate test set of the cropped images
    """
    try:
        os.makedirs("./images_cropped_test/")
    except FileExistsError:
        # directory already exists
        pass

    for folder1 in os.listdir("images_cropped/"):
        print("lettera", folder1)
        if not folder1.startswith("."):
            #print("entrato nell'if")
            counter = 0


            path, dirs, files = next(os.walk("images_cropped/" + folder1))
            file_count = len(files)
            how_much = math.floor(file_count * 0.1)  # split 10 TEST e 90 dataset
            a = np.random.randint(file_count, size=(1, how_much))
            for filename in os.listdir("images_cropped/" + folder1):

                #print(counter)

                if not filename.startswith("."):
                    counter += 1
                    if filename.endswith(".png") and counter in a:
                        #os.rename("./dataset5/A/" + folder1 + "/" + filename, "./validation/A/" + folder1 + "/" + filename)
                        #print("sono qui")
                        try:
                            os.makedirs("./images_cropped_test/" + folder1)
                        except FileExistsError:
                            # directory already exists
                            pass
                        shutil.move("./images_cropped/" + folder1 + "/" + filename, "./images_cropped_test/" + folder1 + "/" + filename)

def Generate_validation_set_cropped():  # trasferisce il 20% delle immagini nel VA
    """
    :return: return a validation set from the training set making folders if don't already exist
    """

    try:
        os.makedirs("./images_cropped_validation/")
    except FileExistsError:
        # directory already exists
        pass

    for folder1 in os.listdir("images_cropped/"):
        if not folder1.startswith("."):
            print("lettera", folder1)
            #print("entrato nell'if")
            counter = 0

            path, dirs, files = next(os.walk("images_cropped/" + folder1))
            file_count = len(files)
            how_much = math.floor(file_count * 0.2)  # split 80 TR e 20 VA
            a = np.random.randint(file_count, size=(1, how_much))
            for filename in os.listdir("images_cropped/" + folder1):

                #print(counter)

                if not filename.startswith("."):
                    counter += 1
                    if filename.endswith(".png") and counter in a:
                        try:
                            os.makedirs("./images_cropped_validation/" + folder1)
                        except FileExistsError:
                            # directory already exists
                            pass
                        shutil.move("./images_cropped/" + folder1 + "/" + filename, "./images_cropped_validation/" + folder1 + "/" + filename)

