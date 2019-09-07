import shutil
import os
import numpy as np
import math
from shutil import copyfile


def extract_color_imgs():
    """
    :return: take all the color images and copy them in images_color_tr folder making that folder if doesn't already exist
    """
    try:
        os.makedirs("./images_color_tr/")
    except FileExistsError:
        # directory already exists
        pass

    for folder1 in os.listdir("./dataset5/"):

        print(folder1)
        if not folder1.startswith('.'):
            for folder2 in os.listdir("./dataset5/" + folder1):
                try:
                    os.makedirs("./images_color_tr/" + folder2)
                except FileExistsError:
                    # directory already exists
                    pass
                print(folder2)
                if not folder2.startswith('.'):
                    for filename in os.listdir("./dataset5/" + folder1 + "/" + folder2):
                        if filename.endswith(".png") and "color" in filename:
                            copyfile("./dataset5/" + folder1 + "/" + folder2 + "/"+ filename,"./images_color_tr/" + folder2 + "/" + "_" + folder1 + filename)


def Generate_test_set(array_random):  # trasferisce il 10% delle immagini del training nel test set
    """
    :return: take 10% of traing set and put it in test folder making that folder if doesn't already exist
    """
    counter = 0
    try:
        os.makedirs("./images_color_test/")
    except FileExistsError:
        # directory already exists
        pass
    try:
        os.makedirs("./images_depth_test/")
    except FileExistsError:
        # directory already exists
        pass

    for folder1 in os.listdir("images_color_tr/"):
        print("lettera", folder1)
        if not folder1.startswith("."):
            #print("entrato nell'if")

            for filename in os.listdir("images_color_tr/" + folder1):

                #print(counter)

                if not filename.startswith("."):
                    counter += 1
                    if filename.endswith(".png") and counter in array_random:

                        try:
                            os.makedirs("./images_color_test/" + folder1)
                        except FileExistsError:
                            # directory already exists
                            pass
                        try:
                            os.makedirs("./images_depth_test/" + folder1)
                        except FileExistsError:
                            # directory already exists
                            pass
                        shutil.move("./images_color_tr/" + folder1 + "/" + filename, "./images_color_test/" + folder1 + "/" + filename)
                        filename_depth = filename.replace("color", "depth")
                        shutil.move("./images_depth_tr/" + folder1 + "/" + filename_depth, "./images_depth_test/" + folder1 + "/" + filename_depth)

def Generate_validation_set(array_random):  # trasferisce il 20% delle immagini nel VA
    """
    :return: take 20% of the remaining tr-test and put it in validation folder making it if doesn't already exist
    """
    counter = 0
    try:
        os.makedirs("./images_color_validation/")
    except FileExistsError:
        # directory already exists
        pass
    try:
        os.makedirs("./images_depth_validation/")
    except FileExistsError:
        # directory already exists
        pass

    for folder1 in os.listdir("images_color_tr/"):
        if not folder1.startswith("."):
            print("lettera", folder1)
            #print("entrato nell'if")



            for filename in os.listdir("images_color_tr/" + folder1):

                #print(counter)

                if not filename.startswith("."):
                    counter += 1
                    if filename.endswith(".png") and counter in array_random:
                        try:
                            os.makedirs("./images_color_validation/" + folder1)
                        except FileExistsError:
                            # directory already exists
                            pass
                        try:
                            os.makedirs("./images_depth_validation/" + folder1)
                        except FileExistsError:
                            # directory already exists
                            pass
                        shutil.move("./images_color_tr/" + folder1 + "/" + filename, "./images_color_validation/" + folder1 + "/" + filename)
                        filename_depth = filename.replace("color", "depth")
                        shutil.move("./images_depth_tr/" + folder1 + "/" + filename_depth,
                                    "./images_depth_validation/" + folder1 + "/" + filename_depth)

