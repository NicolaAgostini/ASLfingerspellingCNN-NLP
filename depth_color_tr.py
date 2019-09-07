import shutil
import os
import numpy as np
import math
from shutil import copyfile


def extract_depth_imgs():
    """
    :return: take all the depth images and copy them in images_depth_tr folder making it if doesn't already exist
    """
    try:
        os.makedirs("./images_depth_tr/")
    except FileExistsError:
        # directory already exists
        pass

    for folder1 in os.listdir("./dataset5/"):

        print(folder1)
        if not folder1.startswith('.'):
            for folder2 in os.listdir("./dataset5/" + folder1):
                try:
                    os.makedirs("./images_depth_tr/" + folder2)
                except FileExistsError:
                    # directory already exists
                    pass
                print(folder2)

                if not folder2.startswith('.'):
                    for filename in os.listdir("./dataset5/" + folder1 + "/" + folder2):
                        if filename.endswith(".png") and "depth" in filename:
                            copyfile("./dataset5/" + folder1 + "/" + folder2 + "/"+ filename, "./images_depth_tr/" + folder2 + "/" + "_" + folder1 + filename)

