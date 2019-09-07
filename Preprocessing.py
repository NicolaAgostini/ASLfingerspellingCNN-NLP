from color_img import *
from depth_color_tr import *
from Cropped_tr_ts import *
from CNN import *


def preprocessing_dataset5():
    # PREPROCESSING


    print("total number of images ", sum([len(files) for r, d, files in os.walk("./dataset5/")]))  # Total number of images

    extract_color_imgs()
    extract_depth_imgs()

    n_color = sum([len(files) for r, d, files in os.walk("./images_color_tr/")])  # Total number of images color
    #print(n_color)

    how_much_ts = math.floor(n_color * 0.1)  # split 10 TEST e 90 dataset
    array_random_ts = np.random.randint(n_color, size=(1, how_much_ts))  # array maschera

    Generate_test_set(array_random_ts)  #GENERATE TEST SET


    n_color = sum([len(files) for r, d, files in os.walk("./images_color_tr/")])  # Remaining number of images color in tr

    how_much_va = math.floor(n_color * 0.2)  # split 20 VA e 80 dataset
    array_random_va = np.random.randint(n_color, size=(1, how_much_va))  # array maschera


    Generate_validation_set(array_random_va)  # GENERATE VALIDATION SET






    #crop_imgs_suitable_for_colab_folder()  # crop the images
    #Generate_test_set_cropped()  # generate test set cropped images
    #Generate_validation_set_cropped()  # generate validation set of cropped images


