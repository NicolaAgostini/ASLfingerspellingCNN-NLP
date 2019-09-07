from CNN import *
from PIL import Image
import os
from Cropped_tr_ts import *
from color_img import *
from depth_color_tr import *
from JASE import *
from Preprocessing import *
import copy


def main():

    #Preprocessing

    preprocessing_dataset5()



    # COLOR MODEL
    num_training = sum([len(files) for r, d, files in os.walk("./images_color_tr/")])

    num_validation = sum([len(files) for r, d, files in os.walk("./images_color_validation/")])

    num_test = sum([len(files) for r, d, files in os.walk("./images_color_test/")])

    train_data_dir_color = './images_color_tr/'
    validation_data_dir_color = './images_color_validation/'
    train_flow_color = create_train_generator(train_data_dir_color)
    validation_flow_color = create_validation_generator(validation_data_dir_color)
    #print(train_flow)

    model_is_in_folder = checkIfModelExist_color()


    model_color = cnn(train_flow_color, validation_flow_color, model_is_in_folder, "color", num_training, num_validation)

    path_images = foundImages_color()



    # make prediction on model color

    label, probabilities = predict(model_color, path_images, 1, "color", num_test)





    # DEPTH MODEL
    num_training = sum([len(files) for r, d, files in os.walk("./images_depth_tr/")])

    num_validation = sum([len(files) for r, d, files in os.walk("./images_depth_validation/")])


    num_test = sum([len(files) for r, d, files in os.walk("./images_depth_test/")])

    train_data_dir_depth = './images_depth_tr/'
    validation_data_dir_depth = './images_depth_validation/'
    train_flow_depth = create_train_generator(train_data_dir_depth)
    validation_flow_depth = create_validation_generator(validation_data_dir_depth)
    # print(train_flow)

    model_is_in_folder = checkIfModelExist_depth()

    model_depth = cnn(train_flow_depth, validation_flow_depth, model_is_in_folder, "depth", num_training, num_validation)

    path_images = foundImages_depth()

    # make prediction on model depth

    label, probabilities = predict(model_depth, path_images, 1, "depth", num_test)


    # ON TEST FOLDER

    predictions_number = 13
    sentence, ambiguo = predict_sentence(predictions_number, model_color, model_depth)
    #print(sentence)
    if ambiguo:
        JASE_out = most_prob_sent(sentence)
    else:
        JASE_out = segment2(sentence)

    print("final classification", JASE_out)





def checkIfModelExist_color():
    """
    :return: true is there is a model color in the directory ( .h5 and .json files)
    """
    model_exist = False
    weight_exist = False
    for File in os.listdir("."):

        if File.endswith(".json") and "color" in File:

            model_exist = True
        if File.endswith(".h5") and "color" in File:
            weight_exist = True

    return model_exist and weight_exist

def checkIfModelExist_depth():
    """
    :return: true is there is a model depth in the directory ( .h5 and .json files)
    """
    model_exist = False
    weight_exist = False
    for File in os.listdir("."):
        if File.endswith(".json") and "depth" in File:
            model_exist = True
        if File.endswith(".h5") and "depth" in File:
            weight_exist = True

    return model_exist and weight_exist

def foundImages_depth():
    return './images_depth_test/'

def foundImages_color():
    return './images_color_test/'

def my_test(tipo):
    if tipo == "color":
        return "./Test/color/"
    if tipo == "depth":
        return "./Test/depth/"

def create_class_dict():
    """
    :return: create a dict that associate a character to every class in range 0 - 23
    """
    dict = {}
    dict[0] = "a"
    dict[1] = "b"
    dict[2] = "c"
    dict[3] = "d"
    dict[4] = "e"
    dict[5] = "f"
    dict[6] = "g"
    dict[7] = "h"
    dict[8] = "i"
    dict[9] = "k"
    dict[10] = "l"
    dict[11] = "m"
    dict[12] = "n"
    dict[13] = "o"
    dict[14] = "p"
    dict[15] = "q"
    dict[16] = "r"
    dict[17] = "s"
    dict[18] = "t"
    dict[19] = "u"
    dict[20] = "v"
    dict[21] = "w"
    dict[22] = "x"
    dict[23] = "y"
    return dict

def average_on_probability(probabilities_color ,probabilities_depth):  # media pesata delle prob (conta di piu quella color perchè ha accuracy piu alta)
    prob = []
    for i in range(len(probabilities_color)):
        prob.append( (probabilities_color[i]*0.535) + (probabilities_depth[i]*0.465) )
    return prob


def evaluate_sentences(predictions, ambigue, indici, strings, counter):
    """
    :param predictions: a simple sentences predicted by the classificator
    :param ambigue: a dict where dict[pos] = [a,b] in which in position pos of the string sentence there could be letter a or letter b
    :param indici: the indices of the dict ambigue
    :param strings: an array of character that contains all the possible strings
    :param counter: current number of value considered in the ambigue
    :return: all the possible variation of "predictions" string considering ambigue characters
    """


    # caso base

    if counter >= len(indici):
        return predictions


    predictions[indici[counter]] = ambigue[indici[counter]][0]
    predictions2 = copy.deepcopy(predictions)

    stringa1 = evaluate_sentences(predictions, ambigue, indici, strings, counter + 1)

    predictions2[indici[counter]] = ambigue[indici[counter]][1]
    stringa2 = evaluate_sentences(predictions2, ambigue, indici, strings, counter + 1)
    return stringa1 + stringa2









def predict_sentence(num_iter, model_color, model_depth):
    """
    :param num_iter: number of images to predict
    :param model_color: the keras model trained on colored imgs
    :param model_depth: the keras model trained on depth imgs
    :return: an array of possible sentences if the classification is ambiguous otherwise a only sequence of unseparated characters
    """
    dict = create_class_dict()
    predictions = []
    ambiguo = False
    ambigue = {}  # indice - > [lista lettere ambigue]
    for i in range(num_iter):
        label_color, probabilities_color = predict(model_color, "./Test/color/" + str(i), 1, "color", 1)
        label_depth, probabilities_depth = predict(model_depth, "./Test/depth/" + str(i), 1, "depth", 1)

        prob = np.asarray(average_on_probability(probabilities_color, probabilities_depth))

        indice_massimo = np.argmax(prob)

        if prob[0][indice_massimo] <= 0.5:
            ambiguo = True

            classificazioni_possibili = prob.argsort()
            classificazioni_possibili = np.flip(classificazioni_possibili)

            classificazioni_possibili = np.take(classificazioni_possibili, [0, 1])  # prendo i primi due più probabili

            ambigue[i] = [dict[classificazioni_possibili[0]], dict[classificazioni_possibili[1]]]


        predictions.append(dict[indice_massimo])
    if ambiguo == True:
        indici = list()  # le chiavi dove ho lettere ambigue
        for i in ambigue.keys():
            indici.append(i)
        strings = []
        strings = evaluate_sentences(predictions, ambigue, indici, strings, 0)

        sentences = []

        for i in range(2**len(indici)):
            sentences.append(strings[i*num_iter:num_iter*(i+1)])
            sentences[i] = "".join(sentences[i])


        return sentences, ambiguo
    else:
        predictions = "".join(predictions)
        return predictions, ambiguo




if __name__ == '__main__':
    main()