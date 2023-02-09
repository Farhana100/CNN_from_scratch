import numpy as np
import pandas as pd
from data_loader import load_data
from preprocessing import shapify_image, preprocess_image
from tqdm import tqdm
from alexnet import AlexNet
from lenet import LeNet
from testmodel import TestNet
import matplotlib.pyplot as plt

from convolution import ConvolutionLayer
from max_pooling import MaxPoolLayer
from flattening import FlatteningLayer
from fully_connected import FullyConnectedLayer
from softmax import Softmax
from relu_activation import ReLU
from evaluation_metrices import accuracy

import pickle
import sys

from env import *

np.random.seed(SEED)

def main():

    file_path = sys.argv[1:][0]

    # create model
    # model = AlexNet(LEARNING_RATE, INITIALIZER, NUM_CLASSES)
    # model = LeNet(LEARNING_RATE, INITIALIZER, NUM_CLASSES)
    model = TestNet(LEARNING_RATE, INITIALIZER, NUM_CLASSES)

    # load pickle file
    with open(MODEL_FILENAME, 'rb') as f:
        trained_model = pickle.load(f)

    model.setLayers(trained_model)

    # Load data
    
    # read csv file
    # df_test = pd.read_csv(TEST_SET_CSV_PATH)
    df_test = pd.read_csv(file_path)

    print(df_test.head(5))
    print(df_test.shape)
    # return

    # remove unnecessary columns
    df_test.drop(['original filename', 'scanid', 'database name original', 'num'], axis=1, inplace=True)

    # split and reduce size
    df_test = df_test.sample(frac=TEST_SET_SECTION, random_state=SEED)
    print(df_test.shape)


    # prepare test data
    # load images
    test_data = load_data(df_test) 

    # Preprocess data
    test_data = shapify_image(test_data, img_size=(28, 28))
    test_data = preprocess_image(test_data)

    # predict
    predictions = model.predict(test_data[0]).argmax(axis=1)
    print(predictions[:10])

    #write to csv
    df_pred = pd.DataFrame()
    df_pred['FileName'] = df_test['filename']
    df_pred['Digit'] = predictions
    df_pred.to_csv(PREDICTION_FILENAME, index=False)

    test_acc = model.evaluate(test_data)
    print('Test accuracy: {}'.format(test_acc))



if __name__ == '__main__':
    main()
