import os
import subprocess
import numpy as np
import pandas as pd
import argparse

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
import tensorflow.keras as keras
import keras.optimizers as optimizers
from keras.models import Sequential
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape, Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121


class models:
  def CNNClassifier(num_hidden_layers, nn_params):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=nn_params['input_shape'], padding = 'same', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(num_hidden_layers-1):
        model.add(Conv2D(64, (3, 3), padding = 'same', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten()) 

    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 64, activation = 'relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    model.add(Dense(units = nn_params['output_neurons'], activation = nn_params['output_activation']))

    # initiate RMSprop optimizer
    opt = keras.optimizers.legacy.RMSprop(learning_rate=1e-5, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss=nn_params['loss'],
                  optimizer=opt,
                  metrics=['accuracy'])    
    return model


class pkg:
  def get_metadata(metadata_path, which_splits = ['train', 'test']):  
    '''returns metadata dataframe which contains columns of:
       * index: index of data into numpy data
       * class: class of image
       * split: which dataset split is this a xpart of? 
    '''
    metadata = pd.read_csv(metadata_path)
    keep_idx = metadata['split'].isin(which_splits)
    return metadata[keep_idx]
  
  def get_data_split(split_name, flatten, all_data, metadata, image_shape):
    '''
    returns images (data), labels from folder of format [image_folder]/[split_name]/[class_name]/
    flattens if flatten option is True 
    '''
    sub_df = metadata[metadata['split'].isin([split_name])]
    index  = sub_df['index'].values
    labels = sub_df['class'].values
    data = all_data[index,:]
    if flatten:
      data = data.reshape([-1, np.product(image_shape)])
    return data, labels
  
  def get_train_data(flatten, all_data, metadata, image_shape):
     return pkg.get_data_split('train', flatten, all_data, metadata, image_shape)
  
  def get_test_data(flatten, all_data, metadata, image_shape):
     return pkg.get_data_split('test', flatten, all_data, metadata, image_shape)

class helpers:
     
    def plot_acc(cnn_history, ax = None, xlabel = 'Epoch #'):
        # i'm sorry for this function's code. i am so sorry. 
        history = cnn_history.history
        #history.update({'epoch':list(range(len(history['val_accuracy'])))})
        #num_epochs = len(history['val_accuracy'])
        # history.update({'epoch':list(range(num_epochs)) })  # Add field "epoch" to dictionary
        history.update({'epoch':cnn_history.epoch })  # Add field "epoch" to dictionary
        history = pd.DataFrame.from_dict(history)

        best_epoch = history.sort_values(by = 'val_accuracy', ascending = False).iloc[0]['epoch']

        if not ax:
            f, ax = plt.subplots(1,1)
            sns.lineplot(x = 'epoch', y = 'val_accuracy', data = history, label = 'Validation', ax = ax)
            sns.lineplot(x = 'epoch', y = 'accuracy', data = history, label = 'Training', ax = ax)
            ax.axhline(0.5, linestyle = '--',color='red', label = 'Chance')
            ax.axvline(x = best_epoch, linestyle = '--', color = 'green', label = 'Best Epoch')  
            ax.legend(loc = 4)    
            ax.set_ylim([0.4, 1])

            ax.set_xlabel(xlabel)
            ax.set_ylabel('Accuracy (Fraction)')
            
            plt.show()
   

#  #### PLOTTING
#  def plot_one_image(data, labels = [], index = None, image_shape = [64,64,3]):
#    '''
#    if data is a single image, display that image
#
#    if data is a 4d stack of images, display that image
#    '''
#    num_dims   = len(data.shape)
#    num_labels = len(labels)
#
#    # reshape data if necessary
#    if num_dims == 1:
#      data = data.reshape(target_shape)
#    if num_dims == 2:
#      data = data.reshape(np.vstack[-1, image_shape])
#    num_dims   = len(data.shape)
#
#    # check if single or multiple images
#    if num_dims == 3:
#      if num_labels > 1:
#        print('Multiple labels does not make sense for single image.')
#        return
#
#      label = labels      
#      if num_labels == 0:
#        label = ''
#      image = data
#
#    if num_dims == 4:
#      image = data[index, :]
#      label = labels[index]
#
#    # plot image of interest
#    print('Label: %s'%label)
#    plt.imshow(image)
#    plt.show()

### defining project variables
# file variables
image_data_path      = 'image_data.npy'
metadata_path        = 'metadata.csv'
metadata_url         = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/' + metadata_path
image_data_url       = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/' + image_data_path
image_shape          = (64, 64, 3)

# CNN: neural net parameters
nn_params = {}
nn_params['input_shape']       = image_shape
nn_params['output_neurons']    = 1
nn_params['loss']              = 'binary_crossentropy'
nn_params['output_activation'] = 'sigmoid'

#
# Saving weights on Keras and callback function:
#   https://stackoverflow.com/questions/61046870/how-to-save-weights-of-keras-model-for-each-epoch
#
#monitor = ModelCheckpoint('./model.h5', monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

# Save models at each epoch:
checkpoint_path = "./checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
if not os.path.exists(checkpoint_dir):
   os.makedirs(checkpoint_dir)
monitor = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')


def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

def SummarizeMetaData(_all_data, _metadata):
    #plt.figure (figsize = (8, 7))
    #_metadata.groupby(['class']).count()
    ##sns.countplot(x = 'class', data = _metadata)
    #sns.countplot(x = _metadata["class"])

    #plt.figure (figsize = (8, 7))
    #_metadata.groupby(['split']).count()
    #sns.countplot(x = "split", data = _metadata)
    ##sns.countplot(x = _metadata["split"])    
    plt.figure (figsize = (4, 3))
    sns.countplot(x = "class", hue="split", data = _metadata)
    plt.ylabel("Number of images")
    plt.xlabel("Normal: 0.0,  Pneumonia: 1.0")

    

def DisplayDataset(train_data, train_labels, image_shape):

    num_plots = 6
    fig, axs = plt.subplots(ncols=num_plots, nrows=2, figsize = (15, 6))
    normal_indices = np.where(train_labels == 0)[0]
    pneumonia_indices = np.where(train_labels == 1)[0]
    for i in range(num_plots):
        axs[0][i].imshow(train_data[normal_indices[i]])
        axs[1][i].imshow(train_data[pneumonia_indices[i]])
    # Set ylabel
    axs[0][0].set_ylabel("Normal")
    axs[1][0].set_ylabel("Pneumonia")        
    

def RunKNeighborsClassifier(train_data, train_labels, test_data):
    classifier = KNeighborsClassifier(n_neighbors = 5)
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict(test_data)
    return predictions

def RunLogisticRegression(train_data, train_labels, test_data):
    classifier = LogisticRegression(max_iter = 1000) 
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict(test_data)
    return predictions

def RunDecisionTreeClassifier(train_data, train_labels, test_data):
    classifier = DecisionTreeClassifier(max_depth = 2)
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict(test_data)
    return predictions

def RunMLPClassifier(train_data, train_labels, test_data):
    classifier = MLPClassifier(alpha=1, max_iter=1000)
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict(test_data)
    return predictions

# This is a method how to verify whether the network structure is what you wanted to create
def TestingNetworkStructure():
    # Basic fully connected CNN: 
    # - input layer   : 3 nodes 
    # - hidden layer1 : 4 nodes 
    # - output layer  : 2 nodes
    model_1 = Sequential()
    model_1.add(Dense(4, input_shape = (3,), activation = 'relu'))
    model_1.add(Dense(2, activation = 'softmax'))
    model_1.compile(loss='categorical_crossentropy',
                    optimizer = 'adam', 
                    metrics = ['accuracy'])
    

    model_1_answer = Sequential()
    model_1_answer.add(Dense(4, input_shape = (3,), activation = 'relu'))
    model_1_answer.add(Dense(2, activation = 'softmax'))
    model_1_answer.compile(loss='categorical_crossentropy',
    optimizer = 'adam', 
    metrics = ['accuracy'])

    # Get the config of the network created
    model_1_config = model_1.get_config()
    
    del model_1_config["name"]
    for layer in model_1_config["layers"]:
        del layer["config"]["name"]

    model_1_answer_config = model_1_answer.get_config()
    del model_1_answer_config["name"]
    for layer in model_1_answer_config["layers"]:
        del layer["config"]["name"]

    if model_1_answer_config == model_1_config:
        print('Good job! Your model worked')
    else: 
        print('Please check your code again!')


# Example of convolution: tf.nn.conv2d - Toy Example
def IllustrateTensorFlowConv2DWith2DInput():
    # See tensorflow.nn.conv2d example from 
    #    https://stackoverflow.com/questions/42883547/intuitive-understanding-of-1d-2d-and-3d-convolutions-in-convolutional-neural-n
    #
    ones_2d = np.ones((5,5))
    weight_2d = np.ones((3,3))
    strides_2d = [1, 1, 1, 1]

    in_2d = tf.constant(ones_2d, dtype=tf.float32)
    filter_2d = tf.constant(weight_2d, dtype=tf.float32)

    in_width = int(in_2d.shape[0])
    in_height = int(in_2d.shape[1])

    filter_width = int(filter_2d.shape[0])
    filter_height = int(filter_2d.shape[1])

    input_2d   = tf.reshape(in_2d, [1, in_height, in_width, 1])
    kernel_2d = tf.reshape(filter_2d, [filter_height, filter_width, 1, 1])

    output_2d = tf.squeeze(tf.nn.conv2d(input_2d, kernel_2d, strides=strides_2d, padding='SAME'))

# Example of convolution: conv2d - LeNet, VGG, ... for 1 filter
def IllustrateTensorFlowConv2DWith3DInput():
    # See tensorflow.nn.conv2d example from 
    #    https://stackoverflow.com/questions/42883547/intuitive-understanding-of-1d-2d-and-3d-convolutions-in-convolutional-neural-n
    #
    in_channels = 32 # 3 for RGB, 32, 64, 128, ... 
    ones_3d = np.ones((5,5,in_channels)) # input is 3d, in_channels = 32
    # filter must have 3d-shpae with in_channels
    weight_3d = np.ones((3,3,in_channels)) 
    strides_2d = [1, 1, 1, 1]

    in_3d = tf.constant(ones_3d, dtype=tf.float32)
    filter_3d = tf.constant(weight_3d, dtype=tf.float32)

    in_width = int(in_3d.shape[0])
    in_height = int(in_3d.shape[1])

    filter_width = int(filter_3d.shape[0])
    filter_height = int(filter_3d.shape[1])

    input_3d  = tf.reshape(in_3d, [1, in_height, in_width, in_channels])
    kernel_3d = tf.reshape(filter_3d, [filter_height, filter_width, in_channels, 1])

    output_2d = tf.squeeze(tf.nn.conv2d(input_3d, kernel_3d, strides=strides_2d, padding='SAME'))


def IllustrateEdgeDetectingConvolutionKernels(test_data):
    pic1=np.expand_dims(test_data[0],axis=0)  # Picture dimension 1x64x64x3  (1xHxWxC)
    filter_v = np.array( [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                          [[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                          [[1, 0, -1], [2, 0, -2], [1, 0, -1]] ])
    filter_h = np.array( [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                          [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                          [[1, 2, 1], [0, 0, 0], [-1, -2, -1]] ])    

    #------------------------------------------------------------------------------------------------------
    # This is the wrong version. Using np.transpose on all axes reverts horizontal edge detector to vertical,
    # and vice versa.
    #
    ## Add dimension using "expand_dims" to make it 1x3x3x3 (1xCxHxW) [i.e. pytorch style 'NCHW']
    #filter1 = np.transpose(np.expand_dims(filter_v,axis=0))  # transpose is WxHxCx1: this reverts vertical filter to horziontal
    #edge_detect_v = tf.math.abs(tf.nn.conv2d(pic1, filter1, strides=[1,1,1,1], padding='SAME') / 12)
    #
    ## Add dimension using "expand_dims" to make it 1x3x3x3 (1xCxHxW) [i.e. pytorch style 'NCHW']
    #filter2 = np.transpose(np.expand_dims(filter_h,axis=0))  # transpose is WxHxCx1: this reverts horziontal filter to vertical
    #edge_detect_h = tf.math.abs(tf.nn.conv2d(pic1, filter2, strides=[1,1,1,1], padding='SAME') / 12)
    #
    #fig, axs = plt.subplots(ncols=3, figsize = (12, 4))
    #axs[0].imshow(test_data[0])
    #axs[1].imshow(edge_detect_v[0],cmap='gray')
    #axs[2].imshow(edge_detect_h[0],cmap='gray')
    #-----------------------------------------------------------------------------------------------------

    # Add dimension using "expand_dims" to make it 1x3x3x3 (1xCxHxW) [i.e. pytorch style 'NCHW']
    filter1_1 = np.transpose(np.expand_dims(filter_v,axis=0), (2,3,1,0))  # change order to HxWxCx1 (i.e. 'HWCN')
    edge_detect_v_1 = tf.math.abs(tf.nn.conv2d(pic1, filter1_1, strides=[1,1,1,1], padding='SAME', data_format='NHWC') / 12)
    # Add dimension using "expand_dims" to make it 1x3x3x3 (1xCxHxW) [i.e. pytorch style 'NCHW']
    filter2_1 = np.transpose(np.expand_dims(filter_h,axis=0), (2,3,1,0))    # change order to HxWxCx1 (i.e. 'HWCN')
    edge_detect_h_1 = tf.math.abs(tf.nn.conv2d(pic1, filter2_1, strides=[1,1,1,1], padding='SAME', data_format='NHWC') / 12)

    fig, axs = plt.subplots(ncols=3, figsize = (12, 4))
    axs[0].set_title("Original")
    axs[1].set_title("Vertical Edge Detection")
    axs[2].set_title("Horizontal Edge Detection")
    axs[0].imshow(test_data[0])
    axs[1].imshow(edge_detect_v_1[0],cmap='gray')
    axs[2].imshow(edge_detect_h_1[0],cmap='gray')    

def main():
    parser = argparse.ArgumentParser(description='Pneumonia detection from X-ray images.')
    parser.add_argument('--training', dest='is_training', action='store_true',
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()

    
    # Download data file
    if not os.path.isfile(metadata_path):
       runcmd('/opt/homebrew/bin/wget -q --show-progress "' + metadata_url + '"', verbose = True)
    if not os.path.isfile(image_data_path):
       runcmd('/opt/homebrew/bin/wget -q --show-progress "' + image_data_url + '"', verbose = True)
    
    ### pre-loading all data of interest
    _all_data = np.load('image_data.npy')
    # _metadata = pkg.get_metadata(metadata_path, ['train','test','field'])
    _metadata = pkg.get_metadata(metadata_path, ['train','test'])

    # Load training dataset for display purposes (i.e. unflattened)
    train_data, train_labels = pkg.get_train_data(False, _all_data, _metadata, image_shape)
    test_data, test_labels = pkg.get_test_data(False, _all_data, _metadata, image_shape)

    #-----------------------------------------------------------------------------
    # Illustration for Neural Networks: Not important, maybe deleted later.
    #
    # In the example NN that we create in TestingNetworkStructure, we will need to use
    # the flattened training and testing datasets, since it is nota CNN
    #
    # train_data, train_labels = pkg.get_train_data(True, _all_data, _metadata, image_shape)
    # test_data, test_labels = pkg.get_test_data(True, _all_data, _metadata, image_shape)

    TestingNetworkStructure()

    IllustrateTensorFlowConv2DWith2DInput() 
    IllustrateTensorFlowConv2DWith3DInput()

    IllustrateEdgeDetectingConvolutionKernels(test_data)
    #-----------------------------------------------------------------------------



    cnn = models.CNNClassifier(num_hidden_layers = 3, nn_params = nn_params)
    if args.is_training == True:
        # Create a CNN based classifier
        cnn_history = cnn.fit(train_data, train_labels, epochs = 70, validation_data = (test_data, test_labels), shuffle = True, callbacks = [monitor])
        helpers.plot_acc(cnn_history)
    else:
        weight_file ="weights/cp-0062.ckpt"  # Best val_accuracy=0.8875 achieved at epoch=62
        cnn.load_weights(weight_file)

    cnn_score = cnn.evaluate(test_data, test_labels)
    cnn.summary()
    

    # Display info about and give samples from the training dataset
    SummarizeMetaData(_all_data, _metadata,)
    DisplayDataset(train_data, train_labels, image_shape)


    # The following scikit classifiers use flattened images
    train_data, train_labels = pkg.get_train_data(True, _all_data, _metadata, image_shape)
    test_data, test_labels = pkg.get_test_data(True, _all_data, _metadata, image_shape)

    accuracy = [cnn_score[1]]
    algorithm = ["CNN (1-layer)"]

    # Run KNeighbors classifier
    print("Running KNeighbors ...")
    predictions = RunKNeighborsClassifier(train_data, train_labels, test_data)
    accuracy.append(accuracy_score(test_labels, predictions))
    algorithm.append("KNeighbors")

    # Run Logistic Regression
    print("Running LogisticRegression ...")
    predictions = RunLogisticRegression(train_data, train_labels, test_data)
    accuracy.append(accuracy_score(test_labels, predictions))
    algorithm.append("LogisticRegression")

    print("Running DecisionTree ...")
    predictions = RunDecisionTreeClassifier(train_data, train_labels, test_data)
    accuracy.append(accuracy_score(test_labels, predictions))
    algorithm.append("DecisionTree")

    print("Running MLP ...")
    predictions = RunMLPClassifier(train_data, train_labels, test_data)
    accuracy.append(accuracy_score(test_labels, predictions))
    algorithm.append("MLP (Perceptron)")
    
    # Using print and format(.): https://pyformat.info 
    for i in range(len(algorithm)):
        print('{:^20}'.format(algorithm[i]), end="")
    print()

    for i in range(len(accuracy)):
        print('{:^20.5f}'.format(accuracy[i]), end="")
    print()

    # Show the plots created before
    plt.show()
        


if __name__ == "__main__":
    main()