import os
import subprocess
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
  #### PLOTTING
  def plot_one_image(data, labels = [], index = None, image_shape = [64,64,3]):
    '''
    if data is a single image, display that image

    if data is a 4d stack of images, display that image
    '''
    num_dims   = len(data.shape)
    num_labels = len(labels)

    # reshape data if necessary
    if num_dims == 1:
      data = data.reshape(target_shape)
    if num_dims == 2:
      data = data.reshape(np.vstack[-1, image_shape])
    num_dims   = len(data.shape)

    # check if single or multiple images
    if num_dims == 3:
      if num_labels > 1:
        print('Multiple labels does not make sense for single image.')
        return

      label = labels      
      if num_labels == 0:
        label = ''
      image = data

    if num_dims == 4:
      image = data[index, :]
      label = labels[index]

    # plot image of interest
    print('Label: %s'%label)
    plt.imshow(image)
    plt.show()

### defining project variables
# file variables
image_data_path      = 'image_data.npy'
metadata_path        = 'metadata.csv'
metadata_url         = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/' + metadata_path
image_data_url       = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/' + image_data_path
image_shape          = (64, 64, 3)

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

def main():

    # Download data file
    if not os.path.isfile(metadata_path):
       runcmd('/opt/homebrew/bin/wget -q --show-progress "' + metadata_url + '"', verbose = True)
    if not os.path.isfile(image_data_path):
       runcmd('/opt/homebrew/bin/wget -q --show-progress "' + image_data_url + '"', verbose = True)
    
    ### pre-loading all data of interest
    _all_data = np.load('image_data.npy')
    # _metadata = pkg.get_metadata(metadata_path, ['train','test','field'])
    _metadata = pkg.get_metadata(metadata_path, ['train','test'])

    # Load training dataset for display purposes (i.e. unflatened)
    train_data, train_labels = pkg.get_train_data(False, _all_data, _metadata, image_shape)

    # Display info about and give samples from the training dataset
    SummarizeMetaData(_all_data, _metadata,)
    DisplayDataset(train_data, train_labels, image_shape)

    # Load flattened training and testing datasets
    train_data, train_labels = pkg.get_train_data(True, _all_data, _metadata, image_shape)
    test_data, test_labels = pkg.get_test_data(True, _all_data, _metadata, image_shape)

    accuracy = []
    algorithm = []

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
    
    for i in range(len(algorithm)):
        print('{:^20}'.format(algorithm[i]), end="")
    print()

    for i in range(len(accuracy)):
        print('{:^20}'.format(accuracy[i]), end="")
    print()

    # Show the plots created before
    plt.show()
        


if __name__ == "__main__":
    main()