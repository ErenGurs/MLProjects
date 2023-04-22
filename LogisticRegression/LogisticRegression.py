import pandas as pd
import os
import subprocess

import numpy as np    # Great for lists (arrays) of numbers
from sklearn.metrics import accuracy_score   # Great for creating quick ML models

import seaborn as sns
import matplotlib.pyplot as plt 

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

def boundary_classifier(target_boundary, radius_mean_series):
  result = [] #fill this in with predictions!
  # YOUR CODE HERE
  for i in radius_mean_series:
    if i > target_boundary:
      result.append(1)
    else:
      result.append(0)
      
  return result


def main():
    data_path  = 'cancer.csv'

    if not os.path.isfile(data_path):
        runcmd('/opt/homebrew/bin/wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%202b%20-%20Logistic%20Regression/cancer.csv"', verbose = True)
        # Change the diagnosis fields: 'B' -> 0 and 'M' -> 1
        data = pd.read_csv('cancer.csv')
        data['diagnosis'].replace({'M':1, 'B':0}, inplace = True)
        data.to_csv('cancer.csv')
        del data

    # Use the 'pd.read_csv('file')' function to read in read our data and store it in a variable called 'dataframe'
    dataframe = pd.read_csv(data_path)

    dataframe = dataframe[['diagnosis', 'perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean']]
    dataframe['diagnosis_cat'] = dataframe['diagnosis'].astype('category').map({1: '1 (malignant)', 0: '0 (benign)'})

    # Show the first few rows
    dataframe.head()

    # Show data type (int, float etc) of each column
    dataframe.info()

    sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', data = dataframe, order=['1 (malignant)', '0 (benign)'])
    dataframe.head()

    
    chosen_boundary = 15 #Try changing this!

    # Add the predicted value (B=0 or M=1) as a new column 'predicted' to dataframe
    y_pred = boundary_classifier(chosen_boundary, dataframe['radius_mean'])
    dataframe['predicted'] = y_pred

    #y_true = dataframe['diagnosis']

    sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', hue = 'predicted', data = dataframe, order=['1 (malignant)', '0 (benign)'])
    plt.plot([chosen_boundary, chosen_boundary], [-.2, 1.2], 'g', linewidth = 2)

    plt.show()




if __name__ == "__main__":
    main()
