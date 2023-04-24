import pandas as pd
import os
import subprocess

import numpy as np    # Great for lists (arrays) of numbers
from sklearn.metrics import accuracy_score   # Great for creating quick ML models
from sklearn.model_selection import train_test_split  # Split data into test and training

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model

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

def BoundaryClassifier(dataframe, boundary):

    # Add the predicted value (B=0 or M=1) as a new column 'predicted' to dataframe
    y_pred = boundary_classifier(boundary, dataframe['radius_mean'])
    dataframe['predicted'] = y_pred

    y_true = dataframe['diagnosis'] # the actual diagnosis from dataframe

    sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', hue = 'predicted', data = dataframe, order=['1 (malignant)', '0 (benign)'])
    plt.plot([boundary, boundary], [-.2, 1.2], 'g', linewidth = 2)
    plt.title("Prediction accuracy = %f"%accuracy_score(y_true,y_pred))


def LogisticRegression1D(dataframe):
    train_df, test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1)

    print('Number of rows in training dataframe:', train_df.shape[0])
    train_df.head()
    print('Number of rows in test dataframe:', test_df.shape[0])
    test_df.head()

    labelx = ['radius_mean']
    labely = 'diagnosis'
    X_train = train_df[labelx]  # column vector of size (N,1)
    y_train = train_df[labely]  # 1D array of size (N,)
    # Here, we create a 'reg' object that handles the line fitting for us!
    logreg_model = linear_model.LogisticRegression()
    logreg_model.fit(X_train, y_train)

    X_test = test_df[labelx]  # radius (from test dataset)
    y_test = test_df[labely]  # actual diagnosis (from test dataset) 
    y_pred=logreg_model.predict(X_test)  # estimated diagnosis of the test data set

    test_df['predicted'] = y_pred.squeeze()
    sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', hue = 'predicted', data=test_df, order=['1 (malignant)', '0 (benign)'])
    #sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', data=test_df, order=['1 (malignant)', '0 (benign)'])
    plt.title("Prediction accuracy = %f"%accuracy_score(y_test,y_pred))

    # Let's visualize the probabilities for `X_test`
    plt.figure("Logistic Regression Model")
    y_prob = logreg_model.predict_proba(X_test)
    X_test_view = X_test[labelx].values.squeeze()
    plt.xlabel('radius_mean')
    plt.ylabel('Predicted Probability')
    plt.title("Actual diagnosis (0: benign, 1: malignant)")
    #sns.scatterplot(x = X_test_view, y = y_prob[:,1], hue = y_test, palette=['purple','green'])
    sns.scatterplot(x = X_test_view, y = y_prob[:,1], hue = y_test)

def main():
    data_path  = 'cancer.csv'

    if not os.path.isfile(data_path):
        runcmd('/opt/homebrew/bin/wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%202b%20-%20Logistic%20Regression/cancer.csv"', verbose = True)
        # Change the diagnosis fields: 'B' -> 0 and 'M' -> 1
        data = pd.read_csv(data_path)
        data['diagnosis'].replace({'M':1, 'B':0}, inplace = True)
        data.to_csv(data_path)
        del data

    # Use the 'pd.read_csv('file')' function to read in read our data and store it in a variable called 'dataframe'
    dataframe = pd.read_csv(data_path)

    dataframe = dataframe[['diagnosis', 'perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean']]
    dataframe['diagnosis_cat'] = dataframe['diagnosis'].astype('category').map({1: '1 (malignant)', 0: '0 (benign)'})

    # Show the first few rows
    #dataframe.head()

    # Show data type (int, float etc) of each column
    #dataframe.info()

    sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', data = dataframe, order=['1 (malignant)', '0 (benign)'])

    # Simple/Manual boundary classifier (for logistic regression)
    BoundaryClassifier(dataframe, boundary=15)

    # Single dimension (radius) ML-based boundary classifier (for logistic regression)
    LogisticRegression1D(dataframe)

    plt.show()




if __name__ == "__main__":
    main()
