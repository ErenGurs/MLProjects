#@title Run this to import libraries and your data! { display-mode: "form" }
import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv). 
import os # Good for navigating your computer's files 
import subprocess

# Quiet deprecation warnings
import warnings
warnings.filterwarnings("ignore")

# pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

# pip install scikit-learn
# pip install sklearn
# pip install scipy
# Note: Even after installing the packages above, I continued to get the error: "liblapack.3.dylib' (no such file)"
# Only after doing the force-reinstall on Macbook (M1) it worked (see https://developer.apple.com/forums/thread/693696)
# pip install --upgrade --force-reinstall scikit-learn
import sklearn
from sklearn.linear_model import LinearRegression
import numpy as np    # Great for lists (arrays) of numbers

# Dataset is hosted on Google Cloud. Here's how we can grab it:
#!wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%202a%20-%20Linear%20Regression/car_dekho.csv"

# Code from https://www.datacamp.com/tutorial/python-subprocess
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

def LinearRegressionModel2D(car_data):

    # Each dot is a single example (row) from the datafrwhatame, with its 
    # x-value as `Age` and its y-value as `Selling_Price`

    X = car_data[['Age', 'Kms_Driven', "Fuel_Type"]]
    y = car_data[['Selling_Price']] 

    fig, axs = plt.subplots(ncols=3, figsize = (8, 7))
    for ax in fig.get_axes():
        ax.label_outer()

    sns.scatterplot(x = 'Age', y = 'Selling_Price', data = car_data, ax=axs[0])
    linear1 = LinearRegression()
    linear1.fit(X[['Age']], y)
    y1_pred = linear1.predict(X[['Age']])
    axs[0].plot(X['Age'], y1_pred, color='red')

    sns.scatterplot(x = 'Kms_Driven', y = 'Selling_Price', data = car_data, ax=axs[1])
    linear2 = LinearRegression()
    linear2.fit(X[['Kms_Driven']], y)
    y2_pred = linear2.predict(X[['Kms_Driven']])
    axs[1].plot(X['Kms_Driven'], y2_pred, color='red')

    # catplot is figure-level, so it can't be used with subplot. Using swarmplot instead
    # sns.catplot(x = 'Fuel_Type', y = 'Selling_Price', data = car_data, kind = 'swarm', s = 2, ax=axs[1])
    sns.swarmplot(x = 'Fuel_Type', y = 'Selling_Price', data = car_data, ax=axs[2])
    # TO DO: We can try linear regression on categories later
    #linear3 = LinearRegression()
    #linear3.fit(X['Fuel_Type'], y)

    #plt.show()

def LinearRegressionModel3D(car_data):

    # Seaborn 3D plot example is inspired from https://www.educba.com/seaborn-3d-plot/
    plt.figure (figsize = (8, 7))
    seaborn_plot = plt.axes (projection='3d')
    seaborn_plot.scatter3D(car_data['Age'], car_data['Kms_Driven']/1000, car_data['Selling_Price'])
    seaborn_plot.set_xlabel ('Age')
    seaborn_plot.set_ylabel ('Kms (x1000)')
    seaborn_plot.set_zlabel ('Price (lakh)')

def main():
    if not os.path.isfile('car_dekho.csv'):
        runcmd('/opt/homebrew/bin/wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%202a%20-%20Linear%20Regression/car_dekho.csv"', verbose = True)

    # This didn't work:
    # subprocess.run(["/opt/homebrew/bin/wget", "-q --show-progress https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%202a%20-%20Linear%20Regression/car_dekho.csv"], capture_output=True, text=True)
    
    # read our data in using 'pd.read_csv('file')'
    data_path  = 'car_dekho.csv'
    car_data = pd.read_csv(data_path)


    # Before using seaborn, I have to install it. Then I realized I was by default using conda' base environemnet in zsh.
    # And I was installing it under conda's "base" environment (see conda env list). But VS Code was not using that environemnt.
    # So I had to install seaborn to the python root environemnt:
    #   1. Upgrade pip : 
    #       $ usr/local/bin/python3 -m pip install --upgrade pip
    #   2. Install seaborn:
    #       $ pip install seaborn

    LinearRegressionModel2D(car_data)

    LinearRegressionModel3D(car_data)
    plt.show()

    print ("Any key to exit")
    os.system('read')


if __name__ == "__main__":
    main()
