#@title Run this to import libraries and your data! { display-mode: "form" }
import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv). 
import os # Good for navigating your computer's files 
import subprocess

# Quiet deprecation warnings
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt

# Our dataset is hosted on Google Cloud. Here's how we can grab it:
#!wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%202a%20-%20Linear%20Regression/car_dekho.csv"

print ("Eren: ", __name__)


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


def main():
    if not os.path.isfile('car_dekho.csv'):
        runcmd('/opt/homebrew/bin/wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%202a%20-%20Linear%20Regression/car_dekho.csv"', verbose = True)

    # This didn't work:
    # subprocess.run(["/opt/homebrew/bin/wget", "-q --show-progress https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%202a%20-%20Linear%20Regression/car_dekho.csv"], capture_output=True, text=True)
    
    # read our data in using 'pd.read_csv('file')'
    data_path  = 'car_dekho.csv'
    car_data = pd.read_csv(data_path)

    # let's look at our 'dataframe'. Dataframes are just like google or excel spreadsheets. 
    # use the 'head' method to show the first five rows of the table as well as their names. 
    car_data.head(100) 


    # Before using seaborn, I have to install it. Then I realized I was by default using conda' base environemnet in zsh.
    # And I was installing it under conda's "base" environment (see conda env list). But VS Code was not using that environemnt.
    # So I had to install seaborn to the python root environemnt:
    #   1. Upgrade pip : 
    #       $ usr/local/bin/python3 -m pip install --upgrade pip
    #   2. Install seaborn:
    #       $ python3 -m pip install seaborn


    # Each dot is a single example (row) from the dataframe, with its 
    # x-value as `Age` and its y-value as `Selling_Price`
    sns.scatterplot(x = 'Age', y = 'Selling_Price', data = car_data)
    sns.catplot(x = 'Fuel_Type', y = 'Selling_Price', data = car_data, kind = 'swarm', s = 2)
    plt.show()

if __name__ == "__main__":
    main()