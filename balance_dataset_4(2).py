import pandas as pd
import matplotlib.pyplot as plt

# Replace the file path with the actual path to your dataset file
file_path = r'C:\Users\Ideapad slim 5\OneDrive\Desktop\data set\Website Data Sets\bank-additional.csv'

# Load the dataset into a DataFrame with the correct delimiter
# Assuming your data has a header row, set header=0
bank_train = pd.read_csv(file_path, delimiter=';', header=0)

# Check the column names
print(bank_train.columns)

# Check the count of each class in the 'y' column
response_counts = bank_train['y'].value_counts()
print(response_counts)

# Resample the 'yes' class to balance the dataset
to_resample = bank_train.loc[bank_train['y'] == "yes"]
our_resample = to_resample.sample(n=841, replace=True)

# Concatenate the resampled data with the original dataset
bank_train_rebal = pd.concat([bank_train, our_resample])

# Check the count of each class after resampling
response_counts_rebal = bank_train_rebal['y'].value_counts()
print(response_counts_rebal)
