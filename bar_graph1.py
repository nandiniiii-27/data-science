import pandas as pd
import matplotlib.pyplot as plt

# Replace the file path with the actual path to your dataset file
file_path = r'C:\Users\Ideapad slim 5\OneDrive\Desktop\data set\Website Data Sets\bank-additional.csv'

# Load the dataset into a DataFrame with the correct delimiter
# Assuming your data has a header row, set header=0
bank_train = pd.read_csv(file_path, delimiter=';', header=0)

# Choose the appropriate columns from your dataset
# For example, assuming you want to use 'education' and 'y' (response) columns
crosstab_01 = pd.crosstab(bank_train['education'], bank_train['y'])

# Plot stacked bar graph
crosstab_01.plot(kind='bar', stacked=True)
plt.title('Stacked Bar Graph for Cross-Tabulation')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.show()

# Normalize the cross-tabulation table
crosstab_norm = crosstab_01.div(crosstab_01.sum(1), axis=0)

# Plot normalized stacked bar graph
crosstab_norm.plot(kind='bar', stacked=True)
plt.title('Normalized Stacked Bar Graph for Cross-Tabulation')
plt.xlabel('Education Level')
plt.ylabel('Proportion')
plt.show()


print(bank_train.columns)
