import pandas as pd

file_path = r'C:\Users\Ideapad slim 5\OneDrive\Desktop\data set\Website Data Sets\bank-additional.csv'
bank_train = pd.read_csv(file_path, delimiter=';', header=0)

# Check the column names
print(bank_train.columns)

# Replace 'response' with the actual column name for the target variable
# If you don't have a 'response' column, replace it with the correct column name
target_variable = 'job'

bank_train['age_binned'] = pd.cut(x=bank_train['age'], bins=[0, 27, 60.01, 100], labels=["Under 27", "27 to 60", "Over 60"], right=False)

crosstab_02 = pd.crosstab(bank_train['age_binned'], bank_train[target_variable])
crosstab_02.plot(kind='bar', stacked=True, title='Bar Graph of Age (Binned) with Response Overlay')
