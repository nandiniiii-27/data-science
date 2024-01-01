import pandas as pd
import numpy as np
import statsmodels.tools.tools as stattools
from sklearn.tree import DecisionTreeClassifier, export_graphviz

file_path = r'C:\Users\Ideapad slim 5\OneDrive\Desktop\data set\Website Data Sets\adult_ch6_training'

adult_tr = pd.read_csv(file_path, delimiter=';', header=0)

# Corrected column names using straight quotation marks
y = adult_tr['Income']
mar_np = np.array(adult_tr['Marital status'])
(mar_cat, mar_cat_dict) = stattools.categorical(mar_np, drop=True, dictnames=True)

mar_cat_pd = pd.DataFrame(mar_cat)

# Removed extra underscore in mar_cat_pd
X = pd.concat((adult_tr[['Cap_Gains_Losses']], mar_cat_pd), axis=1)

X_names = ["Cap_Gains_Losses", "Divorced", "Married", "Never-married", "Separated", "Widowed"]
y_names = ["<=50K", ">50K"]

cart01 = DecisionTreeClassifier(criterion="gini", max_leaf_nodes=5).fit(X, y)

export_graphviz(cart01, feature_names=X_names, class_names=y_names)

# This will generate the DOT format representation of the tree, which you can print or visualize.
