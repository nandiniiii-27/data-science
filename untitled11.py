import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

# Load the training dataset
wine_train = pd.read_csv(r"C:\Users\Ideapad slim 5\OneDrive\Desktop\data set\Website Data Sets\wine_flag_training.csv")
wine_test = pd.read_csv(r"C:\Users\Ideapad slim 5\OneDrive\Desktop\data set\Website Data Sets\wine_flag_test.csv", encoding='utf-8')

# Select features for clustering
X = wine_train[['Alcohol_flag', 'Sugar_flag']]

# Convert columns to numeric data types
X['Alcohol_flag'] = pd.to_numeric(X['Alcohol_flag'], errors='coerce')
X['Sugar_flag'] = pd.to_numeric(X['Sugar_flag'], errors='coerce')

# Drop rows with missing values
X = X.dropna()

# Standardize the features using z-score
Xz = pd.DataFrame(stats.zscore(X), columns=['Alcohol_flag', 'Sugar_flag'])

# Apply k-Means clustering with 2 clusters
kmeans01 = KMeans(n_clusters=2).fit(Xz)

# Assign cluster labels to the training dataset
cluster = kmeans01.labels_
Cluster1 = Xz.loc[cluster == 0]
Cluster2 = Xz.loc[cluster == 1]

# Display statistics for each cluster
print("Cluster 1 Statistics:")
print(Cluster1.describe())

print("\nCluster 2 Statistics:")
print(Cluster2.describe())

# Load the test dataset
wine_test = pd.read_csv(r"C:\Users\Ideapad slim 5\OneDrive\Desktop\data set\Website Data Sets\white_wine_test.csv")  # Replace with the correct path
X_test = wine_test[['Alcohol_flag', 'Sugar_flag']]

# Convert columns to numeric data types in the test dataset
X_test['Alcohol_flag'] = pd.to_numeric(X_test['Alcohol_flag'], errors='coerce')
X_test['Sugar_flag'] = pd.to_numeric(X_test['Sugar_flag'], errors='coerce')

# Drop rows with missing values in the test dataset
X_test = X_test.dropna()

# Standardize the features for the test dataset
Xz_test = pd.DataFrame(stats.zscore(X_test), columns=['Alcohol_flag', 'Sugar_flag'])

# Apply the previously fitted k-Means model to the test dataset
cluster_test = kmeans01.predict(Xz_test)

# Assign test dataset to clusters
Cluster1_test = Xz_test.loc[cluster_test == 0]
Cluster2_test = Xz_test.loc[cluster_test == 1]

# Display statistics for each cluster in the test dataset
print("\nCluster 1 Test Statistics:")
print(Cluster1_test.describe())

print("\nCluster 2 Test Statistics:")
print(Cluster2_test.describe())
