import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.outliers_influence as inf
cereals = pd.read_csv(r'C:\Users\Ideapad slim 5\OneDrive\Desktop\data set\Website Data Sets\cereals.CSV')
X = pd.DataFrame(cereals[['Sugars', 'Fiber', 'Potass']])
pd.plotting.scatter_matrix(X)


X = X.dropna()
X = sm.add_constant(X)
vif = [inf.variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
vif_results = pd.DataFrame({'Variable': X.columns[1:], 'VIF': vif})

print(vif_results)
