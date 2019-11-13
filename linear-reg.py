import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
boston_df = load_boston()

bostonDF = pd.DataFrame(data=np.c_[boston_df['data'], boston_df['target']], columns=list(boston_df['feature_names']) +
                                                                                                   ['MEDV'])
columns_names = boston_df.feature_names
y = boston_df.target
X = boston_df.data


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)

linear = LinearRegression()
linear.fit(X_train, y_train)

# print(f"Intercept: {linear.intercept_}\n")
# print(f"Coeficients: {linear.coef_}\n")
# print(f"Named Coeficients: {pd.DataFrame(linear.coef_, columns_names)}")

prediction = linear.predict(X_test)

for (real, predicted) in list(zip(y_test, prediction)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")

sns.set(palette="inferno")

sns.scatterplot(y_test, prediction)
plt.plot([0, 50], [0, 50], '--')
plt.title('(linear regression)')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.show()

residuals = y_test - prediction
sns.scatterplot(y_test, residuals)
plt.plot([50, 0], [0, 0], '--')
plt.title('(linear regression)')
plt.xlabel('Real Value')
plt.ylabel('Residual (difference)')
plt.show()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Residual (difference) Distribution (linear regression)')
plt.show()


print(f"Printing MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, prediction)}")
print(f"Printing MSE error: {metrics.mean_squared_error(y_test, prediction)}")
print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, prediction))}")
print(f"Printing r2 score linear regression: {metrics.r2_score(y_test, prediction)}")

#########################################################################

kreg = KNeighborsRegressor()
kreg.fit(X_train, y_train)

# print(f"Intercept2: {linear.intercept_}\n")
# print(f"Coeficients2: {linear.coef_}\n")
# print(f"Named Coeficients2: {pd.DataFrame(linear.coef_, columns_names)}")

prediction2 = kreg.predict(X_test)


for (real, predicted) in list(zip(y_test, prediction2)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")

sns.set(palette="inferno")

sns.scatterplot(y_test, prediction2)
plt.plot([0, 50], [0, 50], '--')
plt.title('(KNeighbors)')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.show()

residuals = y_test - prediction2
sns.scatterplot(y_test, residuals)
plt.plot([50, 0], [0, 0], '--')
plt.title('(KNeighbors)')
plt.xlabel('Real Value')
plt.ylabel('Residual (difference)')
plt.show()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Residual (difference) Distribution (KNeighbors)')
plt.show()


print(f"Printing MAE error(avg abs residual)2: {metrics.mean_absolute_error(y_test, prediction2)}")
print(f"Printing MSE error2: {metrics.mean_squared_error(y_test, prediction2)}")
print(f"Printing RMSE error2: {np.sqrt(metrics.mean_squared_error(y_test, prediction2))}")
print(f"Printing r2 score KNeighborsRegression: {metrics.r2_score(y_test, prediction2)}")

for n_neighbors in range(1, 20):
    clf = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"r2 score for KNeighbors(n_neighbors={n_neighbors}): {metrics.r2_score(y_test, y_pred)}")

###################################################################################################

GRegression = GradientBoostingRegressor()
GRegression.fit(X_train, y_train)
prediction3 = GRegression.predict(X_test)

for (real, predicted) in list(zip(y_test, prediction3)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")

sns.set(palette="inferno")

sns.scatterplot(y_test, prediction3)
plt.plot([0, 50], [0, 50], '--')
plt.title('(gradient regressor)')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.show()

residuals = y_test - prediction3
sns.scatterplot(y_test, residuals)
plt.plot([50, 0], [0, 0], '--')
plt.title('(gradient regressor)')
plt.xlabel('Real Value')
plt.ylabel('Residual (difference)')
plt.show()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Residual (difference) Distribution (gradient regressor)')
plt.show()

print(f"Printing MAE error(avg abs residual)2: {metrics.mean_absolute_error(y_test, prediction3)}")
print(f"Printing MSE error2: {metrics.mean_squared_error(y_test, prediction3)}")
print(f"Printing RMSE error2: {np.sqrt(metrics.mean_squared_error(y_test, prediction3))}")
print(f"Printing r2 score GradientBoostingRegressor: {metrics.r2_score(y_test, prediction3)}")

###################################################################################################


