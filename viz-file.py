import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d, Axes3D

boston = pd.read_csv(filepath_or_buffer='data/housing.data', sep='\\s+', header=0)
os.makedirs('plots', exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].scatter(boston['CRIM'], boston['AGE'], s=10, marker='*')
axes[0].set_title('CRIM x AGE')
axes[0].set_xlabel('per capita crime rate by town')
axes[0].set_ylabel('proportion of owner-occupied units built prior to 1940')
axes[1].scatter(boston['CRIM'], boston['DIS'], s=10, color='red')
axes[1].set_title('CRIM x DIS')
axes[1].set_xlabel('per capita crime rate by town')
axes[1].set_ylabel('weighted distances to five Boston employment centres')
axes[2].scatter(boston['CRIM'], boston['MEDV'], s=10, color='green')
axes[2].set_title('CRIM x MEDV')
axes[2].set_xlabel('per capita crime rate by town')
axes[2].set_ylabel('Median value of owner-occupied homes in $1000s')
plt.tight_layout()
plt.savefig('plots/1.png')
plt.clf()

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].scatter(boston['NOX'], boston['CRIM'], s=10, marker='*')
axes[0].set_title('NOX x CRIM')
axes[0].set_xlabel('nitric oxides concentration(ppm)')
axes[0].set_ylabel('per capita crime rate by town')
axes[1].scatter(boston['NOX'], boston['AGE'], s=10, color='red')
axes[1].set_title('NOX x AGE')
axes[1].set_xlabel('nitric oxides concentration(ppm)')
axes[1].set_ylabel('proportion of owner-occupied units built prior to 1940')
axes[2].scatter(boston['NOX'], boston['DIS'], s=10, color='green')
axes[2].set_title('NOX x DIS')
axes[2].set_xlabel('nitric oxides concentration(ppm)')
axes[2].set_ylabel('weighted distances to five Boston employment centres')
plt.tight_layout()
plt.savefig('plots/2.png')
plt.clf()

fig, axes = plt.subplots(1, 1, figsize=(10, 5))
axes.grid(axis='y', alpha=0.5)
axes.scatter(boston['NOX'], boston['ZN'], s=10, marker='*', label='proportion of residential land zoned for lots over '
                                                                  '25,000 sq.ft')
axes.scatter(boston['NOX'], boston['AGE'], s=10, marker='o', label='proportion of owner-occupied units built prior to '
                                                                   '1940')
axes.set_title(f'Population concentration around industrial zones')
axes.set_xlabel('nitric oxides concentration(ppm)')
axes.legend()
plt.tight_layout()
plt.savefig(f'plots/3.png', dpi=100)
plt.clf()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(boston['RM'], boston['LSTAT'], s=10, marker='*')
axes[0].set_title('RM x LSTAT')
axes[0].set_xlabel('average number of rooms per dwelling')
axes[0].set_ylabel('% lower status of the population')
axes[1].scatter(boston['RM'], boston['MEDV'], s=10, color='red')
axes[1].set_title('RM x MEDV')
axes[1].set_xlabel('average number of rooms per dwelling')
axes[1].set_ylabel('Median value of owner-occupied homes in $1000s')
plt.tight_layout()
plt.savefig('plots/4.png')
plt.clf()

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].scatter(boston['DIS'], boston['B'], s=10, marker='*')
axes[0].set_title('DIS x B')
#axes[0].set_xlabel('weighted distances to five Boston employment centres')
axes[0].set_ylabel('proportion of blacks by town')
axes[1].scatter(boston['DIS'], boston['LSTAT'], s=10, color='red')
axes[1].set_title('DIS x LSTAT')
axes[1].set_xlabel('weighted distances to five Boston employment centres')
axes[1].set_ylabel('% lower status of the population')
axes[2].scatter(boston['DIS'], boston['MEDV'], s=10, color='green')
axes[2].set_title('DIS x MEDV')
#axes[2].set_xlabel('weighted distances to five Boston employment centres')
axes[2].set_ylabel('Median value of owner-occupied homes in $1000s')
plt.tight_layout()
plt.savefig('plots/5.png')
plt.clf()

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d')
line1 = axes.scatter(boston['INDUS'], boston['DIS'], boston['MEDV'])
#axes.legend()
axes.set_xlabel('proportion of non-retail business')
axes.set_ylabel('distances to employment centres')
axes.set_zlabel('Median value of homes in $1000s')
plt.tight_layout()
plt.savefig('plots/6.png')
plt.clf()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(boston['LSTAT'], boston['AGE'], s=10, color='red')
axes[0].set_title('LSTAT x AGE')
axes[0].set_xlabel('% lower status of the population')
axes[0].set_ylabel('proportion of owner-occupied units built prior to 1940')
axes[1].scatter(boston['LSTAT'], boston['B'], s=10, color='red')
axes[1].set_title('LSTAT x B')
axes[1].set_xlabel('% lower status of the population')
axes[1].set_ylabel('proportion of blacks by town')
plt.tight_layout()
plt.savefig('plots/7.png')
plt.clf()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(boston['AGE'], boston['LSTAT'], s=10, color='red')
axes[0].set_title('AGE x LSTAT')
axes[0].set_ylabel('% lower status of the population')
axes[0].set_xlabel('proportion of owner-occupied units built prior to 1940')
axes[1].scatter(boston['AGE'], boston['B'], s=10, color='red')
axes[1].set_title('AGE x B')
axes[1].set_ylabel('proportion of blacks by town')
axes[1].set_xlabel('proportion of owner-occupied units built prior to 1940')
plt.tight_layout()
plt.savefig('plots/8.png')
plt.clf()

fig, axes = plt.subplots(1, 1, figsize=(10, 5))
axes.scatter(boston['MEDV'], boston['LSTAT'], s=10, color='red')
axes.set_title('MEDV x LSTAT')
axes.set_ylabel('% lower status of the population')
axes.set_xlabel('Median value of owner-occupied homes in $1000s')
plt.tight_layout()
plt.savefig('plots/9.png')
plt.clf()













