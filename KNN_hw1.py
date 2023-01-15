from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

iris_dataset = load_iris()
iris_dataset['feature_names']
print(iris_dataset['feature_names'])
new1 = np.delete(iris_dataset.data, 1, axis=1)
new1_c = np.delete(iris_dataset['feature_names'], 1, axis=0)

df1 = pd.DataFrame(new1, columns=new1_c)
# print(df1)
new2 = np.delete(iris_dataset.data, 0, axis=1)
new2_c = np.delete(iris_dataset['feature_names'], 0, axis=0)

df2 = pd.DataFrame(new2, columns=new2_c)
# print(df2)

ax = plt.axes(projection='3d')

zdata = df1['sepal length (cm)']# точки оси Z
xdata = df1['petal length (cm)'] # точки оси X
ydata = df1['petal width (cm)'] # точки оси Y
colors = c=iris_dataset.target

ax.scatter3D(xdata, ydata, zdata, alpha=.8, c=colors)
plt.show()

ax = plt.axes(projection='3d')

zdata = df2['sepal width (cm)']# точки оси Z
xdata = df2['petal length (cm)'] # точки оси X
ydata = df2['petal width (cm)'] # точки оси Y
colors = c=iris_dataset.target

ax.scatter3D(xdata, ydata, zdata, alpha=.8, c=colors)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(df1, iris_dataset.target, random_state=17)
#
# x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(df2, iris_dataset.target, random_state=17)


# knn1 = KNeighborsClassifier(n_neighbors=5)
# knn_model1 = knn1.fit(x_train_1, y_train_1)
# knn_pred1 = knn1.predict(x_test_1)
#
# knn2 = KNeighborsClassifier(n_neighbors=3)
# knn_model2 = knn2.fit(x_train_2, y_train_2)
# knn_pred2 = knn2.predict(x_test_2)
#
#
#
# accuracy_1 = accuracy_score(y_test_1, knn_pred1)
# accuracy_2 = accuracy_score(y_test_2, knn_pred2)
# # print(f'Accuracy_1: {accuracy_1}')
# print(f'Accuracy_1: {accuracy_1}, accuracy_2: {accuracy_2}')
acc1 = []
acc2 = []
x_i = []
for i in range(1,21, 1):
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(df1, iris_dataset.target, random_state=17)

    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(df2, iris_dataset.target, random_state=17)

    knn1 = KNeighborsClassifier(n_neighbors=i)
    knn_model1 = knn1.fit(x_train_1, y_train_1)
    knn_pred1 = knn1.predict(x_test_1)

    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn_model2 = knn2.fit(x_train_2, y_train_2)
    knn_pred2 = knn2.predict(x_test_2)

    accuracy_1 = accuracy_score(y_test_1, knn_pred1)
    accuracy_2 = accuracy_score(y_test_2, knn_pred2)
    x_i.append(i)
    acc1.append(accuracy_1)
    acc2.append(accuracy_2)
    print(f'KNN: n_neighbors: {i}, Accuracy_1: {accuracy_1}, accuracy_2: {accuracy_2}')

plt.plot(x_i, acc1, label = 'acc1')
plt.plot(x_i, acc2, label = 'acc2')
plt.show()
