from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

dataset = load_iris()
print(dataset.keys())
# print(dataset['DESCR'])
print(dataset['target_names'])
print(dataset['feature_names'])
df = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
print(df.head())
scat_mtrx = pd.plotting.scatter_matrix(df,
                                       c=dataset['target'],
                                       figsize=(10, 10),
                                       hist_kwds={
                                           'bins': 20
                                       },
                                       s=40,
                                       alpha=0.8)
plt.show()
df_simple = pd.DataFrame(dataset.data[:, 2:4], columns=dataset['feature_names'][2:4])
scat_mtrx = pd.plotting.scatter_matrix(df_simple,
                                       c=dataset['target'],
                                       figsize=(10, 10),
                                       hist_kwds={
                                           'bins': 20
                                       },
                                       s=40,
                                       alpha=0.8)
plt.show()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(dataset.data[:, 2:4],
                                                      dataset.target, random_state=0)
print(X_train.shape, X_valid.shape)
import numpy as np

X_train_concat = np.concatenate((X_train, y_train.reshape(112, 1)), axis=1)
X_valid_concat = np.concatenate((X_valid, y_valid.reshape(38, 1)), axis=1)

print(pd.DataFrame(X_train_concat).head())

import math


def euclidean_distance(data1, data2):
    distance = 0
    for i in range(len(data1) - 1):
        distance += (data1[i] - data2[i]) ** 2
    return math.sqrt(distance)


def get_neighbors(train, test, k=1):
    distances = [(train[i][-1], euclidean_distance(train[i], test))
                 for i in range(len(train))]
    distances.sort(key=lambda elem: elem[1])

    neighbors = [distances[i][0] for i in range(k)]
    return neighbors


def prediction(neighbors):
    count = {}
    for instance in neighbors:
        if instance in count:
            count[instance] += 1
        else:
            count[instance] = 1
    target = max(count.items(), key=lambda x: x[1])[0]
    return target


def accuracy(test, test_prediction):
    correct = 0
    for i in range(len(test)):
        if test[i][-1] == test_prediction[i]:
            correct += 1
    return (correct / len(test))


predictions = []
for x in range(len(X_valid_concat)):
    neighbors = get_neighbors(X_train_concat, X_valid_concat[x], k=5)
    result = prediction(neighbors)
    predictions.append(result)
#     print(f'predicted = {result}, actual = {x_test_concat[x][-1]}') # если есть интерес посмотреть, какие конкретно прогнозы некорректны
accuracy = accuracy(X_valid_concat, predictions)
print(f'Accuracy: {accuracy}')
