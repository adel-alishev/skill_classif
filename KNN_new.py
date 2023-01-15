from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
'''-----------------загрузка данных-----------------'''
dataset = load_iris()
# print(dataset.keys())
# # print(dataset['DESCR'])
# print(dataset['target_names'])
# print(dataset['feature_names'])
'''-----------------просмотр матрицы данных-----------------'''
df = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
scat_mtrx = pd.plotting.scatter_matrix(df,
                                       c=dataset['target'],
                                       figsize=(10, 10),
                                       hist_kwds={'bins': 20},
                                       s=40,
                                       alpha=0.8)
plt.show()
'''-----------------сепарирование данных-----------------'''
df_simple = pd.DataFrame(dataset.data[:, 2:4], columns=dataset['feature_names'][2:4])
scat_mtrx = pd.plotting.scatter_matrix(df_simple,
                                       c=dataset['target'],
                                       figsize=(10, 10),
                                       hist_kwds={'bins': 20},
                                       s=40,
                                       alpha=0.8)
plt.show()

'''-----------------разбиение на выборки-----------------'''
X_train, X_valid, y_train, y_valid = train_test_split(dataset.data[:, 2:4],
                                                      dataset.target, random_state=0)
print(X_train.shape, X_valid.shape)
'''-----------------использование библиотечный KNN-----------------'''
knn = KNeighborsClassifier(n_neighbors=5)
knn_model = knn.fit(X_train, y_train)
knn_pred = knn.predict(X_valid)
'''-----------------проверка качества-----------------'''
accuracy = accuracy_score(y_valid, knn_pred)
print(f'Accuracy: {accuracy}')