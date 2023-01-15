from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = load_wine()
df = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
# scat_mtrx = pd.plotting.scatter_matrix(df,
#                                        c=dataset['target'],
#                                        figsize=(25, 25),
#                                        hist_kwds={'bins': 20},
#                                        s=40,
#                                        alpha=0.8)
# plt.savefig('image.jpg')
# plt.show()
#
# df_simple = pd.DataFrame(dataset.data[:, 11:], columns=dataset['feature_names'][11:])
# scat_mtrx = pd.plotting.scatter_matrix(df_simple,
#                                        c=dataset['target'],
#                                        figsize=(10, 10),
#                                        hist_kwds={'bins': 20},
#                                        s=120,
#                                        alpha=0.8)
# plt.show()

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(dataset.data[:, 11:],
                                                      dataset.target, random_state=17)
print(X_train.shape, X_valid.shape)
knn = GaussianNB()
knn_model = knn.fit(X_train, y_train)
knn_pred = knn.predict(X_valid)
gnb_pred = knn.predict_proba(X_valid[11].reshape(1,-1))
print(gnb_pred)
print(y_valid[11])
'''-----------------проверка качества-----------------'''
accuracy = accuracy_score(y_valid, knn_pred)
print(f'Accuracy: {accuracy}')