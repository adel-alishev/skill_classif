import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
df[['Cabin', 'Embarked']] = df[['Cabin', 'Embarked']].fillna('U')

df1 = df.groupby('Sex', group_keys=True)['Age'].apply(lambda x: x.fillna(x.median())).reset_index()

df1.columns = ['Sex', 'index', 'Age']
df1.sort_values(by=['index'], inplace=True)
df1 = pd.DataFrame.set_index(df1, ['index'])
df.update(df1)
# print(df.isna().sum())
df4 = df[(df['PassengerId'] <= 20) & (df['PassengerId'] >= 6)][['PassengerId', 'Age']]
# print(df4.isnull().sum())

df['Sex'].replace({'male': 1, 'female': 0}, inplace=True)

df['Embarked'].replace({'U': 0,
                        'S': 1,
                        'C': 2,
                        'Q': 3}, inplace=True)
# print(df[['Sex','Embarked']].head())
print(df.isna().sum())
from sklearn.model_selection import train_test_split
target = df['Survived']
dataset = df.drop(['Survived', 'Name', 'Ticket','Cabin'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(dataset,
                                                      target, random_state=17, test_size=0.25)
print(X_train.shape, X_valid.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

knn = KNeighborsClassifier(n_neighbors=5)
knn_model = knn.fit(X_train, y_train)

gnb = GaussianNB()
gnb_model = gnb.fit(X_train, y_train)

dtc = DecisionTreeClassifier()
dtc_model = dtc.fit(X_train, y_train)

# reg = LogisticRegression()
# reg_model = reg.fit(X_train, y_train)