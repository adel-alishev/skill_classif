from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz

dataset = load_wine()
df = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])

X_train, X_valid, y_train, y_valid = train_test_split(dataset.data[:, 11:],
                                                      dataset.target, random_state=17)
print(X_train.shape, X_valid.shape)

dtc = DecisionTreeClassifier()
dtc_model = dtc.fit(X_train, y_train)


def print_graph(data):
    dot_data = tree.export_graphviz(data, out_file=None,
                                    feature_names=dataset.feature_names[11:],
                                    class_names=dataset.target_names,
                                    filled=True)
    graph = graphviz.Source(dot_data)
    return graph.render('dtree',view=True)

print_graph(dtc_model)

dtc_predictions = dtc.predict(X_valid)
accuracy = dtc.score(X_valid, y_valid)
print(f'Accuracy: {accuracy}')

dtc_entrp = DecisionTreeClassifier(criterion='entropy')
dtc_model_entrp = dtc_entrp.fit(X_train, y_train)
dtc_predictions_entrp = dtc_entrp.predict(X_valid)
accuracy = dtc_entrp.score(X_valid, y_valid)
print(f'Accuracy: {accuracy}')

print_graph(dtc_model_entrp)


