from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
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