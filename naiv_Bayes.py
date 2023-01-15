import numpy as np
import pandas as pd
data = np.array([[400, 350, 450, 500],
                 [0, 150, 300, 300],
                 [30, 180, 100, 200],
                 [430, 680, 850, 1000]])
idx = ['Banana', 'Orange', 'Plum', 'Total']
col = ['Long', 'Sweet', 'Yellow', 'Total']

fruits = pd.DataFrame(data, columns=col, index=idx)
print(fruits.head())

result = {}
for i in range(fruits.values.shape[0] - 1):
    p = 1
    for j in range(fruits.values.shape[1] - 1):
        p *= fruits.values[i, j] / fruits.values[i, -1]
    p *= fruits.values[i, -1] / fruits.values[-1, -1]
    result[fruits.index[i]] = p

print('вероятность, что фрукт будет соответствовать всем трем признакам: ', result)