from scipy.special import expit
import csv
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

def read_file(filename):
    data = []
    with open('/Users/wenbo/Documents/TopicML/hw2/hw2-data/%s' % filename) as file:
        content = csv.reader(file)
        for c in content:
            data.append(c)
    return np.matrix(data).astype(np.float32)


X_train = read_file('X_train.csv')
X_test = read_file('X_test.csv')
Y_train = read_file('y_train.csv')
Y_test = read_file('y_test.csv')

X_train2 = X_train.copy()
X_test2 = X_test.copy()
Y_train2 = Y_train.copy()
Y_test2 = Y_test.copy()


X_train2 = np.column_stack((X_train2, np.ones(X_train2.shape[0])))
X_test2 = np.column_stack((X_test2, np.ones(X_test2.shape[0])))


Y_train2[Y_train2 == 0] = -1
Y_test2[Y_test2 == 0] = -1


weights = np.zeros(X_train2.shape[1]).reshape(-1, 1)




iterations = 10000  # 00
objective = []
for t in range(1, iterations + 1):
    # print('t=%d'%t)
    e = 1 / (1e5 * np.sqrt(t + 1))

    sigm = expit(np.multiply(Y_train2, X_train2.dot(weights)))


    objective.append(np.sum(np.log(sigm + 1e-10)))


    update = X_train2.T.dot(np.multiply(Y_train2, 1 - sigm))


    weights += e * update

    # print(weights[:10])

x = list(range(1, 10001))
plt.plot(x, objective)
plt.show()
