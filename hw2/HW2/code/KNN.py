from operator import itemgetter
import csv
import numpy as np
import matplotlib.pyplot as plt
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
# K = 1
K = 1
length = {}


def cal_distance(X_train, x_test):
    for i in range(len(X_train)):
        length[i] = np.sum(np.abs(X_train[i] - x_test))
    return length


def predict(X_train, x_test):
    result = cal_distance(X_train, x_test)

    result2 = {}
    result2 = sorted(result.items(), key=itemgetter(1))

    n = [0, 0]
    for i in range(K):
        if Y_train[result2[i][0]] == 0:
            n[0] = n[0] + 1
        elif Y_train[result2[i][0]] == 1:
            n[1] = n[1] + 1

    if n[0] > n[1]:
        num = 0
    elif n[1] > n[0]:
        num = 1
    else:
        num = np.random.randint(2)
    return num


# a,n = predict(X_train, X_test[0])

res = np.zeros((20, len(Y_test)))

for k in range(20):
    K = k + 1
    for i in range(len(Y_test)):
        res[k][i] = predict(X_train, X_test[i])

#print(res)

accuracy = []

for k in range(20):
    acc = 0
    for i in range(len(Y_test)):
        if res[k][i] == Y_test[i]:
            acc = acc + 1
    accuracy.append(acc/93)
print(accuracy)


x = list(range(1,21))
#print(x)
plt.figure(1)
# markerline, stemlines, baseline = plt.stem(x, accuracy[x], '-.')
plt.plot(x, accuracy)
plt.xticks(x)

plt.show()