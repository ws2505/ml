import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import numpy as np

# print(X_train)
# print(Y_train)

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
x = X_test[0]
# print(X_train)
# print(x)

mu = np.zeros((2, 57))
# sigma = np.zeros((2))
# print(np.shape(mu))
n = np.array([0, 0])
# print(np.shape(X_train)[1])
# print(np.shape(X_train))
# print(np.shape(mu))
# X_train2 = np.zeros((4508,57))
X_train2 = X_train.copy()
# print("X_train")
# print(X_train[0])
# print(np.shape(X_train2[0]))
# print(np.shape(X_train2))
# print(np.shape(X_train))
# for i in range(np.shape(X_train)[0]):
#     for j in range(54,57):
#         X_train2[1][5] = np.log(1)
# print(X_train2)
# print(X_train2)
X_train2[:, -3:] = np.apply_along_axis(np.log, 0, X_train2[:, -3:])
# print("X_train2")
# print(X_train2)
# print(X_train2[0])
for i in range(np.shape(X_train)[0]):
    if Y_train[i] == 0:
        n[0] = n[0] + 1
        mu[0] = mu[0] + X_train2[i]
    elif Y_train[i] == 1:
        n[1] = n[1] + 1
        mu[1] = mu[1] + X_train2[i]
mu[0] = mu[0] / n[0]
mu[1] = mu[1] / n[1]

mu[0][-3:] = 1 / mu[0][-3:]
mu[1][-3:] = 1 / mu[1][-3:]


# print("mu1")
# #print(n[0])
# print(mu[1])

def fx(x, mu):
    # initialize to class priors
    pi = np.mean(Y_train)
    result = [1 - pi, pi]
    tr_bern = 1
    tr_pareto = 1
    x = [item for sublist in x.flatten().tolist() for item in sublist]
    for j in range(2):
        # mutliply weights by row for bern
        # print(row.flatten())

        # print(j)
        p1 = [(mu[j][i] ** x[i]) * ((1 - mu[j][i]) ** (1 - x[i]))
                   for i in range(54)]
        # mutliply weights by row for pareto
        p2 = [mu[j][i] * (x[i] ** (-mu[j][i] - 1))
                     for i in range(54, 57)]
        result[j] *= np.prod(p1) * np.prod(p2)
    return np.argmax(result)


# print("mu")
# print(mu[0][0])
# print("weights")
# print(weights[0][0])
result1 = np.zeros((2,2))
for i in range(93):
    if Y_test[i] == 0 and fx(X_test[i], mu) == 0:
        result1[0][0] = result1[0][0] + 1
    elif Y_test[i] == 0 and fx(X_test[i], mu) == 1:
        result1[0][1] = result1[0][1] + 1
    elif Y_test[i] == 1 and fx(X_test[i], mu) == 0:
        result1[1][0] = result1[1][0] + 1
    else:
        result1[1][1] = result1[1][1] + 1
print(result1)
print((result1[0][0] + result1[1][1]) / 93)




#Plot the figure


x = list(range(1,54))
#print(x)
plt.figure(1)
plt.subplot(211)
markerline, stemlines, baseline = plt.stem(x, mu[0][x], '-.')
plt.setp(baseline, 'color', 'r', 'linewidth', 2)
plt.subplot(212)
markerline, stemlines, baseline = plt.stem(x, mu[1][x], '-.')
plt.setp(baseline, 'color', 'r', 'linewidth', 2)

plt.show()

