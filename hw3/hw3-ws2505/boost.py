import numpy as np
import csv
import matplotlib.pylab as plt

def read_file(filename):
    data = []
    with open('hw3-data/%s' % filename) as file:
        content = csv.reader(file)
        for c in content:
            data.append(c)
    return np.matrix(data).astype(np.float32)
X_train = read_file('boosting/X_train.csv')
X_test = read_file('boosting/X_test.csv')
y_train = read_file('boosting/y_train.csv')
y_test = read_file('boosting/y_test.csv')



#init varaibel
num = len(X_train)
init_weights = 1/num * np.ones((num, 1))
test_err = []
eps_res = []
alphas_res = []
sample_selected = np.zeros(num)
eps_upperbound = 0
eps_upper = []
total_train_pred = 0
total_test_pred = 0
train_err = []
iter = 1500
num = len(X_train)


#add one column to X_train and X_test
X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))


def predict(X, weights):
    res = np.sign(X.dot(weights))
    return res

for t in range(iter):
    #predict the training data and test data
    data_selected = np.random.choice(num, num,replace=True, p=list(np.asarray(init_weights).flatten()))
    X = X_train[data_selected, :]
    y = y_train[data_selected]
    weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    predict_train = predict(X, weights)
    predict_test = predict(X_test, weights)
    # calculate eps:
    eps = np.sum(init_weights[~np.equal(y_train, predict_train)])
    if eps > 0.5:
        weights = -weights
        predict_train = np.sign(X_train.dot(weights))
        predict_test = np.sign(X_test.dot(weights))
        eps = np.sum(init_weights[~np.equal(y_train, predict_train)])
    # calculate alpha
    alpha = 1/2 * np.log((1 - eps) / eps)
    eps_res.append(eps)
    alphas_res.append(alpha)
    # calculate error
    total_train_pred += alpha * predict_train
    train_error = np.mean(np.sign(total_train_pred) != y_train)
    train_err.append(train_error)

    total_test_pred += alpha * predict_test
    test_error = np.mean(np.sign(total_test_pred) != y_test)
    test_err.append(test_error)
    # update weights
    index = np.exp(-alpha * np.multiply(y_train, predict_train))
    init_weights = np.multiply(init_weights, index) / sum(np.multiply(init_weights, index))
    eps_upperbound += (0.5 - eps) ** 2
    eps_upper.append(np.exp(-2 * eps_upperbound))

    # calculate number of each data selected
    for i in data_selected:
        sample_selected[i] += 1

# #part1
# x = range(1, T + 1)
# y1 = train_errors
# y2 = test_errors
# xticks=np.linspace(1, T, 6)
# xlabel='Iteration Number'
# ylabel='Training and Testing Error'
# legend=['train error', 'test error']
# plt.figure()
# plt.plot(x, y1, 'b')
# plt.plot(x, y2, 'r')
# plt.legend(legend)
# plt.xticks(xticks)
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.show()

# #part2
# x = range(1,T+1)
# y1 = eps_upper_bound
# xticks=np.linspace(1,T,6)
# xlabel='Iteration Number'
# ylabel='Upper Bound of Training Error'
# plt.figure()
# plt.plot(x, y1)
# plt.xticks(xticks)
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.show()

#part3
plt.figure()
plt.bar(range(1, num + 1), sample_selected)
plt.xlabel('training data')
plt.ylabel('#time data is selected')
plt.show()

#part4
# x = range(1,T+1)
# y1 = alphas_res
# xticks = np.linspace(1,T,6)
# xlabel='Iteration Numbers'
# ylabel='alpha'
# plt.figure()
# plt.plot(x, y1, 'b')
# plt.xticks(xticks)
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.show()

#
# x = range(1,T+1)
# y1 = eps_res
# xticks = np.linspace(1,T,6)
# xlabel='Iteration Numbers'
# ylabel='eps'
# plt.figure()
# plt.plot(x, y1)
# plt.xticks(xticks)
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.show()