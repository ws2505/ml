import csv
import numpy as np
import matplotlib.pylab as plt

def read_file(filename):
    data = []
    with open(filename) as file:
        content = csv.reader(file)
        for c in content:
            data.append(c)
    return np.matrix(data).astype(np.float32)


X_train = read_file('hw3-data/gaussian_process/X_train.csv')
y_train = read_file('hw3-data/gaussian_process/y_train.csv')
X_test = read_file('hw3-data/gaussian_process/X_test.csv')
y_test = read_file('hw3-data/gaussian_process/y_test.csv')



def calculate_kernel(b, X, X_train):
    K = np.zeros((len(X),len(X_train)))

    for i in range(len(X)):
        for j in range(len(X_train)):
            x = np.asarray(X[i]).flatten()
            xtrain = np.asarray(X_train[j]).flatten()
            K[i][j] = np.exp(-1 / b * np.dot(x - xtrain, x - xtrain))
    return K




def predict(K_test, K_train, y_train, var):
    ypredict = K_test.dot(np.linalg.inv(var * np.identity(len(K_train)) + K_train)).dot(y_train)
    return ypredict

# K_train = calculate_kernel(5, X_train,X_train)
# K_test = calculate_kernel(5,X_test,X_train)
# y_predict = predict(K_test, K_train, y_train, 0.1)


def calculate_rmse(ytest, ypredict):
        x1 = np.asarray(ytest).flatten()
        x2 = np.asarray(ypredict).flatten()
        rmse = np.sqrt(np.dot(x1 - x2, x1 - x2) / len(ytest))
        return rmse

# print(np.shape(K_test))
# print(np.shape(K_train))
# rmse = calculate_rmse(y_test, y_predict)
# print(y_predict)

# #save the result
# file = open('result1.txt', 'w')
# for item in y_predict:
#   file.write("%s\n" % item)

#part 2
# b = [5,7,9,11,13,15]
# var = [x/10 for x in range(1,11)]
# # print(var)
# rmse = np.zeros((len(b), len(var)))
# for i in range(6):
#     for j in range(10):
#         currentb = b[i]
#         currentvar = var[j]
#         print(currentb, " ", currentvar)
#
#         K_train = calculate_kernel(b[i], X_train, X_train)
#         K_test = calculate_kernel(b[i], X_test, X_train)
#         y_predict = predict(K_test, K_train, y_train, var[j])
#         rmse[i][j] = calculate_rmse(y_test, y_predict)
#         print(rmse[i][j])
#
# print(rmse)
#
# file = open('result2.txt', 'w')
# for item in rmse:
#   file.write("%s" % item)


#part4
#gp = gaussian_process(Xtrain[:, 3], ytrain, Xtrain[:, 3], ytrain)
b=5
var=2

K_train = calculate_kernel(b, X_train[:,3],X_train)
K_train = calculate_kernel(b,X_train[:,3],X_train)
y_predict = predict(K_train, K_train, y_train, var)


a = X_train[:,3]
# print(a)
# print(y_predict)
x = np.asarray(a).flatten()

xsortind = np.argsort(x)
# print("xsort", len(xsortind))
y1 = np.asarray(y_train).flatten()
y2 = np.asarray(y_predict).flatten()
plt.figure()
plt.scatter(x[xsortind], y1[xsortind], label = "actual y")
plt.plot(x[xsortind], y2[xsortind], 'r-', label ="predicted")
plt.xlabel('4 dimension')
plt.ylabel('result')
# plt.title('Visualizing model through single dimension')
# plt.savefig('hw3_gaussian_dim4_viz')
plt.show()