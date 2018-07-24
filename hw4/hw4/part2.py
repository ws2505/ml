# Python3
import numpy as np
import pandas as pd
import csv
import matplotlib.pylab as plt


def read_file(filename):
    data = []
    with open(filename) as file:
        content = csv.reader(file)
        for c in content:
            data.append(c)
    return np.matrix(data).astype(np.float32)


train = "hw4-data/ratings.csv"
test = "hw4-data/ratings_test.csv"
mmap = "hw4-data/movies.txt"
num_user = 943
num_movies = 1682
ntrain = 0
ntest = 0
var = 0.25
d = 10
lmb = 1
movies = []
objective = []
Q = np.repeat(np.nan, num_user * d).reshape(num_user, d)
R = np.random.multivariate_normal(np.repeat(0, d),
                                  np.identity(d) / lmb,
                                  num_movies).T



def create_matrix(self, file, train=False, test=False):
    matrix = np.repeat(np.nan, self.nusers * self.nmovies).reshape(self.nusers, self.nmovies)
    with open(file) as f:
        for row in f:
            val = row.rstrip('\n').split(',')
            # print(val)
            matrix[int(val[0]) - 1, int(val[1]) - 1] = float(val[2])
            if train:
                self.num_train_cases += 1
            if test:
                self.num_test_cases += 1
    return matrix

M = create_matrix(train, train=True)
Mtest = create_matrix(test, test=True)



def error(matrix):
    predicted = Q.dot(R)
    observed_ind = ~np.isnan(matrix)
    error = ((matrix[observed_ind] - predicted[observed_ind]) ** 2).sum()
    return error


def square(matrix):
    return (matrix ** 2).sum()

for p in range(100):
    for i in range(num_user):
            observed_ind = ~np.isnan(M[i, :])
            # print(sum(observed_ind))
            Ri = R[:, observed_ind]
            Mi = M[i, observed_ind]
            # print(Ri.shape, Mi.shape)
            Q[i, :] = np.linalg.inv(Ri.dot(Ri.T) + lmb * var * np.identity(d)).dot(Ri.dot(Mi.T))

    for j in range(num_movies):
            observed_ind = ~np.isnan(M[:, j])
            # print(sum(observed_ind))
            Qj = Q[observed_ind, :]
            Mj = M[observed_ind, j]
            # print(Ri.shape, Mi.shape)
            R[:, j] = np.linalg.inv(Qj.T.dot(Qj) + lmb * var * np.identity(d)).dot(
                    Qj.T.dot(Mj.T))

    if p > 1:
        obj_neg = (error(M) / (2 * var) + square(Q) * lmb / 2 + square(R) * lmb / 2)
        objective.append(-obj_neg)



num_runs = 10
x_vals = list(range(2, 100))


results = pd.DataFrame(index=range(num_runs), columns=['s.no.', 'objective', 'test_rmse'])


query_movies = ["Star Wars", "My Fair Lady", "GoodFellas"]

movie_results = pd.DataFrame(index=range(10),columns=query_movies)

dist_results = pd.DataFrame(index=range(10),columns=query_movies)

max_obj = -np.inf
results['s.no.'] = list(range(1, num_runs + 1))

plt.figure()
for i in range(num_runs):


        objective = []
        for p in range(100):
            for i in range(num_user):
                observed_ind = ~np.isnan(M[i, :])

                Ri = R[:, observed_ind]
                Mi = M[i, observed_ind]

                Q[i, :] = np.linalg.inv(Ri.dot(Ri.T) +
                                                     lmb * var * np.identity(d)).dot(Ri.dot(Mi.T))

            for j in range(num_movies):
                observed_ind = ~np.isnan(M[:, j])

                Qj = Q[observed_ind, :]
                Mj = M[observed_ind, j]

                R[:, j] = np.linalg.inv(Qj.T.dot(Qj) + lmb * var * np.identity(d)).dot(
                            Qj.T.dot(Mj.T))

            if p > 1:
                obj_neg = (error(M) / (2 * var) +
                                   square(Q) * lmb / 2 +
                                   square(R) * lmb / 2)
                objective.append(-obj_neg)
            plt.plot(x_vals, objective, label='run%d' % (i + 1))
            results.loc[i, 'objective'] = objective[-1]
            results.loc[i, 'test_rmse'] = np.sqrt(error(Mtest) / ntest)
if objective[-1] > max_obj:
    max_obj = objective[-1]

    for movie in query_movies:
        movie_id = [i for i in range(num_movies) if movie in movies[i]][0]

    distances = np.sqrt(((R - R[:, movie_id].reshape(-1, 1)) ** 2).sum(axis=0))
    min_movies_id = np.argsort(distances)[1:11]
    movie_results[movie] = movies[min_movies_id]
    dist_results[movie] = distances[min_movies_id]

plt.xticks([int(x) for x in np.linspace(0, 100, 9)])
plt.xlabel('Iteration')
plt.ylabel('Objective Function')
plt.legend(loc='best')
plt.savefig('hw4_2a_obj.png')
plt.show()



results = results.sort_values(by='objective', axis=0, ascending=False)
for movie in query_movies:
    movie_id = [i for i in range(num_movies) if movie in movies[i]][0]
    distances = np.sqrt(((R - R[:, movie_id].reshape(-1, 1)) ** 2).sum(axis=0))
    min_movies_id = np.argsort(distances)[1:11]
    movie_results[movie] = movies[min_movies_id]
    dist_results[movie] = distances[min_movies_id]

movie_results.to_csv('hw4_2b_query_movie.csv', index=False)
dist_results.to_csv('hw4_2b_query_distances.csv', index=False)
results.to_csv('hw4_2a_obj_rmse.csv', index=False)
