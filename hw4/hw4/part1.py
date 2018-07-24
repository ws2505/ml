import numpy as np
import matplotlib.pylab as plt


ntrain = 500
weights = (0.2, 0.5, 0.3)

cov = np.matrix([[1, 0], [0, 1]])
mu1 = np.array([0, 0])
mu2 = np.array([3, 0])
mu3 = np.array([0, 3])
gauss1 = np.random.multivariate_normal(mu1, cov, ntrain)
gauss2 = np.random.multivariate_normal(mu2, cov, ntrain)
gauss3 = np.random.multivariate_normal(mu3, cov, ntrain)
choice = np.random.choice(range(3), size=ntrain, p=weights)
data = np.concatenate((gauss1[choice == 0, :], gauss2[choice == 1, :], gauss3[choice == 2, :]))
k = 2
centers = np.random.uniform(low=0, high=1, size=(k, 2))
objective = []
k_val = range(2, 6)
colors = ['blue', 'green', 'red', 'black', 'yellow']
T = 20
n_train = 500
weights = (0.2, 0.5, 0.3)
k_backup = [3, 5]
cluster_store = []
plt.figure()


def get_closest(row):
    errors = np.sum((centers - row) ** 2, axis=1)
    sel = np.argmin(errors)
    return (sel, errors[sel])

for i in range(len(k_val)):
    k = k_val[i]
    centers = np.random.uniform(low=0, high=1, size=(k, 2))
    objective = []
    for t in range(T):
        cluster_assgn = np.apply_along_axis(get_closest,
                                            1, data)

        objective.append(np.sum(cluster_assgn[:, 1]))

    plt.plot(range(1, T + 1), objective, colors[i])

    # store cluster assignments for k=3,5
    if k_val[i] in k_backup:
        cluster_store.append(cluster_assgn[:, 0])

plt.xticks(range(1, T + 1))
plt.xlabel('Iterations')
plt.ylabel('Objective')
plt.title('Objective vs Iteration for K = [2,3,4,5]')
plt.legend(['K = %d' % i for i in k_val])