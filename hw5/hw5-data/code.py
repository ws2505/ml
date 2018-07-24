import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.sparse.linalg import eigs

res_team = []
res_score = []
res_delta = []

def train(itr):
    record = 10
    score_file = "CFB2017_scores.csv"
    team_file = "TeamNames.txt"

    with open(team_file, 'r') as f:
        name = [line.strip() for line in f]
    name = np.array(name)
    num = name.shape[0]

    M = np.zeros((num, num))

    with open(score_file) as f:
        for row in f:
            a, pt_a, b, pt_b = [int(x) for x in row.rstrip('\n').split(',')]

            a_weight = pt_a / (pt_a + pt_b)
            a_wins = int(pt_a > pt_b)
            i = a - 1
            j = b - 1

            M[i, i] += a_wins + a_weight
            M[j, i] += a_wins + a_weight
            M[j, j] += 2 - a_wins - a_weight
            M[i, j] += 2 - a_wins - a_weight

    M = M / np.sum(M, axis=1).reshape(-1, 1)

    w_inf = eigs(M.T, 1)[1].flatten()
    w_inf = w_inf / np.sum(w_inf)

    r = record
    w = np.repeat(1 / num, num)

    for i in range(itr):
        w = np.dot(w, M)
        res_delta.append(np.sum(np.abs(w - w_inf)))
        if (i == r - 1):
            print(i)
            r *= record
            team_rank = np.argsort(w)[::-1][:25]
            name = [str(name[n]) for n in team_rank]
            team_score = [w[n] for n in team_rank]
            res_team.append(name)
            res_score.append(team_score)


itr = 10000
train(itr)

with open("hw5_team.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(res_team)

np.savetxt("hw5_score.csv", res_score, delimiter=",")

plt.figure()
plt.plot(range(1, itr + 1), res_delta)
plt.xlabel('Iteration of t')
plt.ylabel('|w_t-w_inf|')
plt.title('from t = 1 to infinite')
plt.show()
