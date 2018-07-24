import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def nmf(doc, vocabulary):
    itr = 100
    doc_num = 8447
    vocab_num = 3012
    d = 25
    X = np.zeros((vocab_num, doc_num))
    dic = 0
    with open(doc) as f:
        for row in f:
            count = row.rstrip('\n').split(',')
            for wc in count:
                ind, cnt = [int(x) for x in wc.split(':')]
                X[ind - 1, dic] = cnt
            dic += 1
    with open(vocabulary) as f:
        words = np.array([x.rstrip('\n')
                          for x in f.readlines()])
    W = np.random.uniform(1, 2, (vocab_num, d))
    H = np.random.uniform(1, 2, (d, doc_num))
    objective = []
    WH = W.dot(H)
    A = X / (WH + 1e-16)

    for i in range(itr):
        if i % 10 == 0:
            print(i)

        H = np.multiply(H, W.T.dot(A)) / np.sum(W, axis=0).reshape(d, 1)
        WH = W.dot(H)
        A = X / (WH + 1e-16)
        W = np.multiply(W, A.dot(H.T)) / np.sum(H, axis=1).reshape(1, d)

        WH = W.dot(H)
        A = X / (WH + 1e-16)

        obj = np.sum(np.multiply(np.log(1 / (WH + 1e-16)), X) + WH)
        objective.append(obj)

    W = W / (np.sum(W, axis=0).reshape(1, -1))
    word_idx = np.apply_along_axis(lambda x: np.argsort(x)[-10:][::-1],
                                   axis=0, arr=W)
    results = pd.DataFrame(index=range(10),
                           columns=['Topic_%d' % i for i in range(1, 26)])

    for i in range(25):
        results.iloc[:, i] = list(zip([format(x, '.3f') for x in
                                       W[word_idx[:, i], i]],
                                      words[word_idx[:, i]]))

    print(results)
    results.to_csv('hw5_words.csv', index=False)

    plt.figure()
    plt.plot(range(1, itr + 1), nmf.objective)
    plt.xticks(np.linspace(1, itr, 10))
    plt.xlabel('Iterations')
    plt.ylabel('Objective')
    plt.title('Variation of objective with iterations')
    plt.savefig('hw5_2a.png')
    plt.show()

nmf(train ="hw5-data/nyt_data.txt", test ="hw5-data/nyt_vocab.dat")