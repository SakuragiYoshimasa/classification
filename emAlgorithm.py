import sys
import numpy as np
import pandas as pd
import math
import csv
from math import pow

def normal_dist(x, mu, delta, dim):
    v = np.matrix(x - mu)
    det = np.linalg.det(delta.I)
    c = pow(det, 0.5)
    res = np.exp(-0.5 * float(np.dot(v, np.dot(delta, np.matrix(v).T)))) / (pow(2 * math.pi, dim / 2.0) * c)
    return res

def lilelihood(X, pis, mus, deltas, N, K):

    lh = 0.0
    for n in range(N):
        temp = 0.0
        for k in range(K):
            temp += pis[k] * normal_dist(X[n], mus[k], deltas[k], 3)
        lh += math.log(temp)
    return lh

def write_params(pis, mus, deltas, datfile):
    f = open('result/' + datfile, 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['PIs'])
    writer.writerow(pis)
    writer.writerow(['MUs'])
    writer.writerows(mus)
    writer.writerow(['Deltas'])
    for i in range(len(deltas)):
        writer.writerow([i])
        writer.writerows(deltas[i].tolist())
    f.close()

def write_z(gamma, zfile):
    f = open('result/' + zfile, 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(gamma.tolist())
    f.close()

def main():
    argvs = sys.argv
    if len(argvs) != 4 :
        print('Bad Args')
        return
    xfile, zfile, datfile = argvs[1], argvs[2], argvs[3]

    #K = 3
    K = 2
    dim = 3
    df = pd.read_csv('./resource/' + xfile)
    X = df.values
    N = X.shape[0]
    gamma = np.matrix([[0.0,0.0] for i in range (N)])
    mus = [np.array([0.1, 0.0, 1.0]), np.array([0.0, 0.1, 0.0])]
    pis = [0.5, 0.5]
    deltas = [np.linalg.inv(np.matrix(np.identity(3))) for i in range(K)]
    diff = 1.0
    iter_c = 0
    bef = 0

    while diff > 0.01 and iter_c < 100:
        iter_c = iter_c + 1

        #Estep
        for n in range(N):
            ns = [pis[j] * normal_dist(X[n], mus[j], deltas[j], dim) for j in range(K)]
            frac_d = sum(ns)
            for k in range(K):
                gamma[n,k] = pis[k] * normal_dist(X[n], mus[k], deltas[k], dim) / frac_d

        #Mstep
        for k in range(K):
            gk = gamma[:, k]
            sk1 = float(sum(gk))
            pis[k] = sk1 / N

            deltas[k] = np.matrix([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
            for j in range(N):
                deltas[k] += float(gk[j]) * np.dot(np.matrix(X[j] - mus[k]).T, np.matrix(X[j] - mus[k]))
            deltas[k] /= sk1
            deltas[k] = deltas[k].I

            mus[k] = np.array([0.0,0.0,0.0])
            for j in range(N):
                mus[k] += float(gk[j]) * X[j]

            mus[k] /= sk1

        lh = lilelihood(X, pis, mus, deltas, N, K)
        print('iter: %d, logLH: %.6f' % (iter_c, lh))
        if iter_c > 1:
            diff = abs(lh - bef)
        bef = lh
    if iter_c < 100:
        print('Conversed')
    else:
        print('Not Found')
    print(pis)

    write_z(gamma, zfile)
    write_params(pis, mus, deltas, datfile)

if __name__ == '__main__':
    main()
