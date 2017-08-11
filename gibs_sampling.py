import sys
import numpy as np
import pandas as pd
import math
import csv
from math import pow
import scipy.stats as ss
from scipy.special import gamma
from numpy.random import *


def normal_dist(x, mu, delta, dim):
    v = np.matrix(x - mu)
    det = np.linalg.det(delta.I)
    c = pow(det, 0.5)
    res = np.exp(-0.5 * float(np.dot(v, np.dot(delta, np.matrix(v).T)))) / (pow(2 * math.pi, dim / 2.0) * c)
    return res

def lilelihood(X, Z, mus, deltas, N, K):

    lh = 0.0
    for n in range(N):
        temp = 0.0
        for k in range(K):
            temp += Z[n,k] * normal_dist(X[n], mus[k], deltas[k], 3)
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

def gen_pi():
    pi1 = rand()
    pi2 = rand() * (1.0 - pi1)
    pi3 = (1.0 - pi1 - pi2)
    return  [pi1, pi2, pi3]


def calc_pi_dirit(pi, dirichletBase, newAlpha):
    coef = 1.0
    for i in range(3):

        print('pi %.6f' % pi[i])
        print('nweAlpha i %.6f' % newAlpha[i] )
        print('pow %.6f' % pow(pi[i], newAlpha[i] - 1.0))
        coef *= pow(pi[i], newAlpha[i] - 1.0)
    print('coef %.6f' % coef)
    return dirichletBase * coef

def calc_pi_dirit_log(pi, dirichletBase, newAlpha):
    com = math.log(dirichletBase)

    for i in range(3):
        com += (newAlpha[i] - 1.0) * math.log(pi[i])
    return com


def calc_sk1(Z):
    sk1 = []

    for k in range(3):
        gk = Z[:, k]
        sk1.append(float(sum(gk)))

    return sk1

def calc_skx(Z,X):
    skx = []

    for k in range(3):
        temp = np.array([0.0, 0.0, 0.0])
        gk = Z[:, k]

        for j in range(len(X)):
            temp += float(gk[j]) * X[j]

        skx.append(temp)
    return skx

def calc_skxx(Z,X):
    skxx = []

    for k in range(3):
        temp = np.matrix([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
        gk = Z[:, k]

        for j in range(len(X)):
            temp += float(gk[j]) * np.dot(np.matrix(X[j]).T, np.matrix(X[j]))

        skxx.append(temp)
    return skxx

def calc_skxxmu(Z,X,mus):

    skxxmu = []

    for k in range(3):
        temp = np.matrix([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
        gk = Z[:, k]

        for j in range(len(X)):
            temp += float(gk[j]) * np.dot(np.matrix(X[j] - mus[k]).T, np.matrix(X[j] - mus[k]))

        skxxmu.append(temp)

    return skxxmu

def calc_wishert_log(S, W, nu):

    p = W.shape[0]

    t1 = (nu - p - 1.0) * 0.5 * np.log(np.linalg.det(S))
    t2 = (nu * p * 0.5) * np.log(2.0)
    t3 = (nu * 0.5) * np.log(np.linalg.det(W))
    t4 = np.log(gamma(nu * 0.5))
    t5 = np.trace(np.dot(W.I,S)) * 0.5
    return t1 - t2 - t3 - t4 - t5

def calc_normal_log(x, mu, delta, dim):
    t1 = dim * 0.5 * np.log(2.0 * math.pi)
    t2 = 0.5 * np.log(np.linalg.det(delta.I))
    t3 = 0.5 * float(np.dot(x, np.dot(delta, np.matrix(x).T)))
    return -t1 - t2 - t3

def gen_mus():
    return [np.array([rand() * 2.0 - 1.0, rand() * 2.0 - 1.0, rand() * 2.0 - 1.0]) for k in range(3)]

def gen_deltas():
    temp = [np.array([rand() * 2.0 - 1.0, rand() * 2.0 - 1.0, rand() * 2.0 - 1.0]) for k in range(3)]
    return [(np.dot(np.matrix(temp[j]).T, np.matrix(temp[j])) + np.linalg.inv(np.matrix(np.identity(3)))).I for j in range(3)]

def calc_gw_log(temp_mus, temp_deltas, newMs, newBetas, newNyus, newWs):
    com = 0.0
    for k in range(3):
        w = calc_wishert_log(temp_deltas[k], newWs[k], newNyus[k])
        n = calc_normal_log(temp_mus[k], newMs[k], newBetas[k] * temp_deltas[k], 3)
        com += n + w
    return com

def main():
    argvs = sys.argv
    if len(argvs) != 4 :
        print('Bad Args')
        return
    xfile, zfile, datfile = argvs[1], argvs[2], argvs[3]

    K = 3
    dim = 3
    df = pd.read_csv('./resource/' + xfile)
    X = df.values
    N = X.shape[0]
    Z = np.matrix([[0.0,0.0,0.0] for i in range (N)])
    mus = [np.array([0.1, 0.0, 1.0]), np.array([0.0, 0.1, 0.0]), np.array([0.0, 1.1, 1.0])]
    pis = [0.5, 0.1, 0.4]
    deltas = [np.linalg.inv(np.matrix(np.identity(3))) for i in range(K)]
    diff = 10000.0
    iter_c = 0
    bef = 0

    #Dist Params
    Alpha0 = [rand() for k in range(K)]
    Beta0 = rand()
    Nyu0 = rand() + 3.0
    M0 = np.array([rand(), rand(), rand()])
    Ws0 = np.linalg.inv(np.matrix(np.identity(3)))

    gcom = 1.0
    for k in range(K):
        gcom *= gamma(Alpha0[k])
    dirichletBase = gamma(float(sum(Alpha0))) / gcom

    while diff > 1000.0 and iter_c < 30:
        iter_c += 1
        #Sample Z from
        for n in range(N):
            ns = [pis[j] * normal_dist(X[n], mus[j], deltas[j], dim) for j in range(K)]
            frac_d = sum(ns)
            zrand = rand()
            com = 0

            for k in range(K):
                com += pis[k] * normal_dist(X[n], mus[k], deltas[k], dim) / frac_d
                if zrand < com:
                    Z[n] = np.array([1.0 if j == k else 0 for j in range(K)])
                    break
        #print(Z)
        Z = np.matrix(Z)
        #print(type(Z))
        sk1 = calc_sk1(Z)
        skx = calc_skx(Z,X)
        skxx = calc_skxx(Z,X)
        skxxmu = calc_skxxmu(Z,X,mus)
        pidirit = 0
        gw = 0

        #Update Alpha
        newAplha = [Alpha0[k] + sk1[k] for k in range(K)]
        #Sample PI p(pi|Z) = Dir(pi|alpha)
        base = calc_pi_dirit_log(pis, dirichletBase, newAplha)
        while True:
            #pirand = rand() * base
            pirand = base * 0.95 + rand() * 0.05 * base
            tempPi = gen_pi()
            pidirit = calc_pi_dirit_log(tempPi, dirichletBase, newAplha)

            if pirand < pidirit:
                pis = tempPi
                print(pis)
                break
        newBetas = [Beta0 + sk1[k] for k in range(K)]
        newMs = [(M0 * Beta0 + skx[k]) / (Beta0 + sk1[k]) for k in range(K)]
        newNyus = [Nyu0 + sk1[k] for k in range(K)]

        c1 = Beta0 * np.dot(np.matrix(M0).T, np.matrix(M0))
        newWs = [(Ws0.I + Beta0 * np.dot(np.matrix(mus[k] - M0).T, np.matrix(mus[k] - M0)) + skxxmu[k] - newBetas[k] * np.dot(np.matrix(mus[k] - newMs[k]).T, np.matrix(mus[k] - newMs[k]))).I for k in range(K)]
        base = calc_gw_log(mus, deltas, newMs, newBetas, newNyus, newWs)

        while True:
            #prand = rand() * base
            prand = base * 0.95 + rand() * 0.05 * base
            temp_mus = gen_mus()
            temp_deltas = gen_deltas()
            gw = calc_gw_log(temp_mus, temp_deltas, newMs, newBetas, newNyus, newWs)

            if prand < gw:
                mus = temp_mus
                deltas = temp_deltas
                break
        #Eval posterior propability
        lh = lilelihood(X, Z, mus, deltas, N, K)
        pp = lh + gw + pidirit
        print('iter: %d, log posterior propability: %.6f' % (iter_c, pp))
        if iter_c > 1:
            diff = abs(pp - bef)
        bef = pp

    if iter_c < 30:
        print('Conversed')
    else:
        print('Not Found')
    print(pis)

    write_z(Z, zfile)
    write_params(pis, mus, deltas, datfile)

if __name__ == '__main__':
    main()
