#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/22 09:14
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : NSGA_III.py
# @Statement : Nondominated sorting genetic algorithm III (NSGA-III)
# @Reference : K. Deb and H. Jain, An evolutionary many-objective optimization algorithm using reference-point based non-dominated sorting approach, part I: Solving problems with box constraints, IEEE Transactions on Evolutionary Computation, 2014, 18(4): 577-601.
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations  # 创建和操作迭代器的工具
from scipy.linalg import LinAlgError    # 用于处理线性代数相关的错误
from scipy.spatial.distance import cdist    # 用于计算距离或相似性


def cal_obj(pop, nobj):
    '''
    :param pop: ndarray，决策变量的取值，形状为(N, D)，N为种群中个体的数量，D为决策变量的数量
    :param nobj: int，目标函数的数量
    :return: ndarray，计算得到的目标函数值，形状为(N, nobj)，N是种群中个体的数量，nobj是目标函数的数量。
    '''
    # 这里的函数是 DTLZ1 函数，用于多目标优化问题。g 是一个中间变量。
    g = 100 * (pop.shape[1] - nobj + 1 + np.sum(
        (pop[:, nobj - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, nobj - 1:] - 0.5)), axis=1))

    objs = np.zeros((pop.shape[0], nobj))   # 创建一个大小为 (pop.shape[0], nobj) 的零矩阵，用于存储目标函数值。

    temp_pop = pop[:, : nobj - 1]   # 从输入的 pop 矩阵中取出前 nobj-1 列，存储在 temp_pop 中。

    for i in range(nobj):   # 遍历目标函数的数量（nobj）
        f = 0.5 * (1 + g)   # 计算 f，其中 g 是前面计算的中间变量。

        f *= np.prod(temp_pop[:, : temp_pop.shape[1] - i], axis=1)  # 使用累积乘法计算 f 的一部分，这部分与 temp_pop 相关。

        if i > 0:
            f *= 1 - temp_pop[:, temp_pop.shape[1] - i]     # 如果 i 大于 0，再乘以 1 减去 temp_pop 的一部分。

        objs[:, i] = f  # 将计算得到的 f 存储在目标函数矩阵的第 i 列。

    return objs


def factorial(n):
    # calculate n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def combination(n, m):
    # choose m elements from an n-length set
    # 计算排列组合数C(n,m)
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))


def reference_points(npop, nvar):
    '''
    calculate approximately npop uniformly distributed reference points on nvar dimensions
    :param npop: int，要生成的均匀分布的参考点的数量
    :param nvar: int，参考点的维度
    :return: ndarray，所生成的均匀分布的参考点的坐标，形状为(npop, nvar)
    '''
    h1 = 0  # 用于控制循环的计数器
    while combination(h1 + nvar, nvar - 1) <= npop: # 目的是确定一个足够大的h1，以便在nvar-1维度上有足够的均匀分布的参考点。
        h1 += 1
    # 使用组合数的计算结果，构建一个 points 数组，这个数组包含了 h1 维度中的参考点坐标
    points = np.array(list(combinations(np.arange(1, h1 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
    # 对 points 数组中的坐标进行变换，以确保它们在 [0, 1] 范围内均匀分布
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < nvar:   # 如果h1不足以在nvar-1维度上生成足够多的参考点
        h2 = 0
        # h2 的值，以便在nvar-1维度上有足够多的均匀分布的参考点。这些参考点将与之前的points组合在一起。
        while combination(h1 + nvar - 1, nvar - 1) + combination(h2 + nvar, nvar - 1) <= npop:
            h2 += 1
        if h2 > 0:
            # 使用上边类似的方式，构建temp_points数组，然后进行坐标变换
            temp_points = np.array(list(combinations(np.arange(1, h2 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * nvar)
            # 将temp_points添加到points中，以得到最终的参考点数组，并将其返回
            points = np.concatenate((points, temp_points), axis=0)
    return points


def nd_sort(objs):
    """
    fast non-domination sort
    :param objs: ndarray，种群中每个个体的目标函数值，形状为(npop, nobj)，其中npop是种群中个体的数量，nobj是目标函数的数量。
    :return pfs: dict，键表示非支配级别，对应的值是一个包含相应级别的Pareto前沿中的个体索引的列表。
    :return rank: ndarray，每个个体的非支配级别，形状为 (npop,)，其中npop是种群中个体的数量。rank数组指示了每个个体所属的非支配级别
    """
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 0
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    pfs.pop(ind)
    return pfs, rank


def selection(pop, pc, rank, k=2):
    """
    binary tournament selection
    :param pop: ndarray，种群中每个个体的决策变量值，形状为(npop,nvar)，即（种群中个体数量，决策变量数量）
    :param pc: float，选择概率
    :param rank: ndarray，每个个体的非支配级别，形状为(npop,)，即（种群中个体数，）
    :param k: 锦标赛选择中参与竞争的个体数量
    :return: ndarray，选择后得到的用于繁殖的个体的决策变量值，形状为(nm,nvar)，nm为选择个体数量，通常等于npop*pc，确保为偶数
    """
    (npop, nvar) = pop.shape
    nm = int(npop * pc)
    nm = nm if nm % 2 == 0 else nm + 1
    mating_pool = np.zeros((nm, nvar))
    for i in range(nm):
        [ind1, ind2] = np.random.choice(npop, k, replace=False)
        if rank[ind1] <= rank[ind2]:
            mating_pool[i] = pop[ind1]
        else:
            mating_pool[i] = pop[ind2]
    return mating_pool


def crossover(mating_pool, lb, ub, pc, eta_c):
    """
    simulated binary crossover (SBX) 模拟二进制交叉
    :param mating_pool: ndarray用于繁殖的个体的决策变量值，形状(noff,nvar)，即（要繁殖的个体数，决策变量数）
    :param lb: ndarray，决策变量下界(lower bound)，形状(nvar,)
    :param ub: ndarray，决策变量上界(upper bound)，形状(nvar,)
    :param pc: float，交叉概率
    :param eta_c: int，扩散因子分布指数，用于控制模拟二进制交叉的分布情况，值越大（>10），分布越均匀
    :return: ndarray，交叉结果，形状为(noff, nvar)
    """
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]  #拆分
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > pc, (1, nvar))] = 1
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2   # 交叉
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)    # 重新拼接
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, pm, eta_m):
    """
    polynomial mutation 多项式变异
    :param pop: ndarray用于繁殖的个体的决策变量值，形状(noff,nvar)，即（要繁殖的个体数，决策变量数）
    :param lb: ndarray，决策变量下界(lower bound)，形状(nvar,)
    :param ub: ndarray，决策变量上界(upper bound)，形状(nvar,)
    :param pm: float，变异概率
    :param eta_m: 扰动因子分布指数，用于控制多项式变异的分布形状，值很大时，变异幅度较小，变异的形状更趋于均匀分布
    :return: ndarray，交叉结果，形状为(noff, nvar)
    """
    (npop, nvar) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, nvar)) < pm / nvar
    mu = np.random.random((npop, nvar))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def environmental_selection(pop, objs, zmin, npop, V):
    """
    NSGA-III environmental selection
    :param pop: ndarray，用于繁殖的个体的决策变量值，形状(noff,nvar)，即（要繁殖的个体数，决策变量数）
    :param objs: ndarray，种群中每个个体的目标函数值，形状为 (npop, nobj)，nobj 是目标函数的数量。
    :param zmin: ndarray，每个目标函数的最小值。它的形状为 (nobj,)
    :param npop: int，环境选择后要保留的个体数量
    :param V: 权向量，用于计算多目标优化中的 Pareto 前沿，形状通常是 (nv, nobj)，即（权向量的数量，目标函数的数量）每个权向量代表一种目标函数权重的组合
    :return:
    """
    pfs, rank = nd_sort(objs)
    nobj = objs.shape[1]
    selected = np.full(pop.shape[0], False)
    ind = 0
    while np.sum(selected) + len(pfs[ind]) <= npop:
        selected[pfs[ind]] = True
        ind += 1
    K = npop - np.sum(selected)

    # select the remaining K solutions
    objs1 = objs[selected]
    objs2 = objs[pfs[ind]]
    npop1 = objs1.shape[0]
    npop2 = objs2.shape[0]
    nv = V.shape[0]
    temp_objs = np.concatenate((objs1, objs2), axis=0)
    t_objs = temp_objs - zmin

    # extreme points
    extreme = np.zeros(nobj)
    w = 1e-6 + np.eye(nobj)
    for i in range(nobj):
        extreme[i] = np.argmin(np.max(t_objs / w[i], axis=1))

    # intercepts
    try:
        hyperplane = np.matmul(np.linalg.inv(t_objs[extreme.astype(int)]), np.ones((nobj, 1)))
        if np.any(hyperplane == 0):
            a = np.max(t_objs, axis=0)
        else:
            a = 1 / hyperplane
    except LinAlgError:
        a = np.max(t_objs, axis=0)
    t_objs /= a.reshape(1, nobj)

    # association
    cosine = 1 - cdist(t_objs, V, 'cosine')
    distance = np.sqrt(np.sum(t_objs ** 2, axis=1).reshape(npop1 + npop2, 1)) * np.sqrt(1 - cosine ** 2)
    dis = np.min(distance, axis=1)
    association = np.argmin(distance, axis=1)
    temp_rho = dict(Counter(association[: npop1]))
    rho = np.zeros(nv)
    for key in temp_rho.keys():
        rho[key] = temp_rho[key]

    # selection
    choose = np.full(npop2, False)
    v_choose = np.full(nv, True)
    while np.sum(choose) < K:
        temp = np.where(v_choose)[0]
        jmin = np.where(rho[temp] == np.min(rho[temp]))[0]
        j = temp[np.random.choice(jmin)]
        I = np.where(np.bitwise_and(~choose, association[npop1:] == j))[0]
        if I.size > 0:
            if rho[j] == 0:
                s = np.argmin(dis[npop1 + I])
            else:
                s = np.random.randint(I.size)
            choose[I[s]] = True
            rho[j] += 1
        else:
            v_choose[j] = False
    selected[np.array(pfs[ind])[choose]] = True
    return pop[selected], objs[selected], rank[selected]


def main(npop, iter, lb, ub, nobj=3, pc=1, pm=1, eta_c=30, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space
    :param pc: crossover probability (default = 1)
    :param pm: mutation probability (default = 1)
    :param eta_c: spread factor distribution index (default = 30) 扩散因子分布指数
    :param eta_m: perturbance factor distribution index (default = 20) 扰动因子分布指数
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # objectives
    V = reference_points(npop, nobj)  # reference vectors
    zmin = np.min(objs, axis=0)  # ideal points
    [pfs, rank] = nd_sort(objs)  # Pareto rank

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 50 == 0:
            print('Iteration: ' + str(t + 1) + ' completed.')

        # Step 2.1. Mating selection + crossover + mutation
        mating_pool = selection(pop, pc, rank)
        off = crossover(mating_pool, lb, ub, pc, eta_c)
        off = mutation(off, lb, ub, pm, eta_m)
        off_objs = cal_obj(off, nobj)

        # Step 2.2. Environmental selection
        zmin = np.min((zmin, np.min(off_objs, axis=0)), axis=0)
        pop, objs, rank = environmental_selection(np.concatenate((pop, off), axis=0), np.concatenate((objs, off_objs), axis=0), zmin, npop, V)

    # Step 3. Sort the results
    pf = objs[rank == 0]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ1')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(91, 400, np.array([0] * 7), np.array([1] * 7))
