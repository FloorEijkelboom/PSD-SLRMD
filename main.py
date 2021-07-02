# To run any of the experiments, simply execute PEEi(), APEi() or Fi(). 
# Some experiments have a full and abridged version, where if no parameter 
# is specified the abridged version is executed. 

from numpy.core.numeric import full
import scipy.linalg as ln
import matgen as mg
import models as mod
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
import itertools
plt.style.use('seaborn-whitegrid')

# Constants
DIM = 50
SPARSITY = 0.15
RANK = 5
GAMMA = 0.05


def GenerateMats_S(rank=RANK, sparsity=SPARSITY, dim=DIM):
    """Generates a symmetric sparse low-rank datapoint."""
    A_gen = mg.sparse_matrix(dim, sparsity)
    B_gen = mg.lowrank_matrix(dim, rank)
    C_obj = A_gen + B_gen

    return (A_gen, B_gen, C_obj)

def GenerateMats_PSD(rank=RANK, sparsity=SPARSITY, dim=DIM):
    """Generates a PSD sparse low-rank datapoint."""
    A_gen = mg.sparse_matrix_PSD(dim, sparsity)
    B_gen = mg.lowrank_matrix_PSD(dim, sparsity)
    C_obj = A_gen + B_gen

    return (A_gen, B_gen, C_obj)

def AvgIters(model, PSD=True, nr_trials=10):
    """Determine avg nr of iters until termination
    for a given model.
    """
    scores = []
    for _ in range(nr_trials):
        if PSD:
            _, _, C_obj = GenerateMats_PSD()
        else:
            _, _, C_obj = GenerateMats_S()
        _, _, nr_iters = model.predict(C_obj, RANK)
        if nr_iters > 1:
            scores.append(nr_iters)
    
    # return average performance
    return (sum(scores)/len(scores))

def ADM_Par(model, deltas, xis, PSD):
    """General function executing PEE.2 and PEE3.
    """
    iters = []

    for delta in deltas:
        for xi in xis:
            mod = model(GAMMA, xi, delta, 200)
            iters.append(AvgIters(mod, PSD))
    

    x, y = list(zip(*[i for i in itertools.product(deltas, xis)]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('$\\delta$')
    ax.set_ylabel('$\\xi$')
    ax.set_zlabel('avg. nr. iterations')

    ax.plot_trisurf(x, y, iters, cmap = cm.coolwarm, linewidth=0, antialiased=False)

    plt.show()

def delta_exp(model, deltas, PSD):
    """Evaluates performance of different delta for 
    ADM.
    """
    delta_scores1 = []
    delta_scores2 = []
    delta_scores3 = []

    for delta in deltas:
        model1 = model(GAMMA, 0.62, delta)
        model2 = model(GAMMA, 1.02, delta)
        model3 = model(GAMMA, 1.42, delta)

        delta_scores1.append(AvgIters(model1, PSD))
        delta_scores2.append(AvgIters(model2, PSD))
        delta_scores3.append(AvgIters(model3, PSD))

    plt.plot(deltas, delta_scores1, label='0.62')
    plt.plot(deltas, delta_scores2, label='1.02')
    plt.plot(deltas, delta_scores3, label='1.42')
    plt.xlabel("$\delta$")
    plt.ylabel("avg. nr. iterations")
    plt.legend()
    plt.show()

def xi_exp(model, xis, PSD):
    """Evaluates performance of different xi for 
    ADM.
    """
    xi_scores1 = []
    xi_scores2 = []
    xi_scores3 = []

    for xi in xis:
        model1 = model(GAMMA, xi, 100)
        model2 = model(GAMMA, xi, 500)
        model3 = model(GAMMA, xi, 1000)

        xi_scores1.append(AvgIters(model1, PSD))
        xi_scores2.append(AvgIters(model2, PSD))
        xi_scores3.append(AvgIters(model3, PSD))

    plt.plot(xis, xi_scores1, label='100')
    plt.plot(xis, xi_scores2, label='500')
    plt.plot(xis, xi_scores3, label='1000')
    plt.xlabel("$\\xi$")
    plt.ylabel("avg. nr. iterations")
    plt.legend()
    plt.show()


def EvaluateModel(model, PSD=True, rank=RANK, sparsity=SPARSITY, max_iter = 10):
    """Evaluates average sparsity, rank and iteration until convergence 
    of a given model.
    """
    sparses = []
    ranks = []
    perfs = []

    for _ in range(max_iter):
        if PSD:
            _, _, C_obj = GenerateMats_PSD()
        else:
            _, _, C_obj = GenerateMats_S()
        A_pred, B_pred, perf = model.predict(C_obj, RANK)
        sparses.append((DIM*DIM - np.count_nonzero(A_pred))/(DIM*DIM))
        ranks.append(np.linalg.matrix_rank(B_pred, 10e-2))
        perfs.append(perf)

    return sum(sparses)/len(sparses), sum(ranks)/len(ranks), sum(perfs)/len(perfs)


def PEE1(full_comp = False):
    scores = []
    etas = [i/1000 for i in range(1, 25000, 2000 - full_comp * 1500)]
    for eta in etas:
        model = mod.SubgradientDescent(GAMMA, eta)
        scores.append(AvgIters(model, False, 10))

    plt.plot(etas, scores)
    plt.xlabel("$\eta$")
    plt.ylabel("$f_{best}$")
    plt.legend()
    plt.show()

def PEE2(full_comp = False):
    deltas = [i for i in range(50, 1000, 150 - full_comp * 125)]
    xis = [i/100 for i in range(25, 2000, 250 - full_comp * 225)]
    ADM_Par(mod.ADM_S, deltas, xis, PSD=False)

def PEE3(full_comp = False):
    deltas = [i for i in range(50, 600, 10 - full_comp * 80)]
    xis = [i/100 for i in range(5, 306, 100 - full_comp * 60)]
    ADM_Par(mod.ADM_PSD, deltas, xis, PSD=False)


def APE1():
    ranks = [i for i in range(4, 30, 4)]
    model = mod.SubgradientDescent(GAMMA, 4.2)
    rank_data = []

    for rank in ranks:
        _, rank_perf, _ = EvaluateModel(model, False, rank)
        rank_data.append(rank_perf)
    
    barWidth = 1
    
    sns.set(style='darkgrid')
    plt.bar(ranks, [rank_data[i] - ranks[i] for i in range(len(ranks))], color='#557f2d', width=barWidth, edgecolor='white', label='SGD')
    plt.xlabel('rank generated matrix')
    plt.ylabel('avg. nr. iterations')

    plt.show()

    # plt.plot(ranks, [rank_data[i] - ranks[i] for i in range(len(ranks))])
    # plt.xlabel("rank generated matrix")
    # plt.ylabel("difference rank predicted and generated")
    # plt.legend()
    # plt.show()

def APE2():
    sparsities = [i/10 for i in range(1, 5)]
    model = mod.SubgradientDescent(GAMMA, 4.2)
    sparisty_data = []

    for sparsity in sparsities:
        sparsity, _, _ = EvaluateModel(model, True, sparsity=sparsity, max_iter=25)
        sparisty_data.append(sparsity)
    
    barWidth = 0.025
    sns.set(style='darkgrid')
    plt.bar(sparsities, [sparisty_data[i] - sparsities[i] for i in range(len(sparsities))], color='#557f2d', width=barWidth, edgecolor='white', label='SGD')
    plt.xlabel('rank generated matrix')
    plt.ylabel('avg. nr. iterations')
    plt.axhline(color='black')

    plt.show()

def APE3():
    ADM = mod.ADM(GAMMA, (DIM * DIM)/315, 600)
    ADM_S = mod.ADM_S(GAMMA, 0.2, 600)
    ADM_PSD = mod.ADM_PSD(GAMMA, 1.08, 600)
    
    ranks = [10, 15, 20, 25, 30]
    ADM_perf = []
    ADMS_perf = []
    ADMPSD_perf = []

    for rank in ranks:
        ADM_scores = []
        ADM_S_scores = []
        ADM_PSD_scores = []

        for _ in range(10):
            A, B, C = GenerateMats_PSD(rank, SPARSITY, DIM)
            _, _, iters_ADM = ADM.predict(C, rank)
            _, _, iters_ADM_S = ADM_S.predict(C, rank)
            _, _, iters_ADM_PSD = ADM_PSD.predict(C, rank)

            ADM_scores.append(iters_ADM)
            ADM_S_scores.append(iters_ADM_S)
            ADM_PSD_scores.append(iters_ADM_PSD)
        
        ADM_perf.append(sum(ADM_scores)/len(ADM_scores))
        ADMS_perf.append(sum(ADM_S_scores)/len(ADM_S_scores))
        ADMPSD_perf.append(sum(ADM_PSD_scores)/len(ADM_PSD_scores))

    barWidth = 1
    r1 = ranks
    r2 = [x - barWidth for x in r1]
    r3 = [x + barWidth for x in r1]
    
    sns.set(style='darkgrid')
    plt.bar(r2, ADM_perf, color='#7f6d5f', width=barWidth, edgecolor='white', label='ADM')
    plt.bar(r1, ADMS_perf, color='#557f2d', width=barWidth, edgecolor='white', label='ADM-S')
    plt.bar(r3, ADMPSD_perf, color='#2d7f5e', width=barWidth, edgecolor='white', label='ADM-PSD')
    plt.xlabel('rank generated matrix')
    plt.ylabel('avg. nr. iterations')

    plt.legend(loc=2)

    plt.show()

def APE4():
    ADM = mod.ADM(GAMMA, (DIM * DIM)/315, 600)
    ADM_S = mod.ADM_S(GAMMA, 0.2, 600)
    ADM_PSD = mod.ADM_PSD(GAMMA, 1.08, 600)
    
    spars = [0.1, 0.2, 0.3, 0.4]
    ADM_perf = []
    ADMS_perf = []
    ADMPSD_perf = []

    for spar in spars:
        ADM_scores = []
        ADM_S_scores = []
        ADM_PSD_scores = []

        for _ in range(10):
            A, B, C = GenerateMats_PSD(RANK, spar, DIM)
            _, _, iters_ADM = ADM.predict(C, RANK)
            _, _, iters_ADM_S = ADM_S.predict(C, RANK)
            _, _, iters_ADM_PSD = ADM_PSD.predict(C, RANK)

            ADM_scores.append(iters_ADM)
            ADM_S_scores.append(iters_ADM_S)
            ADM_PSD_scores.append(iters_ADM_PSD)
        
        ADM_perf.append(sum(ADM_scores)/len(ADM_scores))
        ADMS_perf.append(sum(ADM_S_scores)/len(ADM_S_scores))
        ADMPSD_perf.append(sum(ADM_PSD_scores)/len(ADM_PSD_scores))

    barWidth = 0.015
    r1 = spars
    r2 = [x - barWidth for x in r1]
    r3 = [x + barWidth for x in r1]
    
    sns.set(style='darkgrid')
    plt.bar(r2, ADM_perf, color='#7f6d5f', width=barWidth, edgecolor='white', label='ADM')
    plt.bar(r1, ADMS_perf, color='#557f2d', width=barWidth, edgecolor='white', label='ADM-S')
    plt.bar(r3, ADMPSD_perf, color='#2d7f5e', width=barWidth, edgecolor='white', label='ADM-PSD')
    plt.xlabel('sparsity generated matrix')
    plt.ylabel('avg. nr. iterations')

    plt.legend(loc=2)

    plt.show()


def F1a():
    deltas = [i for i in range(50, 1000, 15)]
    delta_exp(mod.ADM_S, deltas, PSD=False)

def F1b():
    xis = [i/100 for i in range(25, 2000, 25)]
    xi_exp(mod.ADM_S, xis, PSD=False)

def F2a():
    deltas = [i for i in range(30, 600, 10)]
    delta_exp(mod.ADM_PSD, deltas, PSD=True)

def F2b():
    xis = [i/100 for i in range(5, 201, 10)]
    xi_exp(mod.ADM_PSD, xis, PSD=True)


# EXAMPLE: 
PEE1()
PEE2()
PEE3()
APE1()
APE2()
APE3()
APE4()
F1a()
F1b()
F2a()
F2b()