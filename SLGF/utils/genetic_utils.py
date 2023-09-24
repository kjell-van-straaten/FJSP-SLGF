"""utility functions for SLGA"""
import numpy as np
from deap.tools.indicator import hv
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def calc_hypervolume(hof, ref=None, from_classic=False, **kwargs):
    """function to calculate hypervolume"""
    if from_classic:
        wobj = np.array([[i[0], i[1]] for i in hof])
    else:
        wobj = np.array([ind.fitness.wvalues for ind in hof]) * -1
    if ref is None:
        ref = [np.max(wobj[:, 0]), np.max(wobj[:, 1])]

    total_hv = hv.hypervolume(wobj, ref)
    return total_hv


def norm_mean(input: np.array):
    """one function to calculate normalized mean of objectives"""
    input_norm = normalize(input, axis=0)
    output = np.mean(input_norm)
    return output


def norm_min(input: np.array):
    """one function to calculate normalized min of objectives"""
    input_norm = normalize(input, axis=0)
    output = np.min(input_norm)
    return output


def norm_std(input: np.array):
    """one function to calculate normalized std of objectives"""
    input_norm = normalize(input, axis=0)
    output = np.std(input_norm)
    return output


def calc_hypervolume_stat(pop, ref=None, **kwargs):
    """function to calculate hypervolume"""

    wobj = np.array([[i[0], i[1]] for i in pop])
    if ref is None:
        ref = [np.max(wobj[:, 0]), np.max(wobj[:, 1])]

    total_hv = hv.hypervolume(wobj, ref)
    return total_hv


def plot_pareto_frontier(pop, hof, hv_ref, maxX=True, maxY=True):
    """function to plot pareto front"""
    Xs = [-i.fitness.wvalues[0] for i in pop]
    Ys = [-i.fitness.wvalues[1] for i in pop]
    for i in hof.keys:
        Xs.append(-i.wvalues[0])
        Ys.append(-i.wvalues[1])
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))])
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)

    '''Plotting process'''
    ax = plt.scatter(Xs, Ys)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    plt.plot(pf_X, pf_Y, c='r', label="Pareto Frontier")
    plt.scatter(hv_ref[0], hv_ref[1], c='r', marker='*')

    plt.title("Pareto Frontier")
    lgd = plt.legend(
        ["Final Population", "Pareto Frontier", "Hypervolume Reference Point"], loc='upper left', bbox_to_anchor=(1, 1))
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Makespan")
    plt.ylabel("Cost")
    plt.savefig("pareto_front.png", bbox_extra_artists=(
        lgd,), bbox_inches='tight')
