import time
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=(
                      FutureWarning, RuntimeWarning))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import multiprocessing


import pickle

import sys
sys.path.append('../RESEARCH/')
sys.path.append('../../RESEARCH/')
from datasets import utils

from datasets.utils import WfInstance, WfInstanceIndv
sys.modules['utils'] = utils

from algorithms.evaluation.evaluation_func import *
from SLEGA.utils.genetic_operations import *
from SLEGA.utils.genetic_utils import *
import multiprocessing

from datasets.dataset_loader import *
from datasets.dataset_loader_sdst import *


cwd = os.getcwd()
sys.path.append(cwd + '/RESEARCH')
sys.path.append(cwd)


from deap import algorithms

from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt


instance: WfInstance = load_lit_dataset(
    "/home/aime/SchedulingAI/RESEARCH/datasets/research/Brandimarte_Data/Text/Mk06.fjs")

# with open(cwd + fr'/DRL/data_sdst/1005/10j_5m_070.fjs.pickle', 'rb') as f:
#     instance: WfInstance = pickle.load(f)


# init toolbox#init toolbox:
creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", dict, fitness=creator.Fitness)
toolbox = base.Toolbox()

toolbox.register("individual", create_individual,
                 creator.Individual, instance)
toolbox.register("population", tools.initRepeat, list,
                 toolbox.individual)
toolbox.register("evaluate", evaluate_schedule,
                 instance=instance, makespan_only=True)

# register genetic operations
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", schedule_crossover)
toolbox.register("mutate", makespan_mutation,
                 indpb=0.01, ope_machines=instance.ope_machines, ope_processing_times=instance.duration, setup_times=instance.sdst)

# initial population and hall of fame
pop = toolbox.population(n=100)
hof = tools.ParetoFront()


def calc_diff_time(pop, start):
    return time.time() - start


# define stats to report
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)
stats.register("time", calc_diff_time, start=time.time())

if __name__ == "__main__":

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    start = time.time()
    makespans = []

    for _ in range(20):
        print("iteration: ", _)

        # initial population and hall of fame
        pop = toolbox.population(n=100)
        hof = tools.ParetoFront()

        # define stats to report
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        stats.register("time", calc_diff_time, start=time.time())

        pop, log = algorithms.eaSimple(
            pop, toolbox, cxpb=1, mutpb=0.00, ngen=100, stats=stats, halloffame=hof, verbose=True)
        makespan = hof[0].fitness.values[0]
        makespans.append(makespan)

    print("best makespan: ", np.min(makespans))
    print(makespans)

    # evaluate_schedule(hof[0], instance=instance, vis=2, makespan_only=False)
    # print("Time taken: ", time.time() - start)
    # print("best makespan: ", makespan)
    # print([i['min'][0] for i in log])
    # print([i['avg'][0] for i in log])
    # print([i['time'] for i in log])
