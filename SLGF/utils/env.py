
import time
import gym
import os

from datasets.dataset_loader_sdst import load_lit_dataset_sdst
cwd = os.getcwd()
import sys
sys.path.append(cwd + '/RESEARCH')
sys.path.append(cwd)
from datasets import utils
from datasets.utils import WfInstance
from datasets.dataset_loader import load_lit_dataset
sys.modules['utils'] = utils

from gym import spaces
from deap import tools, base, algorithms, creator
from deap.base import Toolbox
import numpy as np
import multiprocessing
from SLGF.utils.genetic_utils import calc_hypervolume, norm_mean, norm_min, norm_std
import random
from SLGF.utils.genetic_operations import *
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor as Pool
creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", dict, fitness=creator.Fitness)


def ext_init_run(instance, multiprocess):
    toolbox = base.Toolbox()

    if multiprocess:
        pool = multiprocessing.Pool(processes=8)
        toolbox.register("map", pool.map)

    toolbox.register("individual", create_individual,
                     creator.Individual, instance)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("evaluate", evaluate_schedule,
                     instance=instance, makespan_only=True)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", schedule_crossover)
    population = toolbox.population(n=100)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg_norm", norm_mean)
    stats.register("min_norm", norm_min)
    stats.register("std_norm", norm_std)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("std", np.std, axis=0)

    def calc_diff_time(pop, start):
        return time.time() - start

    stats.register("time", calc_diff_time, start=time.time())
    hof = tools.ParetoFront()
    return population, toolbox, stats, hof


class SLGFEnv(gym.Env):
    """SLGF Environment that follows gym interface"""

    def __init__(self, config, from_folder, instance, multiprocess=True, **kwargs):
        super(SLGFEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        # info
        self.from_folder = from_folder
        self.worst_fitness = 0
        self.best_fitness = float('inf')
        self.highest_std = 1e-5
        if from_folder:
            self.instances = [instance + i for i in os.listdir(instance)]
        else:
            self.instances = [instance]

        self.multi = multiprocess
        self.instance: WfInstance = []
        self.total_reward = 0
        self.df_list = []
        self.toolbox = None

        self.config = config
        self.data = {}

        # EA hyper params
        self.ngen = config['n_generations']
        self.population_size = config['population_size']
        self.cr = 0.8
        self.mpb = 0.2
        self.logbook = False
        self.start = time.time()

        # gym related part
        self.population = []
        self.previous_population = []
        self.hof = tools.ParetoFront()
        self.reward = 0  # total episode reward
        self.done = False  # termination
        self.episode = 0  # episode -> ngen generations
        self.gen = 0  # current gen in episode

        # tracking objects
        self.initial_solution = 0
        self.best_solution = 0
        self.current_solution = 0
        self.actions = []
        self.stagcount = 0
        self.state = []
        self.states = []

        self.deterministic_instances = config['deterministic_instances']
        if self.deterministic_instances:
            self.instance = random.choice(self.instances)
            print("instance: ", self.instance.split("/")[-1])
            if self.config['sdst']:
                with open(self.instance, "rb") as handle:
                    self.instance: WfInstance = pickle.load(handle)
            else:
                self.instance = load_lit_dataset(self.instance)
        # action and observation space
        self.action_space = spaces.Box(low=np.array([0, 0, 0], dtype=np.float32),
                                       high=np.array([1, 1, 1], dtype=np.float32))
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
                                            high=np.array([1, 1, 1, 1, 1], dtype=np.float32))

        # observation space:
        # current fitness -> normalized by min + max of seen solutions
        # mean fitness  ->  normalized by min + max of seen solutions
        # std fitness
        # remaining budget of function evlauations
        # dimension of function being solved
        # stagnation count
        #
        # histogram of normalized Ft
        # ^-> deviding entire range of values into a series of intervals, counting how many values fall into each bin
        # moving average of the histogram vecotrs ver the past G generations

        # observation space 1:
        # Change in objective value (makespan)
        # number of unique chromosomes in new generation

        # reward:
        # large makespan increase
        # large diversity increase ()

    def binary_hv(self):
        """reward function implementation of binary hypervolume increase"""
        return int(calc_hypervolume(self.population, self.instance.hv_ref)
                   > calc_hypervolume(self.previous_population, self.instance.hv_ref))

    def obj_increase(self, objs=[0, 1]):
        """reward function implementation of makespan increase"""
        reward = 0
        for obj in objs:
            if int(self.logbook[-1]['min'][obj] > self.logbook[-2]['min'][obj]):
                reward += 1
        return reward

    def abs_obj_increase(self, objs=[0, 1]):
        """reward function implementation of makespan increase"""
        reward = 0
        for obj in objs:
            reward += int(self.logbook[-2]['min']
                          [obj] - self.logbook[-1]['min'][obj])
        return reward

    def distance_reward(self, inverse=False):
        wobj = np.array([ind.fitness.wvalues for ind in self.population])
        wobj_prev = np.array(
            [ind.fitness.wvalues for ind in self.previous_population])
        wobj_hof = wobj = np.array([ind.fitness.wvalues for ind in self.hof])

        if inverse:
            ind = IGD(wobj_hof)
        else:
            ind = GD(wobj_hof)

        return int(ind(wobj) < ind(wobj_prev))

    def get_observation(self):
        """get observation space"""
        if self.logbook[-1]['max'][0] > self.worst_fitness:
            self.worst_fitness = self.logbook[-1]['max'][0]
        if self.logbook[-1]['min'][0] < self.best_fitness:
            self.best_fitness = self.logbook[-1]['min'][0]
        if self.logbook[-1]['std'][0] > self.highest_std:
            self.highest_std = self.logbook[-1]['std'][0]

        fitness_range = max(self.worst_fitness - self.best_fitness, 1)

        norm_best_fitness = (
            self.logbook[-1]['min'][0] - self.best_fitness) / fitness_range
        norm_mean_fitness = (
            self.logbook[-1]['avg'][0] - self.best_fitness) / fitness_range
        # std_fitness = self.logbook[-1]['std'][0] / self.highest_std
        std_fitness = self.logbook[-1]['std'][0]
        #remaining_budget_norm = (1-self.gen)/self.ngen
        remaining_budget_norm = 1 - (self.gen / self.ngen)
        stag_count_norm = (self.stagcount) / self.ngen

        state = np.array([
            norm_best_fitness,
            norm_mean_fitness,
            std_fitness,
            remaining_budget_norm,
            stag_count_norm,
        ])

        self.state = state
        self.states.append(state.tolist())
        return state

    def step(self, action: np.ndarray):
        self.reward = 0
        self.gen += 1
        self.stagcount += 1

        # adjust GA based on agent actions
        self.toolbox.register("mutate", makespan_mutation,
                              indpb=action[0], ope_machines=self.instance.ope_machines, ope_processing_times=self.instance.duration, setup_times=self.instance.sdst)

        self.cr = action[1]
        self.mpb = action[2]
        self.actions.append(action.tolist())

        if self.config['verbose']:
            print(self.state)
            print("time_diff: ", time.time() - self.episode_start,
                  " gen: ", self.gen, "action: ", action)

        offspring = self.toolbox.select(self.population, len(self.population))

        # vary population
        offspring = algorithms.varAnd(
            offspring, self.toolbox, self.cr, self.mpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)

        # deze moet paralel ^^ Multicore processing -> CPU
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        if self.hof:
            self.hof.update(offspring)

        self.previous_population = deepcopy(self.population)
        self.population[:] = offspring

        record = self.stats.compile(self.population) if self.stats else {}
        self.logbook.record(gen=self.gen, nevals=len(invalid_ind), **record)
        if self.config['verbose']:
            print(self.logbook.stream)

        # self.distance_reward(inverse=True)#self.binary_hv()
        reward = self.abs_obj_increase([0])
        if reward > 0:
            self.stagcount = 0
        self.total_reward += reward

        observation = self.get_observation()

        if self.gen == self.ngen:
            self.done = True

        self.data[self.episode].append(
            calc_hypervolume(self.population, self.instance.hv_ref))

        if self.config['verbose']:
            print("reward: ", reward)
            print("\n")

        if self.done and self.config['verbose']:
            print("end of episode ", self.episode,
                  " at ", round(time.time() - self.start, 2), " episode duration: ", round(time.time() - self.episode_start, 2))
            print("best solution: ",
                  self.hof[0].fitness.values[0], " total reward: ", self.total_reward)
            print("\n")
        return observation, reward, self.done, {}

    def reset(self):
        """
        first state after reset
        """

        if self.logbook:
            self.df_list.append([i['avg'] for i in self.logbook])

        # change instance
        if self.episode % 10 == 0:
            if not self.deterministic_instances:
                self.instance = random.choice(self.instances)
                if self.from_folder:
                    print("instance: ", self.instance.split("/")[-1])
                    if self.config['sdst']:
                        with open(self.instance, "rb") as handle:
                            self.instance: WfInstance = pickle.load(handle)
                    else:
                        self.instance = load_lit_dataset(self.instance)

        self.episode += 1
        if self.config['verbose']:

            print("start of episode ", self.episode,
                  " at ", round(time.time() - self.start, 2))

        self.data[self.episode] = []

        # Reset GA / state spaces
        self.worst_fitness = 0
        self.best_fitness = float('inf')
        self.stagcount = 0
        self.cr = 0.8
        self.mpb = 0.2

        # initialize everything for EA
        self.init_run()

        # init DRL
        self.gen, self.reward, self.total_reward = 0, 0, 0
        self.done = False
        return self.get_observation()  # reward, done, info can't be included

    def init_run(self):
        """initialize a run of genetic algorithm"""

        self.population, self.toolbox, self.stats, self.hof = ext_init_run(
            self.instance, self.multi)

        self.episode_start = time.time()

        # self.stats.register('hv', calc_hypervolume_stat,
        #                     ref=self.instance.hv_ref)

        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + \
            (self.stats.fields if self.stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        # with Pool(max_workers=8) as pool:
        #     fitnesses = pool.map(
        #         self.toolbox.evaluate, invalid_ind)
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if self.hof is not None:
            self.hof.update(self.population)

        record = self.stats.compile(self.population) if self.stats else {}
        self.logbook.record(gen=0, nevals=len(invalid_ind), **record)

        if self.config['verbose']:
            print(self.logbook.stream)

    def run(self, model, episodes=5):
        """
        Use a trained model to select actions:
        """

        try:
            for episode in range(episodes):
                self.done = False
                state = self.reset()
                while not self.done:
                    action = model.predict(state)
                    state, reward, self.done, _ = self.step(action[0])

            return self.hof[0].fitness.values[0], round(time.time() - self.episode_start, 2), self.hof[0]

        except KeyboardInterrupt:
            pass

    def sample(self, nr_episodes=5):
        """
        Sample random actions and run the environment
        """
        for episode in range(nr_episodes):
            self.done = False
            state = self.reset()
            print(f"start episode: {episode}, with start state: {state}")
            while not self.done:
                action = self.action_space.sample()
                state, reward, self.done, _ = self.step(action)
                print(
                    f"gen: {self.gen}, action: {action}, new state: {state}, reward: {reward:2.3f}")
                action = self.action_space.sample()
