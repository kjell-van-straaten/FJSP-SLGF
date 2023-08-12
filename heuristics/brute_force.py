import pickle
import itertools
from random_scheduling import random_scheduling
from evaluation.evaluation_func import *
import math

with open(
        r'C:\git-repos\RESEARCH\datasets\instances\instance_00.pickle',
        'rb') as handle:
    instance = pickle.load(handle)

instance['schedule'] = random_scheduling(instance)
allocation = instance['schedule']['allocation']

order = instance['schedule']['order']
all_schedules = itertools.product([0, 1], repeat=len(allocation) - 2)
all_orders = itertools.permutations(order)

count = 0
count_2 = 0
best_makespan = 9999999999999999999999
for order in all_orders:
    instance['schedule']['order'] = order
    for schedule in all_schedules:
        schedule = list(schedule)
        schedule.append(0)
        schedule.append(1)
        instance['schedule']['allocation'] = schedule

        makespan = evaluate_schedule(
            instance)[0]

        if makespan < best_makespan:
            best_makespan = makespan
            evaluate_schedule(instance, vis=1)
            print(count / (math.factorial(49) * 2 ** 49))

        count += 1
