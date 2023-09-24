"""module for random scheduling function"""
import sys
sys.path.append('../RESEARCH/')
sys.path.append('../../RESEARCH/')

import pickle
import random
from datasets.utils import WfInstance
from datasets import utils
sys.modules['utils'] = utils


def random_scheduling(instance: WfInstance):
    """just completely random"""

    schedule = {"allocation": [], "order": []}

    order = [i for i in range(
        instance.nr_jobs) for j in range(instance.job_nr_operations[i])]
    random.shuffle(order)
    schedule['order'] = order

    allocation = []

    machine_choices_ixes = [[ix for ix, j in enumerate(
        instance.ope_machines[i]) if j == 1] for i in range(instance.nr_operations)]
    for operation in range(instance.nr_operations):
        allocation.append(random.choice(
            machine_choices_ixes[operation]))

    schedule['allocation'] = allocation

    return schedule


# with open(
#         r'C:\git-repos\RESEARCH\datasets\instances\instance_00.pickle',
#         'rb') as handle:
#     instance: WfInstance = pickle.load(handle)

# schedule = random_scheduling(instance)
# instance.schedule = schedule
# print(instance.schedule)
