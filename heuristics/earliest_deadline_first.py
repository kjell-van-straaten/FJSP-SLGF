"""module for greedy scheduling function, based on earliest deadline"""

import pickle
from pprint import pprint
import sys
sys.path.append('../RESEARCH')

from algorithms.evaluation.evaluation_func import evaluate_schedule


def earliest_deadline_first(instance):
    """earliest deadline first, only works for company dataset, randomly selects machines"""
    schedule = {"allocation": [], "order": []}

    n = 0
    for job in instance['deadlines'].argsort():
        for _ in range(int(instance['nr_stages'][job])):
            schedule['order'].append(job)

    for job in instance['jobs'].keys():
        if n % 2 == 0:
            if 0 in instance['js_machine'][(job, 0)]:
                machine = 0
            else:
                machine = 1
        else:
            if 1 in instance['js_machine'][(job, 0)]:
                machine = 1
            else:
                machine = 0
        for _ in range(int(instance['nr_stages'][job])):
            schedule['allocation'].append(machine)
        n += 1

    return schedule


if __name__ == '__main__':
    with open(
            r'C:\git-repos\RESEARCH\datasets\instances\instance_00.pickle',
            'rb') as handle:
        instance = pickle.load(handle)
    instance['schedule'] = earliest_deadline_first(instance)
    print(instance['schedule'])
    # print(instance['schedule'])
    print(evaluate_schedule(instance, vis=0, print_output=False))


# with open(
#         r'C:\projects\Plan-it Revolve\research\datasets\instances\instance_2.pickle',
#         'wb') as handle:
#     pickle.dump(instance, handle, protocol=pickle.HIGHEST_PROTOCOL)
