"""module for random scheduling function"""
import sys

sys.path.append('../RESEARCH/')
sys.path.append('../../RESEARCH/')

import pickle
import random
from datasets.utils import WfInstance
from datasets import utils
import numpy as np
sys.modules['utils'] = utils
from datasets.dataset_loader import load_lit_dataset


def most_work_remaining(instance: WfInstance):
    """schedule instances based on amount of work remaining, while scheduling on the machine with the least work remaining"""
    schedule = {
        "allocation": [-1 for _ in range(instance.nr_operations)],
        "order": []
    }
    scheduled = [False for _ in range(instance.nr_operations)]

    job_ope_counter = [0 for _ in range(instance.nr_jobs)]
    total_load = [0 for _ in range(instance.nr_machines)]

    for _ in range(instance.nr_operations):
        most_work = [0 for _ in range(instance.nr_jobs)]
        for i in range(instance.nr_operations):
            if scheduled[i]:
                continue
            job_nr = instance.ix_jobstage[str(i)][0]
            most_work[job_nr] += np.mean(instance.duration[i])

        job_nr = np.argmax(most_work)
        schedule['order'].append(job_nr)
        job_ope_counter[job_nr] += 1
        scheduled_op = instance.job_first_operation[job_nr] + \
            job_ope_counter[job_nr] - 1
        scheduled[scheduled_op] = True

        temp_total_load = [total_load[j] if instance.duration[scheduled_op]
                           [j] > 0 else np.inf for j in range(instance.nr_machines)]
        selected_machine = np.argmin(temp_total_load)
        total_load[selected_machine] += instance.duration[scheduled_op][selected_machine]
        schedule['allocation'][scheduled_op] = selected_machine

    return schedule


if __name__ == "__main__":

    instance: WfInstance = load_lit_dataset(
        r"C:\git-repos\RESEARCH\datasets\research\Brandimarte_Data\Text\Mk01.fjs")

    schedule = most_work_remaining(instance)
    instance.schedule = schedule
    print(instance.schedule)
