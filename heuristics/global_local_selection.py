"""module for random scheduling function"""
import sys

sys.path.append('../RESEARCH/')
sys.path.append('../../RESEARCH/')

import pickle
import random
from datasets.dataset_loader import load_lit_dataset

from datasets.utils import WfInstance
from datasets import utils
sys.modules['utils'] = utils
import numpy as np
from pprint import pprint
# Global selection (Zhang et al., 2011)


def global_selection(instance: WfInstance):
    jobs_to_be_scheduled = [
        job_id for job_id in range(instance.nr_jobs)]
    machine_occupation_times = [0 for _ in range(instance.nr_machines)]
    machine_allocation = [-1 for _ in range(instance.nr_operations)]
    scheduled_operations = []
    operation_seq = []

    for job_ix in random.sample(jobs_to_be_scheduled, len(jobs_to_be_scheduled)):
        first_op_ix = instance.job_first_operation[job_ix]
        nr_ops = instance.job_nr_operations[job_ix]
        for operation_ix in range(first_op_ix, first_op_ix + nr_ops):
            updated_occupation_times = []

            for machine_ix, eligible in enumerate(instance.ope_machines[operation_ix]):
                if eligible == 0:
                    updated_occupation_times.append(np.inf)
                    continue
                updated_occupation_times.append(
                    machine_occupation_times[machine_ix] +
                    instance.duration[operation_ix][machine_ix])

            selected_machine_ix = np.argmin(updated_occupation_times)
            duration = instance.duration[operation_ix][selected_machine_ix]
            scheduled_operations.append(operation_ix)
            machine_occupation_times[selected_machine_ix] += duration
            machine_allocation[operation_ix] = selected_machine_ix
            operation_seq.append(job_ix)

    sorted_operation_sequence = [i for i in range(
        instance.nr_jobs) for _ in range(instance.job_nr_operations[i])]
    operation_sequence = random.sample(
        sorted_operation_sequence, len(sorted_operation_sequence))

    schedule = {
        "allocation": machine_allocation,
        "order": operation_seq
    }
    return schedule

# local selection (Zhang et al., 2011)


def local_selection(instance: WfInstance):
    jobs_to_be_scheduled = [
        job_id for job_id in range(instance.nr_jobs)]
    machine_allocation = [-1 for _ in range(instance.nr_operations)]
    scheduled_operations = []
    operation_seq = []

    for job_ix in random.sample(jobs_to_be_scheduled, len(jobs_to_be_scheduled)):
        machine_occupation_times = [0 for _ in range(instance.nr_machines)]
        first_op_ix = instance.job_first_operation[job_ix]
        nr_ops = instance.job_nr_operations[job_ix]
        for operation_ix in range(first_op_ix, first_op_ix + nr_ops):
            updated_occupation_times = []

            for machine_ix, eligible in enumerate(instance.ope_machines[operation_ix]):
                if eligible == 0:
                    updated_occupation_times.append(np.inf)
                    continue
                updated_occupation_times.append(
                    machine_occupation_times[machine_ix] +
                    instance.duration[operation_ix][machine_ix])

            selected_machine_ix = np.argmin(updated_occupation_times)
            duration = instance.duration[operation_ix][selected_machine_ix]
            scheduled_operations.append(operation_ix)
            machine_occupation_times[selected_machine_ix] += duration
            machine_allocation[operation_ix] = selected_machine_ix
            operation_seq.append(job_ix)

    sorted_operation_sequence = [i for i in range(
        instance.nr_jobs) for _ in range(instance.job_nr_operations[i])]
    operation_sequence = random.sample(
        sorted_operation_sequence, len(sorted_operation_sequence))

    schedule = {
        "allocation": machine_allocation,
        "order": operation_seq
    }

    return schedule


if __name__ == "__main__":
    import os
    cwd = os.getcwd()
    instance: WfInstance = load_lit_dataset(
        cwd + r"/datasets/research/Brandimarte_Data/Text/Mk01.fjs")

    schedule = global_selection(instance)
    print(schedule)
    instance.schedule = schedule
