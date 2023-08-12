"""this module implements a constructive algorithm from the following paper:"""
# https://link.springer.com/content/pdf/10.1007/s00170-013-5510-z.pdf?pdf=button

import json
import numpy as np
import random
import pickle
import sys
sys.path.append('../RESEARCH')
from algorithms.evaluation.evaluation_func import evaluate_schedule
from datasets.utils import WfInstance
import os
cwd = os.getcwd()


def construct_solution(instance: WfInstance, is_random=True) -> dict:
    schedule = {"allocation": [], "order": []}
    if random.random() > 0.5 and is_random:
        optimizer = 1
    else:
        optimizer = 0
    # sji list: mean processing time of a job i
    # sky lits: total weighted processing time on machine y

    if instance.config == 'lit_dataset':
        sji = instance.job_duration
    else:
        sji = [instance.job_duration[i] / instance.job_nr_operations[i]
               for i in range(instance.nr_jobs)]
    order = np.argsort(sji)
    addition = True
    stage_count = 0
    while addition:
        addition = False
        for job_ix in order:
            if stage_count < instance.job_nr_operations[job_ix]:
                schedule['order'].append(job_ix)
                addition = True
        stage_count += 1

    schedule['allocation'] = [-1 for _ in range(len(schedule['order']))]

    job_stage_count = {}
    for index, job in enumerate(schedule['order']):

        stage = job_stage_count.get(job, 0)
        first_operation = instance.job_first_operation[job]
        operation = first_operation + stage
        # print("scheduling job: " + str(job) + " stage: " +
        #       str(stage) + " (operation: " + str(operation) + ")")
        machine_options = [i for i, x in enumerate(
            instance.ope_machines[operation]) if x == 1]
        objective = 9999999999999999999
        selected_machine = -1
        for machine in machine_options:
            schedule['allocation'][operation] = machine
            partial_order = schedule['order'][:index + 1]
            act_schedule = {
                "allocation": schedule['allocation'], "order": partial_order}
            objective_option = evaluate_schedule(
                act_schedule, instance)[optimizer]
            if objective_option < objective:
                objective = objective_option
                selected_machine = machine
        # print('scheduled on machine: ', selected_machine)
        schedule['allocation'][operation] = selected_machine
        job_stage_count[job] = stage + 1

    return schedule


if __name__ == "__main__":
    from datasets.dataset_loader import load_lit_dataset
    instance: WfInstance = load_lit_dataset(
        cwd + "/DRL/data_dev/1005/10j_5m_001.fjs")

    schedule = construct_solution(instance, is_random=False)
    # print(evaluate_schedule(schedule, instance, vis=1))

    print(schedule)
    # instance['schedule'] = schedule
