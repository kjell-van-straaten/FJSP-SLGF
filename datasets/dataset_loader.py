"""module to load literature datasets into kjell standard"""

import numpy as np
import sys
sys.path.append('../RESEARCH/')

import copy
from pprint import pprint

from algorithms.evaluation.evaluation_func import evaluate_schedule
from algorithms.random_scheduling import random_scheduling

from datasets.utils import WfInstance, WfInstanceIndv


def load_lit_dataset(path, read_file=True):
    wf_instance = WfInstance()
    if read_file:
        file = open(path, 'r')
        firstLine = file.readline()
        path = file.readlines()
    else:
        path = copy.deepcopy(path)
        firstLine = path.pop(0).strip()
    firstLineValues = list(map(int, firstLine.split()[0:2]))

    wf_instance.nr_jobs = firstLineValues[0]
    wf_instance.nr_machines = firstLineValues[1]

    wf_instance.ix_job = {
        str(i): f"job {i}" for i in range(wf_instance.nr_jobs)}
    wf_instance.job_ix = {v: k for k, v in wf_instance.ix_job.items()}
    wf_instance.ix_machines = {
        str(i): f"machine {i}" for i in range(wf_instance.nr_machines)}
    wf_instance.machine_ix = {v: k for k, v in wf_instance.ix_machines.items()}

    wf_instance.job_quantity = [1 for _ in range(wf_instance.nr_jobs)]

    operation_nr = 0
    duration_temp = {}
    for job in range(wf_instance.nr_jobs):
        total_duration = 0
        nr_options = 0

        currentLine = path.pop(0).strip()
        currentLineValues = list(map(int, currentLine.split()))

        wf_instance.job_nr_operations.append(currentLineValues[0])
        wf_instance.nr_operations += currentLineValues[0]

        i = 1
        stage = 0
        wf_instance.job_first_operation.append(operation_nr)
        while i < len(currentLineValues):
            nr_machines = currentLineValues[i]
            wf_instance.ix_jobstage[str(operation_nr)] = (job, stage)
            wf_instance.jobstage_ix[(job, stage)] = operation_nr
            wf_instance.ix_operation[str(
                operation_nr)] = "operation {}".format(operation_nr)
            duration_temp[operation_nr] = {}
            wf_instance.job_stages.append((job, stage))
            i = i + 1
            for _ in range(nr_machines):
                machine_ix = currentLineValues[i] - 1
                i = i + 1
                processing_time = currentLineValues[i]
                duration_temp[operation_nr][machine_ix] = processing_time
                total_duration += processing_time
                nr_options += 1
                i = i + 1

            stage += 1
            operation_nr += 1

        wf_instance.job_duration.append(total_duration / nr_options)

    duration_matrix = np.zeros(
        (wf_instance.nr_operations, wf_instance.nr_machines), dtype=float)
    operation_machines = np.zeros(
        (wf_instance.nr_operations, wf_instance.nr_machines), dtype=int
    )
    preconstr = np.zeros(
        (wf_instance.nr_operations, wf_instance.nr_operations), dtype=int
    )

    for job in range(wf_instance.nr_jobs):
        for stage in range(1, wf_instance.job_nr_operations[job]):
            preconstr[wf_instance.job_first_operation[job] + stage][
                wf_instance.job_first_operation[job] + stage - 1] = 1
    for operation_nr, machine_dict in duration_temp.items():
        for machine_ix, duration in machine_dict.items():
            duration_matrix[operation_nr][machine_ix] = duration
            operation_machines[operation_nr][machine_ix] = 1

    wf_instance.duration = duration_matrix.tolist()
    wf_instance.ope_machines = operation_machines.tolist()
    wf_instance.job_release_dates = [0 for _ in range(wf_instance.nr_jobs)]
    wf_instance.job_deadlines = [-1 for _ in range(wf_instance.nr_jobs)]
    wf_instance.ope_clamp_orientation = [
        0 for _ in range(wf_instance.nr_operations)]
    wf_instance.ope_clamp_tops = [0 for _ in range(wf_instance.nr_operations)]
    wf_instance.ope_material = [0 for _ in range(wf_instance.nr_operations)]
    wf_instance.ope_pallet = [0 for _ in range(wf_instance.nr_operations)]
    wf_instance.ope_night_operation = [
        1 for _ in range(wf_instance.nr_operations)]
    wf_instance.ope_night_setup = [
        1 for _ in range(wf_instance.nr_operations)]
    wf_instance.ope_tool_consumption = {str(i) : {} for i in range(
        wf_instance.nr_operations)}
    wf_instance.ope_tool_similarities = {str(i): {} for i in range(
        wf_instance.nr_operations)}
    wf_instance.ope_spacers = [0 for _ in range(wf_instance.nr_operations)]
    wf_instance.ope_tools = [[] for _ in range(wf_instance.nr_operations)]
    wf_instance.sdst = np.zeros(shape=(wf_instance.nr_machines, wf_instance.nr_operations,
                                       wf_instance.nr_operations)).tolist()
    wf_instance.preconstr = preconstr.tolist()

    wf_instance.config = 'lit_dataset'

    return wf_instance


if __name__ == "__main__":
    import os
    cwd = os.getcwd()
    instance: WfInstance = load_lit_dataset(
        cwd + r"/DRL/data_dev/1005/10j_5m_001.fjs")
    instance.schedule = random_scheduling(instance)
    res = evaluate_schedule(instance.schedule, instance)
    print(res)
