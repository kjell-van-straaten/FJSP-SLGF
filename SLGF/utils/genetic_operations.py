"""module for definition of genetic operations across individuals"""
# https://reader.elsevier.com/reader/sd/pii/S095741741000953X?token=D38DD498793E2A76EA959D75BB7BF7CB78544AE6988F424473DAF21144C40E94A7C23330FF421690C217C81EE79F9DDA&originRegion=eu-west-1&originCreation=20230108145625
from copy import deepcopy
import random
import numpy as np
import sys
sys.path.append('../RESEARCH')

from algorithms.random_scheduling import random_scheduling
from algorithms.global_local_selection import *
from algorithms.constructive_algorithm import construct_solution
from algorithms.evaluation.evaluation_func import *
from datasets.utils import WfInstance
from deap import base


# ========================= init ===========================
def create_individual(icls, instance: WfInstance):
    """function to create individual, 60% global selection, 30% local selection, 10% random scheduling."""

    number = random.random()
    if number < 0.6:
        instance.schedule = global_selection(instance)
    elif number < 0.9:
        instance.schedule = local_selection(instance)
    else:
        instance.schedule = random_scheduling(instance)
    return icls(instance.schedule)


# ============================================================== schedule level ===========================================================
def schedule_crossover(indv1: WfInstance, indv2: WfInstance):
    """schedule crossover function, uses precedence preserving crossover for OS (POX) and two-point crossover or uniform crossover for MS (MPX)"""
    indv1, indv2 = MPX(indv1, indv2)
    indv1, indv2 = POX(indv1, indv2)
    return indv1, indv2


def makespan_mutation(schedule: WfInstance, indpb, ope_machines, ope_processing_times, setup_times):
    """makespan mutation function, uses random mutation for OS and twopoint crossover for MS, only focussed on makespan"""
    schedule = swap_mutation(schedule, indpb)
    schedule = greedy_proc_time_mutation(
        schedule, indpb, ope_machines=ope_machines, ope_processing_times=ope_processing_times, setup_times=setup_times)
    return schedule,


def schedule_mutation(indv1: dict, fjsp_instance: WfInstance, indpb, ope_machines, ope_processing_times, setup_times):
    """schedule_mutation function, uses 3 os mutations and 3 ms mutations"""
    os_muts = 3
    prob1 = random.random()
    prob2 = random.random()

    # os mutations
    if prob1 < 1 / os_muts:
        indv1 = grd_wip_mutation_os(indv1, indpb)
    elif prob1 < 2 / os_muts:
        indv1 = swap_mutation(indv1, indpb)
    else:
        indv1 = grd_deadline_mutation_os(indv1, fjsp_instance, indpb)

    ms_muts = 3
    if prob2 < 1 / ms_muts:
        indv1 = grd_wip_mutation_ms(
            indv1, indpb, fjsp_instance.ix_jobstage, ope_machines)
    elif prob2 < 2 / ms_muts:
        indv1 = greedy_proc_time_mutation(indv1, indpb, ope_machines=ope_machines,
                                          ope_processing_times=ope_processing_times, setup_times=setup_times)
    else:
        indv1 = grd_addition_mutation_ms(indv1, indpb, fjsp_instance)

    return indv1,

# ============================================================= operation level ===========================================================


# ------------------------------------------------------------ crossover -----------------------------------------------------------

def MPX(schedule1: dict, schedule2: dict):
    "MPX: Machine string, 50% two poitn crossover, 50% uniform crossover"

    if random.random() < 0.5:
        (schedule1, schedule2) = two_point_crossover(schedule1, schedule2)
    else:
        (schedule1, schedule2) = uniform_crossover(schedule1, schedule2)

    return (schedule1, schedule2)


def uniform_crossover(schedule1: dict, schedule2: dict):
    "MS: uniform crossover, applied to machine allocation string"
    allocation1 = schedule1['allocation']
    allocation2 = schedule2['allocation']

    size = len(allocation1)
    for i in range(size):
        if random.random() < 0.5:
            allocation1[i], allocation2[i] = allocation2[i], allocation1[i]

    schedule1['allocation'] = allocation1
    schedule2['allocation'] = allocation2

    return (schedule1, schedule2)


def two_point_crossover(schedule1: dict, schedule2: dict):
    "MS: twopoint crossover, applied to machine allocation string"
    allocation1 = schedule1['allocation']
    allocation2 = schedule2['allocation']

    size = len(allocation1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    allocation1[cxpoint1:cxpoint2], allocation2[cxpoint1:cxpoint2] \
        = allocation2[cxpoint1:cxpoint2], allocation1[cxpoint1:cxpoint2]

    schedule1['allocation'] = allocation1
    schedule2['allocation'] = allocation2

    return (schedule1, schedule2)


def POX(schedule1: dict, schedule2: dict):
    "OS: precedence preservering crossover, applied to scheduling order string"
    order1 = np.array(schedule1['order'])
    order2 = np.array(schedule2['order'])

    jobs = np.unique(order1).tolist()
    nr_jobs = len(jobs)

    js1 = random.sample(
        jobs, nr_jobs // 2)

    order1[~np.isin(order1, js1)] = -1
    order2[np.isin(order2, js1)] = -1

    # duplicate p1 in j1
    remaining2 = order1[order1 != -1]
    remaining1 = order2[order2 != -1]

    counter = 0
    for i in remaining2:
        no_adjust = True
        while no_adjust:
            if order2[counter] == -1:
                order2[counter] = i
                no_adjust = False
            counter += 1

    counter = 0
    for i in remaining1:
        no_adjust = True
        while no_adjust:
            if order1[counter] == -1:
                order1[counter] = i
                no_adjust = False
            counter += 1

    schedule1['order'] = list(order1)
    schedule2['order'] = list(order2)

    return (schedule1, schedule2)


# ------------------------------------------------------------ Mutation -----------------------------------------------------------


def swap_mutation(schedule: dict, indpb: float):
    """OS: swap order of two jobs"""
    order = schedule['order']
    size = len(order)
    for i in range(size):
        if random.random() < indpb:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            order[i], order[swap_indx] = order[swap_indx], order[i]
    schedule['order'] = order
    return schedule


def greedy_proc_time_mutation(schedule: dict, indpb: WfInstance, ope_machines: list, ope_processing_times: list, setup_times: list):
    """MS: swap machine allocation"""
    allocation = schedule['allocation']
    size = len(allocation)
    op_on_machine = np.ones(
        shape=(len(ope_machines[0])), dtype=int) * -1  # machines
    setup_times = np.array(setup_times)  # shape: machine, operation, operation
    for index in range(size):
        machine = allocation[index]
        op_on_machine[machine] = index

        if random.random() < indpb:
            # shape: nr machines
            setup_time = setup_times[np.arange(
                len(ope_machines[0])), op_on_machine, index]
            setup_time[op_on_machine == -1] = 0
            ope_processing_times_matrix = np.array(
                deepcopy(ope_processing_times[index]), dtype=np.float64)
            ope_processing_times_matrix[ope_processing_times_matrix ==
                                        0.0] = math.inf
            total_processing_times = ope_processing_times_matrix + setup_time
            new_machines = np.where(
                total_processing_times == np.min(total_processing_times))[0]
            new_mas = random.choice(new_machines)
            allocation[index] = new_mas

    schedule['allocation'] = allocation
    return schedule


# ========================================================= company mutation =========================================================


def grd_wip_mutation_os(schedule: dict, indpb):
    """OS: put jobs next to eachother to minimize WIP"""

    order = schedule['order']

    for index, value in enumerate(order):
        if random.random() < indpb:
            try:
                new_location = order[0:index].index(value)
            except:
                try:
                    new_location = order[index + 1:len(order)].index(value)
                except:
                    continue
            del order[index]
            order.insert(new_location, value)
    schedule['order'] = order
    return schedule


def grd_wip_mutation_ms(indv1: WfInstance, indpb, ix_jobstage, ope_machines):
    """MS: put job on similar machine if possible, minimize logistics + wip"""
    allocation = indv1['allocation']

    for index, value in enumerate(allocation):
        if random.random() < indpb:
            job = ix_jobstage[str(index)][0]
            mach = value
            if index > 0:
                prev_job = ix_jobstage[str(index - 1)][0]
                prev_mach = indv1['allocation'][index - 1]
                if prev_job == job and prev_mach != mach:
                    if ope_machines[index][prev_mach] in [1]:
                        allocation[index] = prev_mach
                        continue

            if index < len(allocation) - 1:
                next_job = ix_jobstage[str(index + 1)][0]
                next_mach = indv1['allocation'][index + 1]
                if next_job == job and next_mach != mach:
                    if ope_machines[index][next_mach] in [1]:
                        allocation[index] = next_mach
                        continue

    indv1['allocation'] = allocation
    return indv1


def grd_deadline_mutation_os(indv1: dict, instance: WfInstance, indpb):
    """OS: move job forward if deadline is missed"""
    env = evaluate_schedule(indv1, instance, return_env=True)
    new_order = np.array(indv1['order'])

    for job_nr in indv1['order']:
        if random.random() < indpb:
            nr_stages = instance.job_nr_operations[job_nr]
            # if last stage didnt meet deadline
            if not env.operations[(job_nr, nr_stages - 1)].deadline_flag:

                new_order = np.delete(new_order, np.where(new_order == job_nr))
                new_order = np.insert(
                    new_order, 0, [job_nr for _ in range(nr_stages)])

    indv1['order'] = new_order

    return indv1


def grd_addition_mutation_ms(indv1: dict, indpb, fjsp_instance: WfInstance):
    """MS: schedule similar jobs on same machines, similarity -> same tools"""
    new_allocation = indv1['allocation']
    for ix, allocation in enumerate(indv1['allocation']):
        if random.random() < indpb:
            similarities = fjsp_instance.ope_tool_similarities[ix]
            if len(similarities) == 0:
                continue
            if sum(similarities) <= 0:
                continue

            similar_job_stage = random.choices(range(len(similarities)),
                                               weights=similarities, k=1)[0]
            similar_index = similar_job_stage

            similar_allocation = indv1['allocation'][similar_index]

            if similar_allocation != allocation:
                if random.random() < 0.5:
                    if fjsp_instance.ope_machines[ix][similar_allocation] in [1]:
                        new_allocation[ix] = similar_allocation
                else:
                    if fjsp_instance.ope_machines[ix][allocation] in [1]:
                        new_allocation[similar_index] = allocation

    indv1['allocation'] = new_allocation
    return indv1


if __name__ == "__main__":
    import pickle
    import sys
    sys.path.append('../../RESEARCH/')
    from datasets import utils
    from datasets.utils import WfInstance, WfInstanceIndv

    sys.modules['utils'] = utils
    with open(r"/home/aime/SchedulingAI/RESEARCH/datasets/company_test2/instance_85_06.pickle", "rb") as f:
        instance: WfInstance = pickle.load(f)
    instance.schedule = random_scheduling(instance)
    print(instance.schedule)
    schedule = greedy_proc_time_mutation(
        instance.schedule, 1, instance.ope_machines, instance.duration, instance.sdst)
    print(schedule)
