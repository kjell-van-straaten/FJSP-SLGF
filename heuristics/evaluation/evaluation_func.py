import sys
import os

cwd = os.getcwd()

sys.path.append('../RESEARCH/')

from algorithms.evaluation.eval_utils import *
from datasets.utils import WfInstance


def calc_inv_cost(TLIFES, instance: WfInstance):
    INVENTORY = {
        "TOOLS": {i: 1 for i in TLIFES.keys()},
        "PALLETS": {i: 1 for i in set(instance.ope_pallet)},
        "CLAMP_TOPS": {i: 1 for i in set(instance.ope_clamp_tops)},
    }
    SUPERINV = {
        j: INVENTORY[i][j] for i in INVENTORY.keys() for j in INVENTORY[i].keys()
    }

    COST = {
        "TOOLS": {i: 300 for i in TLIFES.keys()},
        "PALLETS": {i: 500 for i in set(instance['pallet'].values())},
        "CLAMP_TOPS": {i: 20 for i in set(instance['clamp_tops'].values())},
    }
    SUPERCOST = {
        j: COST[i][j] for i in INVENTORY.keys() for j in COST[i].keys()
    }
    return SUPERINV, SUPERCOST


from pathlib import Path
import time


def evaluate_schedule(schedule: dict, instance: WfInstance, makespan_only=False, vis=False, breakdown=False, print_output=False, return_env=False, full_breakdown=False, save_path="gantt1.png", backfill=True, show=False):
    """
    vis -> 1 or 2
    breakdown -> print ocst breakdown
    print output -> print output
    return->env, returns env
    full_breakdown -> displays total cost
    """
    instance.schedule = schedule
    if not makespan_only:
        try:
            base_path = cwd.split("experiments")[0]
            with open(
                Path(base_path +
                     '/datasets/statics/tools_lifetime.pickle'),
                    'rb') as handle:
                TLIFES = pickle.load(handle)
            with open(base_path + '/datasets/statics/mtools_initial.pickle', 'rb') as f:
                INIT_TOOLS = pickle.load(f)

            with open(
                    base_path + '/datasets/statics/test.pickle',
                    'rb') as handle:
                SUPERINV, SUPERCOST = pickle.load(handle)
        except Exception as e:
            base_path = cwd.split("datasets")[0]
            with open(
                Path(base_path +
                     '/datasets/statics/tools_lifetime.pickle'),
                    'rb') as handle:
                TLIFES = pickle.load(handle)
            with open(base_path + '/datasets/statics/mtools_initial.pickle', 'rb') as f:
                INIT_TOOLS = pickle.load(f)

            with open(
                    base_path + '/datasets/statics/test.pickle',
                    'rb') as handle:
                SUPERINV, SUPERCOST = pickle.load(handle)
    else:
        INIT_TOOLS = {}
        SUPERINV = {}
        SUPERCOST = {}
        TLIFES = {}

    # config
    config = {
        "DAYSTART": 6 * 60 * 60,
        "DAYEND": 20 * 60 * 60,
        "WEEKLENGTHDAYS": 5,
        "FULLDAY": 24 * 3600,
        "INITTIME": 0,
        "NEWPALLET": 500,
        "CHANGEPALLET": 8,  # 8min
        "CHANGEWASTE": 5,  # 5min
        "CHANGETOPS": 3,  # 3min
        "CHANGEORIENTATION": 3,  # 3min
        "CHANGETOOL": 1,  # 1min
        "NEWTOOL": 5,  # 5min
        "WIPTHRESHOLD": 25,  # 25 qty units is too much.
        "WIPCOST": 10,  # 10 euro per minute of high WIP
        "FPCOST": 5,  # 5 euro per week per pallet
        "MANUALCOST": 1.5,  # 1.5 euro per minute
        "DEADLINEFACTOR": 0.5,  # cost of exceedin deadline is 50% of product price
        "MACHINEIDLE": 0,  # 1 euro per minute of machine idle
        "TRANSPORT": 50,  # 50 euros for each transportation movement between machines
        # number of pallets / weeks we need to have, 25 qty per pallet
        "PALLETWEEK": 7 * 60 * 60 * 24 * 25
    }

    config["DURATIONDAY"] = (config["DAYEND"] - config["DAYSTART"])
    config["DURATIONNIGHT"] = config["FULLDAY"] - config["DURATIONDAY"]
    job_stage_count = {i: 0 for i in range(instance.nr_jobs)}

    # calc some helping classes
    instance.schedule['order_stage'] = []
    instance.machine_schedule = {str(i): []
                                 for i in instance.machine_ix.values()}
    for job in instance.schedule['order']:
        instance.schedule['order_stage'].append((job, job_stage_count[job]))
        job_stage_count[job] += 1

    for job_stage in instance.schedule['order_stage']:
        mac = instance.schedule['allocation'][instance.jobstage_ix[job_stage]]
        instance.machine_schedule[str(mac)].append(job_stage)

    # more calcs
    machines = {i: Machine(instance.ix_machines[str(i)], i, INIT_TOOLS,
                           "Aluminum", TLIFES) for i in range(instance.nr_machines)}
    ops = {}
    for machine, machine_plan in instance.machine_schedule.items():
        machine = int(machine)
        for job_stage in machine_plan:
            operation_index = instance.jobstage_ix[job_stage]
            reqs = {'material': instance.ope_material[operation_index], 'clamp_tops': instance.ope_clamp_tops[operation_index],
                    'clamp_orientation': instance.ope_clamp_orientation[operation_index], 'pallet': instance.ope_pallet[operation_index]}

            release_date = instance.job_release_dates[job_stage[0]]

            new_operation = Operation(job_name=instance.ix_job[str(job_stage[0])],
                                      operation_name=instance.ix_operation[
                                          str(operation_index)],
                                      job_stage=job_stage, operation_index=operation_index, processing_time=instance.duration[
                                          operation_index][machine],
                                      deadline=instance.job_deadlines[job_stage[0]],
                                      tools=instance.ope_tools[operation_index],
                                      machine=machines[machine],
                                      amount=instance.job_quantity[job_stage[0]],
                                      night_setup=instance.ope_night_setup[operation_index],
                                      night_operation=instance.ope_night_operation[operation_index],
                                      reqs=reqs,
                                      release_date=release_date,
                                      prec_constraints=[str(i) for i, x in enumerate(
                                          instance.preconstr[operation_index]) if x in [1]],
                                      tool_lifetimes=instance.ope_tool_consumption[
                                          str(operation_index)],
                                      product_value=50)
            ops[(job_stage)] = new_operation

    ss = WfScheduler(instance, machines, ops, config, SUPERINV,
                     SUPERCOST, makespan_only=makespan_only)
    ss.reset_plan()
    ss.schedule_operations(backfill_on=backfill)
    ss.check_deadlines()

    if vis == 1:
        ss.visualize_plan(save_path=save_path, show=show)
    elif vis == 2:
        ss.visualize_plan2(show=show)

    if print_output:
        print(ss)

    if return_env:
        return ss
    return ss.calculate_objectives(breakdown, full_breakdown)


if __name__ == "__main__":
    from algorithms.random_scheduling import random_scheduling
    with open(
            "/home/aime/SchedulingAI/RESEARCH/datasets/company_test/instance_06_02.pickle",
            'rb') as handle:
        instance = pickle.load(handle)
        instance.schedule = {'allocation': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                             'order': [0, 1, 4, 1, 4, 5, 6, 7, 5, 3, 3, 2],
                             'order_stage': [(0, 0),
                                             (1, 0),
                                             (4, 0),
                                             (1, 1),
                                             (4, 1),
                                             (5, 0),
                                             (6, 0),
                                             (7, 0),
                                             (5, 1),
                                             (3, 0),
                                             (3, 1),
                                             (2, 0)]}
        evaluate_schedule(instance.schedule, instance,
                          vis=1, print_output=False)
