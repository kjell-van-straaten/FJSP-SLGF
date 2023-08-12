import pandas as pd
# Importing the matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import colors as mcolors
from datasets.utils import WfInstance
from pprint import pprint

import pandas as pd

colors = []
for name, hex in mcolors.cnames.items():
    colors.append(name)


class Machine():
    def __init__(self, name: str, number: int, tool_dict: dict, init_material, lifetimes: dict) -> None:
        self.name = name
        self.std_tools = tool_dict
        self.tool_dict = tool_dict
        self.tools_used = {key: 0 for key in tool_dict.keys()}
        self.tools_lifetimes = lifetimes

        self.tool_RLU = {}
        for tool in tool_dict.keys():
            try:
                self.tool_RLU[tool] = lifetimes[tool]
            except:
                self.tool_RLU[tool] = 500

        self.pallet = 0
        self.orientation = 0
        self.clamp_tops = 0
        self.time_state = 0
        self.material = init_material
        self.number = number
        self.prev_op: Operation = 0
        self.capacity = 120
        self.work_instructions = []
        self.slots = []

    def check_tool(self, tool_name):
        if tool_name in self.tool_dict.keys():
            if self.tool_dict[tool_name] > 0:
                return True, True
            else:
                return False, True
        else:
            return False, False

    def __str__(self) -> str:
        return f"{self.name}"

    def reset_tools(self):
        self.tool_dict = self.std_tools.copy()

    def remove_tool(self, ignore, timestamp):
        tools_not_req = {tool: self.tools_used[tool] for tool in list(
            set(self.tools_used.keys()) - set(ignore))}

        tool_lru = min(tools_not_req, key=tools_not_req.get)
        self.tool_dict[tool_lru] -= 1
        self.work_instructions.append((timestamp, tool_lru, f'remove'))

    def remove_tool_eol(self, timestamp, tool):
        self.tool_dict[tool] = - 1
        self.work_instructions.append((timestamp, tool, 'remove-eol'))

    def insert_tool(self, tool_name, ignore, timestamp, option=""):
        # check capacity
        remaining_capacity = self.capacity - sum(self.tool_dict.values())
        if remaining_capacity > 0:
            pass
        elif remaining_capacity == 0:
            # if no capacity, remove tool
            self.remove_tool(ignore, timestamp)
        else:
            print("negative capacity impossible WTF?")

        # add tool to machine
        self.work_instructions.append(
            (timestamp, tool_name, f'insert{option}'))
        if tool_name in self.tool_dict.keys():
            self.tool_dict[tool_name] += 1
            try:
                self.tool_RLU[tool_name] += self.tools_lifetimes[tool_name]
            except:
                self.tool_RLU[tool_name] += 500
        else:
            self.tool_dict[tool_name] = 1
            try:
                self.tool_RLU[tool_name] = self.tools_lifetimes[tool_name]
            except:
                self.tool_RLU[tool_name] = 500

    def update_tools(self, tools, timestamp):
        for i in tools:
            self.tools_used[i] = timestamp

    def remove_instructions(self, timestamp):
        self.work_instructions = [
            i for i in self.work_instructions if i[0] != timestamp]

    def consume_lifetimes(self, timestamp_start, timestamp_end, tools: list):
        for tool in tools:
            self.tool_RLU[tool] -= tools[tool]

            while self.tool_RLU[tool] < 0:
                self.insert_tool(tool, tools.keys(),
                                 timestamp_start, '-lifetime')
                self.remove_tool_eol(timestamp_end, tool)
                try:
                    self.tool_RLU[tool] += self.tools_lifetimes[tool]
                except:
                    self.tool_RLU[tool] += 500


class Operation():
    def __init__(self, operation_name, job_name, job_stage, operation_index, processing_time, deadline, tools, machine: Machine, amount, night_setup, night_operation, reqs, release_date, tool_lifetimes, prec_constraints, product_value) -> None:
        # name, job/stage/operation index
        self.operation_name = operation_name
        self.job_name = job_name
        self.job_stage = job_stage
        self.job = job_stage[0]
        self.stage = job_stage[1]
        self.operation_index = operation_index
        self.machine = machine

        # information
        self.deadline = deadline
        self.processing_time = processing_time
        self.tools = tools
        self.tool_lifetimes = tool_lifetimes
        self.amount = amount
        self.night_setup = night_setup
        self.night_operation = night_operation
        self.reqs = reqs
        self.release_date = release_date
        self.value = product_value
        self.prec_constraints = prec_constraints

        # actuals
        self.start_time = 0
        self.completion_time = 0
        self.deadline_flag = True
        self.setup_time = 0
        self.end_setup_time = 0
        self.backfilled = False
        self.req_setup = []

    def __str__(self) -> str:
        return f"Job {self.job_stage[0] : <2}, stage {self.job_stage[1] : <1} starting at {self.start_time:<8.0f}, finished at {self.completion_time:<8.0f}, duration {self.completion_time - self.start_time:<8.0f}, setup_time {self.setup_time:<5.0f}, backfilled {self.backfilled:<2}, release_date {self.release_date:<8.0f}, night_setup: {self.night_setup:<2}, night_operation: {self.night_operation:<2}"


class WfScheduler():
    def __init__(self, instance: WfInstance, machines: dict[int, Machine], operations: dict[int, Operation], config: dict, inventory: dict, cost: dict, makespan_only: bool) -> None:
        self.allocation = instance.schedule['allocation']
        self.order = instance.schedule['order']
        self.order_stage = instance.schedule['order_stage']
        self.machines = machines
        self.operations = operations
        self.machine_plan = instance.machine_schedule
        self.instance = instance
        self.scheduled = {}
        self.config = config
        self.adjust_setup = []
        self.inventory = inventory
        self.cost = cost
        self.dataset_type = instance.config
        self.makespan_only = makespan_only

        for machine in machines.values():
            machine.time_state = self.config['INITTIME']

    def __str__(self):
        for machine in self.machines.values():
            print(machine, len(machine.slots))
            for job in machine.slots:
                print(job)
        return ""

    def reset_plan(self):
        """reset plan to recalculate"""
        for machine in self.machines.values():
            machine.prev_op = 0
            machine.time_state = self.config['INITTIME']
            machine.reset_tools()
            machine.work_instructions = []
            machine.slots = []

        for op in self.operations.values():
            op.start_time = 0
            op.setup_time = 0
            op.completion_time = 0
            op.end_setup_time = 0
            op.req_setup = []

    def calculate_tools(self):
        for machine in self.machines.values():
            for operation in machine.slots:
                for tool in operation.tools:
                    in_machine, created = machine.check_tool(tool)
                    if not (created):
                        machine.insert_tool(
                            tool, operation.tools, operation.start_time, " new")
                    elif not (in_machine):
                        machine.insert_tool(
                            tool, operation.tools, operation.start_time, " existing")
                machine.consume_lifetimes(
                    operation.start_time, operation.completion_time, operation.tool_lifetimes)
                machine.update_tools(
                    operation.tools, operation.completion_time)
            machine.work_instructions.sort(key=lambda tup: tup[0])

    def calculate_req_setup(self, machine: Machine, operation: Operation):
        """function to calculate required setup of operation on machine"""

        # if self.dataset_type == "lit_dataset":
        #     return 0
        if not machine.prev_op == 0:
            setup_time = self.instance.sdst[machine.number][machine.prev_op.operation_index][operation.operation_index]
        else:
            setup_time = 0
        if machine.pallet != operation.reqs['pallet']:
            machine.work_instructions.append(
                (machine.time_state, operation.reqs['pallet'], 'build'))

        if machine.orientation != operation.reqs['clamp_orientation']:
            machine.orientation = operation.reqs['clamp_orientation']
            machine.work_instructions.append(
                (machine.time_state, operation.reqs['clamp_orientation'], 'set'))

        if machine.clamp_tops != operation.reqs['clamp_tops']:
            machine.clamp_tops = operation.reqs['clamp_tops']
            machine.work_instructions.append(
                (machine.time_state, operation.reqs['clamp_tops'], 'install'))

        if machine.material != operation.reqs['material']:
            machine.material = operation.reqs['material']
            machine.work_instructions.append(
                (machine.time_state, operation.reqs['material'], 'waste flow'))

        return setup_time

    def add_setup(self, o1: Operation, o2: Operation, machine: Machine, timestamp):
        """function to add required setup to work instructions"""
        if o2.reqs['pallet'] != o1.reqs['pallet']:
            machine.work_instructions.append(
                (timestamp, o2.reqs['pallet'], 'build'))
        if o2.reqs['clamp_orientation'] != o1.reqs['clamp_orientation']:
            machine.work_instructions.append(
                (timestamp, o2.reqs['clamp_orientation'], 'set'))
        if o2.reqs['clamp_tops'] != o1.reqs['clamp_tops']:
            machine.work_instructions.append(
                (timestamp, o2.reqs['clamp_tops'], 'install'))
        if o2.reqs['material'] != o1.reqs['material']:
            machine.work_instructions.append(
                (timestamp, o2.reqs['material'], 'waste flow'))

    def adjust_start_next_day(self, start_time, rel_start_time):
        """function to move operation to next day if it crosses night time"""
        if rel_start_time < self.config['DAYSTART']:
            adjusted_time = start_time - \
                rel_start_time + self.config['DAYSTART']
        elif rel_start_time > self.config['DAYSTART']:
            adjusted_time = start_time + \
                self.config['FULLDAY'] - rel_start_time + \
                self.config['DAYSTART']
        else:
            adjusted_time = start_time
        return adjusted_time

    def check_option_between_ops(self, operation: Operation, oo1: Operation, oo2: Operation, start_time):
        """function to check if operation can be scheduled between two operations"""

        if oo1.operation_name == 'dummy_op':
            setup1 = 0
        else:
            setup1 = self.instance.sdst[operation.machine.number][oo1.operation_index][operation.operation_index]
        setup2 = self.instance.sdst[operation.machine.number][oo2.operation_index][operation.operation_index]
        prop_start = start_time % self.config['FULLDAY']

        prop_finish = (start_time + setup1 +
                       operation.processing_time) % self.config['FULLDAY']

        prop_abs_start_o2 = (oo2.start_time + oo2.setup_time -
                             setup2)
        prop_start_o2 = prop_abs_start_o2 % self.config['FULLDAY']
        finish = (start_time + setup1 + operation.processing_time)

        schedule = True

        if self.dataset_type != "lit_dataset":
            # check whether operation can be scheduled between two operations

            # if not start during day
            if not (prop_start >= self.config['DAYSTART'] and prop_start <= self.config['DAYEND']):
                # if cannot run OR start at night
                if not (operation.night_setup == 1 and operation.night_operation == 1):
                    schedule = False

            # if not finish during day
            if not ((prop_finish >= self.config['DAYSTART']) and (prop_finish <= self.config['DAYEND'])):
                # if not run at night
                if not (operation.night_operation == 1):
                    schedule = False

            # if o2 start during day
            if not (prop_start_o2 >= self.config['DAYSTART'] and prop_start_o2 <= self.config['DAYEND']) and (oo2.night_setup == 0 or oo2.night_operation == 0):
                schedule = False
        for prec_op_index in operation.prec_constraints:
            prec_op_js = self.instance.ix_jobstage[prec_op_index]
            prec_satis = self.scheduled[prec_op_js]
            if prop_start < prec_satis:
                schedule = False

        if finish > oo2.start_time:
            schedule = False
        elif start_time < oo1.completion_time:
            schedule = False
        # if oo1.completion_time + operation.processing_time <= oo2.start_time:
        #     if len(operation.prec_constraints) > 1:
        #         print(self.scheduled[operation.prec_constraints[0]])
        #     print(prop_start)
        #     print(operation.processing_time)
        #     print(oo2.start_time)
        #     print(schedule)
        #     print("\n")
        return schedule

    def calc_start_time_options(self, operation: Operation, oo1: Operation, oo2: Operation, min_start_time):
        options = []
        if (oo1.completion_time >= min_start_time):
            options.append(oo1.completion_time)

        if oo1.operation_name == 'dummy_op':
            setup1 = 0
        else:
            setup1 = self.instance.sdst[operation.machine.number][oo1.operation_index][operation.operation_index]
        setup2 = self.instance.sdst[operation.machine.number][oo2.operation_index][operation.operation_index]
        total_required_time = setup1 + setup2 + \
            operation.processing_time - oo2.setup_time
        oo2end = oo2.start_time - total_required_time
        if (oo2end >= min_start_time):
            options.append(oo2end)

        ts1 = oo1.completion_time
        ts2 = oo2end
        for i in range(50):
            day_start = 6 * 60 * 60 + 24 * 3600 * i
            if day_start > ts1 and day_start < ts2:
                if day_start >= min_start_time:
                    options.append(day_start)

        options = sorted(options)
        # add options between oo1.completion and oo2end, i.e., when dayswitch for example.
        return options

    def check_backfill(self, machine: Machine, operation: Operation, loc, orig_start_time, min_start_time):
        """function to check if the operation can be backfilled anywhere"""
        backfill = False
        verbose = False
        if operation.job_stage == (7, 1):
            verbose = True
        # check if we can backfill
        for i in range(len(machine.slots)):
            if i == 0:
                oo1 = Operation('dummy_op', 'dummy_job', (-1, -1), -1, 0, -1, {}, 0,
                                50, 1, 1, -1, -5, {}, {}, 50)
            else:
                oo1: Operation = machine.slots[i - 1]
            oo2: Operation = machine.slots[i]

            options = self.calc_start_time_options(
                operation, oo1, oo2, min_start_time)

            result = False
            for start_time in options:
                result = self.check_option_between_ops(
                    operation, oo1, oo2, start_time)

                if result:
                    break

            if not result:
                continue

            if oo1.operation_name == 'dummy':
                setup1 = 0
            else:
                setup1 = self.instance.sdst[operation.machine.number][oo1.operation_index][operation.operation_index]
            setup2 = self.instance.sdst[operation.machine.number][oo2.operation_index][operation.operation_index]

            # adjust setup after backfill
            machine.remove_instructions(oo2.start_time)
            oo2.start_time = oo2.start_time - setup2 + oo2.setup_time
            oo2.setup_time = setup2
            self.add_setup(operation, oo2, machine, oo2.start_time)

            loc = i
            backfill = True

            return loc, backfill, start_time, setup1
        return loc, backfill, orig_start_time, 0

    def schedule_operations(self, backfill_on=True):
        """schedule operations according to order_stage and allocation lists"""
        for job_stage in self.order_stage:
            operation_nr = self.instance.jobstage_ix[job_stage]
            # print('scheduling: ', job_stage)
            machine_nr = self.allocation[operation_nr]

            # extract operation and machine objects
            operation: Operation = self.operations[job_stage]
            machine: Machine = self.machines[machine_nr]

            # calculate earliest start according to prereq. constraints
            prec_satis = 0
            for prec_op_index in operation.prec_constraints:

                prec_op_js = self.instance.ix_jobstage[prec_op_index]
                prec_satis2 = self.scheduled[prec_op_js]
                prec_satis = max(prec_satis, prec_satis2)

            # only check release date for first stage
            if job_stage[1] == 0:
                min_start_time = max(prec_satis, operation.release_date)
            else:
                min_start_time = prec_satis

            # calculate start_time
            start_time = max(machine.time_state, min_start_time)

            # location to schedule job
            loc = len(machine.slots)

            # check for backfill
            if backfill_on:
                loc, backfill, start_time, setup_time = self.check_backfill(
                    machine, operation, loc, start_time, min_start_time)

            else:
                backfill = False
            # calculate relative start time (i.e., hour in the day)
            rel_start_time = start_time % self.config['FULLDAY']

            # check whether start time works, between dayend and start
            start_ok = (rel_start_time < self.config['DAYEND']) and (
                rel_start_time > self.config['DAYSTART']) or (operation.night_setup == 1 and operation.night_operation == 1)

            # calculate setup time
            if not (backfill):
                setup_time = self.calculate_req_setup(machine, operation)

            # calc expected completion
            end_setup_time = start_time + setup_time
            completion_time = end_setup_time + operation.processing_time
            rel_completion_time = completion_time % self.config['FULLDAY']

            # check whether end is ok, if we dont need operator we can ignore
            end_ok = ((rel_completion_time <= self.config['DAYEND']) and (
                rel_completion_time >= self.config['DAYSTART'])) or operation.night_operation == 1

            # if start or completion is not ok, we adjust operation to start next day.
            if self.dataset_type == "lit_dataset":
                operation.start_time = start_time
            elif not (start_ok and end_ok):
                operation.start_time = self.adjust_start_next_day(
                    start_time, rel_start_time)
            else:
                operation.start_time = start_time

            # calculate final end-setup and completion
            operation.end_setup_time = operation.start_time + setup_time
            operation.setup_time = setup_time
            operation.completion_time = operation.end_setup_time + operation.processing_time

            # update time state and prev operation of machines
            if not (backfill):
                machine.time_state = operation.completion_time
                machine.prev_op = operation
            else:
                operation.backfilled = True

            # add operation to list of scheduled ops.
            self.scheduled[job_stage] = operation.completion_time
            machine.slots.insert(loc, operation)

        # fix tool operation scheduling
        self.calculate_tools()

    def visualize_plan(self, save_path="gantt1.png", show=False):
        """function to visualize plan"""
        nr_machines = len(self.machines)
        bb_width = 50
        bar_start = 10
        bar_width = 5

        # Declaring a figure "gnt"
        fig, gnt = plt.subplots()

        fig.set_figwidth(20)
        fig.set_figheight(8)
        # Setting Y-axis limits
        gnt.set_ylim(0, 25 + 10 * nr_machines)

        # Setting X-axis limitsWW

        # Setting labels for x-axis and y-axis
        gnt.set_xlabel('Seconds since start')
        gnt.set_ylabel('Machine')

        # Setting ticks on y-axis
        gnt.set_yticks([15 + 10 * i for i in range(nr_machines)])
        # Labelling tickes of y-axis
        gnt.set_yticklabels(
            [f"{i.name}-{i.number+1}" for i in self.machines.values()])

        # Setting graph attribute
        gnt.grid(True)

        bar_job_map = {}

        iterator = 0
        max_completion = 0
        # for each job, create a bar for job switch setup
        for (machine, schedule) in self.machine_plan.items():

            for job_stage in schedule:
                op: Operation = self.operations[job_stage]
                dl_color = 'tab:green'
                # if deadline is not met, change to red
                if not op.deadline_flag:
                    dl_color = 'tab:red'
                if self.dataset_type == "lit_dataset":
                    gnt.broken_barh([(op.start_time, op.completion_time - op.start_time)], (bar_start * (
                        op.machine.number + 1), bar_width), facecolors=("tab:blue", dl_color))
                    gnt.text((op.start_time + op.completion_time) / 2, (bar_start *
                                                                        (op.machine.number + 1)) - 2.5, f'{op.job}', fontsize=10)
                else:
                    gnt.broken_barh([(op.start_time, op.end_setup_time - op.start_time), (op.end_setup_time, op.completion_time -
                                    op.end_setup_time - bb_width)], (bar_start * (op.machine.number + 1), bar_width), facecolors=("tab:blue", dl_color))
                    gnt.text((op.end_setup_time + op.completion_time) / 2, (bar_start *
                                                                            (op.machine.number + 1)) - 2.5, f'{op.job}', fontsize=10)
                bar_job_map[iterator] = op
                iterator += 1

                max_completion = max(op.completion_time, max_completion)

        offset = 500
        if self.dataset_type == "lit_dataset":
            offset = 50

        gnt.set_xlim(0,
                     max_completion + offset)

        # annotation on hover
        annot = gnt.annotate("test", xy=(15000, 15), xytext=(20, 30), textcoords="offset points",
                             bbox=dict(boxstyle="round",
                                       fc="yellow", ec="b", lw=2),
                             arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        for i in range(7):
            gnt.plot([self.config["DAYSTART"] + i * self.config["FULLDAY"],
                     self.config["DAYSTART"] + i * self.config["FULLDAY"]], [0, 10 + nr_machines * 10], 'k-')
            gnt.plot([self.config["DAYEND"] + i * self.config["FULLDAY"],
                     self.config["DAYEND"] + i * self.config["FULLDAY"]], [0, 10 + nr_machines * 10], 'k-')

        def update_annot(brokenbar_collection, op: Operation, ind, x, y):
            annot.xy = (x, y)
            box = brokenbar_collection.get_paths()[ind].get_extents()
            if ind == 1:
                text = f"""ON timeness of {op.job_stage} ({op.operation_index}) {(op.deadline - op.completion_time):.1f}, 
                deadline is {op.deadline:.1f}, \n while job is complete at {op.completion_time:.1f} 
                \n fabricate {op.amount} units, operator: {op.night_flexibility}"""
            elif ind == 0:
                setups = '\n'.join(op.reqs)
                text = f"Setup of job {op.operation_index}: setups: \n {setups} "
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.9)

        def hover(event):

            vis = annot.get_visible()
            if event.inaxes == gnt:
                for ix, brokenbar_collection in enumerate(gnt.collections):
                    cont, ind = brokenbar_collection.contains(event)
                    if cont:
                        update_annot(
                            brokenbar_collection, bar_job_map[ix], ind['ind'][0], event.xdata, event.ydata)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        if show:
            plt.show()
        plt.savefig(save_path)
        plt.close()

    def visualize_plan2(self, show=False):

        nb_row = len(self.machine_plan.keys())

        pos = np.arange(0.5, nb_row * 0.5 + 0.5, 0.5)

        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(111)

        index = 0
        max_len = []
        for (machine, schedule) in self.machine_plan.items():
            for job_stage in schedule:
                op = self.operations[job_stage]

                max_len.append(op.completion_time)
                c = random.choice(colors)
                rect = ax.barh((index * 0.5) + 0.5, op.completion_time - op.start_time, left=op.start_time, height=0.3, align='center',
                               edgecolor=c, color=c, alpha=0.8)

                # adding label
                width = int(rect[0].get_width())
                Str = "OP_{}".format(job_stage)
                xloc = op.start_time + 0.50 * width
                clr = 'black'
                align = 'center'

                yloc = rect[0].get_y() + rect[0].get_height() / 2.0
                ax.text(xloc, yloc, Str, horizontalalignment=align,
                        verticalalignment='center', color=clr, weight='bold',
                        clip_on=True)
            index += 1

        ax.set_ylim(ymin=-0.1, ymax=nb_row * 0.5 + 0.5)
        ax.grid(color='gray', linestyle=':')
        ax.set_xlim(0, max(10, max(max_len)))

        labelsx = ax.get_xticklabels()
        plt.setp(labelsx, rotation=0, fontsize=10)

        locsy, labelsy = plt.yticks(pos, self.machine_plan.keys())
        plt.setp(labelsy, fontsize=14)

        font = font_manager.FontProperties(size='small')
        ax.legend(loc=1, prop=font)

        ax.invert_yaxis()

        plt.title("Flexible Job Shop Solution")
        plt.savefig('gantt.png')
        if show:
            plt.show()
        plt.close()

    def check_deadlines(self):
        """function to check whether deadlines are met"""
        for op in self.operations.values():
            if op.completion_time > op.deadline:
                op.deadline_flag = False

    def calculate_inventory(self):
        """function to check inventory"""
        wip_inventory = {}
        fp_inventory = 0
        for job in range(self.instance.nr_jobs):
            last_stage = (job, self.instance.job_nr_operations[job] - 1)
            if last_stage not in self.instance.schedule['order_stage']:
                continue
            fp = max(self.operations[last_stage].deadline -
                     self.operations[last_stage].completion_time, 0) * self.operations[last_stage].amount
            fp_inventory += fp

        for job in range(self.instance.nr_jobs):
            for i in range(self.instance.job_nr_operations[job] - 1):
                first_stage = (job, i)
                second_stage = (job, i + 1)

                if second_stage not in self.instance.schedule['order_stage']:
                    continue
                wip_start = self.operations[first_stage].completion_time
                wip_end = self.operations[second_stage].start_time

                wip_inventory[wip_start -
                              0.0001] = self.operations[first_stage].amount
                wip_inventory[wip_end + 0.0001] = - \
                    self.operations[first_stage].amount

        if len(wip_inventory) == 0:
            return 0, fp_inventory / self.config['PALLETWEEK']

        wip_df = pd.DataFrame.from_dict(wip_inventory, 'index').reset_index()
        wip_df = wip_df.sort_values(by='index')
        wip_df['inv'] = wip_df[0].cumsum()
        wip_df['timediff'] = wip_df['index'].diff(1)

        threshold = self.config['WIPTHRESHOLD']

        wip_too_much = wip_df.query('inv > @threshold')

        total_wip = sum(wip_too_much['timediff'] * wip_too_much['inv'])

        return total_wip / 60, fp_inventory / self.config['PALLETWEEK']

    def calc_req_resources(self):
        resource_dict = {}
        prev_dict = {}
        max_dict = {}
        all_instructions = [(i[0], i[1], i[2], machine.number)
                            for machine in self.machines.values() for i in machine.work_instructions]

        all_instructions.sort(key=lambda tup: tup[0])

        for instruction in all_instructions:
            if type(instruction[1]) not in [str, tuple]:
                if math.isnan(instruction[1]):
                    continue
                elif instruction[1] == 'nan':
                    continue

            if instruction[2] not in ['waste flow', 'insert new', 'insert existing', 'remove', 'insert-lifetime', 'remove-eol', 'set', 'logistics']:
                if (instruction[3], instruction[2]) in prev_dict.keys():
                    resource_dict[prev_dict[(
                        instruction[3], instruction[2])]] -= 1
                prev_dict[(instruction[3], instruction[2])] = instruction[1]

                if instruction[1] in resource_dict.keys():
                    resource_dict[instruction[1]] += 1
                else:
                    resource_dict[instruction[1]] = 1

                if instruction[1] not in max_dict.keys():
                    max_dict[instruction[1]] = 1
                elif resource_dict[instruction[1]] > max_dict[instruction[1]]:
                    max_dict[instruction[1]] = resource_dict[instruction[1]]

        all_tools = [(i.start_time, i.tools, "add")
                     for i in self.operations.values()]
        all_tools_remove = [(i.completion_time - 1, i.tools, "remove")
                            for i in self.operations.values()]
        all_tools.extend(all_tools_remove)
        all_tools.sort(key=lambda tup: tup[0])

        tool_req = {}
        tool_max_req = {}

        for tool_event in all_tools:
            for tool in tool_event[1]:
                if tool_event[2] == "add":
                    if not tool in tool_req.keys():
                        tool_req[tool] = 1
                    else:
                        tool_req[tool] += 1
                elif tool_event[2] == "remove":
                    tool_req[tool] -= 1

                if tool not in tool_max_req.keys():
                    tool_max_req[tool] = 1
                elif tool_req[tool] > tool_max_req[tool]:
                    tool_max_req[tool] = tool_req[tool]

        return max_dict, tool_max_req

    def calculate_manual_time(self, breakdown=False):
        time_manual = [0, 0, 0, 0, 0, 0]
        for i in self.machines.values():
            for instruction in i.work_instructions:
                if instruction[2] == 'build':
                    time_manual[0] += self.config['CHANGEPALLET']
                elif instruction[2] == 'set':
                    time_manual[1] += self.config['CHANGEORIENTATION']
                elif instruction[2] == 'install':
                    time_manual[2] += self.config['CHANGETOPS']
                elif instruction[2] == 'waste flow':
                    time_manual[3] += self.config['CHANGEWASTE']
                elif instruction[2] == 'insert new':
                    time_manual[4] += self.config['NEWTOOL']
                else:
                    time_manual[5] += self.config['CHANGETOOL']
        if breakdown:
            print(time_manual)
        return sum(time_manual)

    def calculate_logistics(self):
        """function to calculate what logistics need to happen between stages"""
        old_stage = False
        transport_list = []
        for job_nr in range(self.instance.nr_jobs):
            for stage_nr in range(self.instance.job_nr_operations[job_nr]):
                job_stage = (job_nr, stage_nr)
                if job_stage not in self.instance.schedule['order_stage']:
                    continue
                operation: Operation = self.operations[job_stage]

                if not (old_stage):
                    old_stage = operation
                    continue
                if operation.job_stage[0] == old_stage.job_stage[0]:
                    if operation.machine != old_stage.machine:
                        transport_list.append(
                            (operation.job_name, old_stage.machine.number, operation.machine.number, old_stage.completion_time, operation.start_time))
                old_stage = operation

        logistic_options = {(j, k): {i[4]: [] for i in transport_list if i[1] == j and i[2] == k}
                            for j in self.machines.keys() for k in self.machines.keys() if j != k}
        logistic_options_sorted = {k: np.sort(
            list(logistic_options[k].keys())) for k in logistic_options}

        transport_list = sorted(transport_list, key=lambda tup: tup[4])
        for i in transport_list:
            # allocate a logistic movement to the latest possible moment, unless an earlier moment is available which is already been used
            allocated = False
            ops = logistic_options_sorted[(i[1], i[2])]
            difference_array = ops - i[4]
            select = np.where(difference_array < 0,
                              difference_array, -np.inf).argmax()

            for index, j in enumerate(ops):
                if j > i[3] and len(logistic_options[(i[1], i[2])][j]) > 0 and index < select:
                    allocated = True
                    logistic_options[(i[1], i[2])][j].append(i[0])
                    break
            if allocated:
                continue

            timestamp = ops[select]
            logistic_options[(i[1], i[2])][timestamp].append(i[0])

        for m1_m2 in logistic_options.keys():
            for option in logistic_options[m1_m2].keys():
                if len(logistic_options[m1_m2][option]) > 0:

                    self.machines[m1_m2[0]].work_instructions.append(
                        (option, (logistic_options[m1_m2][option], self.instance.ix_machines[str(m1_m2[1])]), 'logistics'))

        for key in self.machines.values():
            key.work_instructions.sort(key=lambda tup: tup[0])
        return len([k for i in logistic_options.keys() for k in logistic_options[i].keys() if len(logistic_options[i][k]) > 0])

    def calculate_time_offline(self, makespan):
        total_time = makespan * len(self.machines.keys())
        total_effective_time = 0
        for operation in self.operations.values():
            total_effective_time += operation.completion_time - operation.end_setup_time
        return total_time - total_effective_time

    def calculate_objectives(self, breakdown=False, all_breakdown=False) -> tuple[int]:
        "function that evaluates the plan, returning objective value"
        # calculate makespan

        makespan = max([i.completion_time for i in self.operations.values()])
        if self.makespan_only:
            return makespan, 0

        # calculate operational cost
        cost_breakdown = {}

        cost_deadlines = 0
        self.check_deadlines()
        for job_stage in self.instance.schedule['order_stage']:
            op = self.operations[job_stage]
            if not (op.deadline_flag) and (self.instance.job_nr_operations[job_stage[0]] - 1 == job_stage[1]):
                cost_deadlines += self.config['DEADLINEFACTOR'] * op.value
        cost_breakdown['DEADLINES'] = cost_deadlines

        manual_cost = self.calculate_manual_time(
            breakdown=all_breakdown) * self.config['MANUALCOST']
        cost_breakdown['MANUALCOST'] = manual_cost

        cost_addition = 0
        max_dict, tool_max_req = self.calc_req_resources()
        for dict_ in [max_dict, tool_max_req]:
            for key, value in dict_.items():
                try:
                    missing_amount = max(value - self.inventory[key], 0)
                except:
                    # print(f'missing initial amount for {key}')
                    missing_amount = value * 0.99
                try:
                    missing_value = missing_amount * self.cost[key]
                except:
                    # print(f'missing cost for {key}')
                    missing_value = missing_amount * 200
                if all_breakdown and missing_value > 0:
                    print(f"{key}, {missing_value}")
                cost_addition += missing_value

        cost_breakdown['COSTADDITION'] = cost_addition

        cost_logistics = self.config['TRANSPORT'] * self.calculate_logistics()
        cost_breakdown["COSTLOGISTICS"] = cost_logistics

        cost_offline = self.config['MACHINEIDLE'] * \
            self.calculate_time_offline(makespan) / 60
        cost_breakdown["COSTOFFLINE"] = cost_offline

        wip, fp = self.calculate_inventory()
        cost_wip = wip * self.config['WIPCOST']
        cost_breakdown['COSTWIP'] = cost_wip

        cost_fp = fp * self.config['FPCOST']
        cost_breakdown['COSTFP'] = cost_fp

        total_cost = cost_deadlines + manual_cost + cost_wip + \
            cost_fp + cost_addition + cost_logistics + cost_offline
        # print(f"Makespan; {makespan:.0f}, cost: {total_cost:.0f}")

        if breakdown:
            pprint(cost_breakdown)        
        if breakdown:
            print("makespan: ", makespan)
            total_setup_time = 0
            for i in self.operations.values():
                total_setup_time += i.end_setup_time - i.start_time
            print("total setup time: ", total_setup_time)
        

        return (round(makespan), round(total_cost))
