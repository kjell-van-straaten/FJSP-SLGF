import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from stable_baselines3 import PPO


import sys
sys.path.append('../RESEARCH/')
from datasets import utils

from datasets.utils import WfInstance, WfInstanceIndv
sys.modules['utils'] = utils

from algorithms.evaluation.evaluation_func import *
from SLEGA.utils.genetic_operations import *
from SLEGA.utils.genetic_utils import *

from datasets.dataset_loader import *
from datasets.dataset_loader_sdst import load_lit_dataset_sdst

from SLEGA.utils.env import SLGAEnv
import os
cwd = os.getcwd()


# with open(
#         cwd + '/datasets/instances/instance_00.pickle',
#         'rb') as handle:
#     instance: WfInstance = pickle.load(handle)

# instance: WfInstance = load_lit_dataset(cwd +
#                                         "/DRL/data_test/1510/1615_15j_10m.fjs")


model_paths = [cwd + "/SLGA/save/exp_3/" + i for i in [
    "15_02/best_model.zip",
    "40_02/best_model.zip",
    "60_04/best_model.zip",
    "80_08/best_model.zip",
]]
model_paths.insert(0, cwd + "/SLGA/save/exp_1/1510/ppo_scheduling_15_10.zip")
case = 'test34'
# test_dir = cwd + f"/datasets/research/Hurink_Data//Text//{case}data//"
dirs = ["/home/aime/SchedulingAI/RESEARCH/datasets/company_test2/"]

test_dir = dirs[0]
test_paths = os.listdir(test_dir)
test_paths = sorted([test_dir + path for path in test_paths])
# # test_paths = [
# #     "/home/aime/SchedulingAI/RESEARCH/datasets/research/fjssp-sdst/1_Fattahi/Fattahi_setup_20.fjs"]
# nr_instances = len(test_paths)
# nr_avg = 10

# Change the standard output to the file we created.


writer = pd.ExcelWriter(
    f'./SLGA/save/exp_3/max_makespan_{case}.xlsx')  # Makespan data storage path
# writer_avg = pd.ExcelWriter(
#     f'./SLGA/save/exp_2/avg_makespan_{case}.xlsx')  # Makespan data storage path
writer_time = pd.ExcelWriter(
    f'./SLGA/save/exp_3/time_{case}.xlsx')  # time data storage path
writer_scheds = pd.ExcelWriter(
    f'./SLGA/save/exp_3/schedules_{case}.xlsx')  # time data storage path
data_file = pd.DataFrame(test_paths, columns=["file_name"])
data_file.to_excel(writer, sheet_name='Sheet1', index=False)
data_file.to_excel(writer_time, sheet_name='Sheet1', index=False)
data_file.to_excel(writer_scheds, sheet_name='Sheet1', index=False)

# data_file.to_excel(writer_avg, sheet_name='Sheet1', index=False)


try:
    counter = 0
    for (index, model_path) in enumerate(model_paths):
        print('model : ', model_path)
        # test_dir = dirs[index]
        # test_paths = os.listdir(test_dir)
        # test_paths = sorted([test_dir + path for path in test_paths])

        total_duration = 0
        makespans_avg = []
        makespans_max = []
        times = []
        scheds = []
        for i in range(len(test_paths)):
            print("Instance: " +
                  str(test_paths[i]), " counter: " + str(counter))
            temp_duration = 0
            # instance = load_lit_dataset(test_paths[i])re

            with open(test_paths[i], 'rb') as handle:
                instance: WfInstanceIndv = pickle.load(handle)
            # instance = load_lit_dataset_sdst(test_paths[i])
            env = SLGAEnv({
                "verbose": 0,
                "n_generations": 100,
                "population_size": 100,
                "deterministic_instances": False,
            }, from_folder=False,
                instance=instance,
                multiprocess=True)
            model = PPO("MlpPolicy", env, verbose=0)
            model = PPO.load(
                model_path)
            makespans = []
            makespan, duration, sched = env.run(model, 1)
            sched = str(sched)
            scheds.append(sched)
            temp_duration += duration
            total_duration += duration
            makespans.append(makespan)
            counter += 1

            makespans_avg.append(np.mean(makespans))
            makespans_max.append(np.min(makespans))
            times.append(temp_duration)

        # data = pd.DataFrame(makespans_avg, columns=[model_path])
        # data.to_excel(writer_avg, sheet_name='Sheet1',
        #               index=False, startcol=index + 1)

        data = pd.DataFrame(makespans_max, columns=[model_path])
        data.to_excel(writer, sheet_name='Sheet1',
                      index=False, startcol=index + 1)

        data = pd.DataFrame(times, columns=[model_path])
        data.to_excel(writer_time, sheet_name='Sheet1',
                      index=False, startcol=index + 1)

        data = pd.DataFrame(scheds, columns=[model_path])
        data.to_excel(writer_scheds, sheet_name='Sheet1',
                      index=False, startcol=index + 1)

except KeyboardInterrupt:
    pass
writer.close()
writer_time.close()
writer_scheds.close()
# writer_avg.close()
