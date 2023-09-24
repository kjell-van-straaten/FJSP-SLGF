import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


import sys
sys.path.append('../RESEARCH/')
from datasets import utils

from datasets.utils import WfInstance, WfInstanceIndv
sys.modules['utils'] = utils

from algorithms.evaluation.evaluation_func import *
from SLGF.utils.genetic_operations import *
from SLGF.utils.genetic_utils import *

from datasets.dataset_loader import *


from SLGF.utils.env import SLGAEnv
import os
cwd = os.getcwd()

from deap import base
from deap import creator
from deap import tools


# with open(
#         cwd + '/datasets/instances/instance_00.pickle',
#         'rb') as handle:
#     instance: WfInstance = pickle.load(handle)

# instance: WfInstance = load_lit_dataset(cwd +
#                                         "/DRL/data_test/1510/1615_15j_10m.fjs")


model_paths = [cwd + "/SLGA/save/exp_1/" +
               i for i in ["mixed/ppo_scheduling_flex.zip"]]
#model_path = cwd + "/algorithms/SLGA/save/1510/ppo_scheduling_15_10_1.zip"
val_options = ["/DRL/data_test/2010/"]


toolbox = base.Toolbox()
f = open('MKGA-val.txt', 'w')
sys.stdout = f  # Change the standard output to the file we created.
for model_path, val_path in zip(model_paths, val_options):
    val_dir = cwd + val_path
    val_paths = os.listdir(val_dir)
    val_paths = sorted([val_dir + path for path in val_paths])

    makespan_list = []
    print('model : ', model_path)
    total_duration = 0
    for i in range(10):
        print("Instance: " + str(val_paths[i]))
        instance = load_lit_dataset(val_paths[i])

        env = SLGAEnv({
            "verbose": 0,
            "n_generations": 100,
            "population_size": 100,
        }, from_folder=False,
            instance=instance,
            toolbox=toolbox,
            creator=creator,
            multiprocess=True)
        model = PPO("MlpPolicy", env, verbose=1)
        model = PPO.load(
            model_path)
        makespan, duration = env.run(model, 1)
        total_duration += duration
        makespan_list.append(makespan)
    print('average makespan: ', np.mean(makespan_list))
    print('total duration: ' + str(total_duration),
          " average: ", str(total_duration / 10))
f.close()
