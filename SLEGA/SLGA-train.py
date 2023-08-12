import warnings
warnings.simplefilter(action='ignore', category=(
                      FutureWarning, RuntimeWarning))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym import make
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
import os

cwd = os.getcwd()
import sys
sys.path.append(cwd + '/RESEARCH')
sys.path.append(cwd)
from heuristics.evaluation.evaluation_func import *
from SLEGA.utils.genetic_operations import *
from SLEGA.utils.genetic_utils import *
from datasets import utils

from datasets.utils import WfInstance, WfInstanceIndv
sys.modules['utils'] = utils


from datasets.dataset_loader import *


from SLEGA.utils.env import SLEGAEnv


class TensorboardCallback(BaseCallback):
    """
    custom callback
    """

    def __init__(self, verbose=0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        value = 0

        gen = self.training_env.get_attr("gen")[0]
        if gen == 99:
            value = self.training_env.get_attr("hof")[0][0].fitness.values[0]

            self.logger.record("fitness", value)
        return True


# instance: WfInstance = load_lit_dataset(
#     r"C:\git-repos\RESEARCH\DRL\data_dev\1005\10j_5m_001.fjs")


if __name__ == '__main__':
    job_list = [40]
    mas_list = [2]

    nr_jobs = 15
    nr_mas = 2
    for nr_mas, nr_jobs, in zip(mas_list, job_list):
        instance = cwd + \
            rf"/datasets/wf_instances/train/{nr_jobs}_{str(nr_mas).zfill(2)}/"
        # instance = cwd + r"/DRL/data_sdst//"
        # env = SLEGAEnv(config, instance, toolbox)

        env = make_vec_env(SLEGAEnv,
                           n_envs=1,
                           env_kwargs={
                               "instance": instance,
                               "from_folder": True,
                               "config": {
                                   "verbose": 0,
                                   "n_generations": 100,
                                   "population_size": 100,
                                   "sdst": True,
                                   "deterministic_instances": False
                               }
                           })

        val_instance = cwd + \
            rf"/datasets/company_valid/{nr_jobs}_{str(nr_mas).zfill(2)}/"

        def make_val_env(env_id):
            def _init():
                env = make(env_id, config={
                    "verbose": 0,
                    "n_generations": 100,
                    "population_size": 100,
                    "sdst": True,
                    "deterministic_instances": True,
                }, instance=val_instance, from_folder=True, multiprocess=False)
                return env
            return _init

        val_env = SubprocVecEnv(
            [make_val_env('SLGA-v0') for _ in range(1)])

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            batch_size=50,
            tensorboard_log=cwd + "/SLGA/PPO_tensorboard/",
            n_steps=100
        )

        callbacks = CallbackList([EvalCallback(val_env, best_model_save_path=cwd + f"/SLGA/save/exp_3/{nr_jobs}_{str(nr_mas).zfill(2)}/",
                                               log_path=cwd + "/SLGA/PPO_tensorboard/", eval_freq=1000,), TensorboardCallback()],
                                 )
        try:
            model.learn(total_timesteps=50000, callback=callbacks)
        except KeyboardInterrupt:
            pass
        # model_save_path = cwd + "/SLGA/save/ppo_scheduling_flex"
        model_save_path = cwd + "/SLGA/save/ppo_scheduling_" + \
            str(nr_jobs) + "_" + str(nr_mas).zfill(2)
        model.save(model_save_path)
        print(f'saved model to {model_save_path}')
