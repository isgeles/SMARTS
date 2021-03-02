# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import numpy as np
import pathlib
import tensorflow as tf
import tf_agents
import time

import json
import sys
import argparse
import pickle
import time
import dill
import gym
import psutil
import ray

from smarts.zoo.registry import make
from ultra.evaluate import evaluation_check
from ultra.utils.episode import episodes


from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from ultra.env import ultra_tf_env

def main(
    scenario_info,
    num_episodes,
    max_episode_steps,
    policy_class,
    eval_info,
    timestep_sec,
    headless,
    seed,
    log_dir,
):

    total_step = 0
    finished = False

    AGENT_ID = "007" 

    # Temp parameters
    headless = True

    spec = make(locator=policy_class, 
        max_episode_steps=1000)

    # spec = make(locator=policy_class, 
    #     max_episode_steps=1000,
    #     checkpoint_dir = "./checkpoint_dir") 

    # This `max_episode_steps` applies to the environment, and goes into AgentInterface().
    # Both of these `max_episode_steps` and `checkpoint_dir` variable can be defined here or
    # in BaselineAgentSpec() definition in /SMARTS/ultra/ultra/baselines/__init__.py .
    
    # print()

    gym_env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={AGENT_ID: spec},
        scenario_info=scenario_info,
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
    )

    print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    # env_name = "ultra.env:ultra-v0"
    # env_test = suite_gym.load(
    #     environment_name = env_name,
    #     gym_kwargs = {
    #         "agent_specs":{AGENT_ID: spec},
    #         "scenario_info":scenario_info,
    #         "headless":headless,
    #         "timestep_sec":timestep_sec,
    #         "seed":seed
    #     },
    # )
    env_test = suite_gym.wrap_env(
        gym_env= gymEnv,
    )    

    ultra_env_kwargs = {
        "agent_specs":{AGENT_ID: spec},
        "scenario_info":scenario_info,
        "headless":headless,
        "timestep_sec":timestep_sec,
        "seed":seed}

    # Parallel training environment
    train_num_parallel_environments=1
    tf_env = TFPyEnvironment(
                ParallelPyEnvironment([
                         lambda: ultra_tf_env.UltraTFEnv(
                            ultra_env_kwargs
                        )
                    ]*c
                )
            )
    tf_env.seed([42]*tf_env.batch_size)
    tf_env.reset()


    print("Finished executing tfpyenvironment")
    exit()

    agent = spec.build_agent()

    # print(agent.e)

    # print("I am inside _TF 99999999999999999999 ")
    # exit()


    for episode in episodes(num_episodes, etag=policy_class, log_dir=log_dir):
        observations = env.reset()
        state = observations[AGENT_ID]
        dones, infos = {"__all__": False}, None
        episode.reset()
        experiment_dir = episode.experiment_dir

        # save entire spec [ policy_params, reward_adapter, observation_adapter]
        if not os.path.exists(f"{experiment_dir}/spec.pkl"):
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)
            with open(f"{experiment_dir}/spec.pkl", "wb") as spec_output:
                dill.dump(spec, spec_output, pickle.HIGHEST_PROTOCOL)

        while not dones["__all__"]:
            if episode.get_itr(AGENT_ID) >= 1000000:
                finished = True
                break
            evaluation_check(
                agent=agent,
                agent_id=AGENT_ID,
                policy_class=policy_class,
                episode=episode,
                log_dir=log_dir,
                max_episode_steps=max_episode_steps,
                **eval_info,
                **env.info,
            )
            action = agent.act(state, explore=True)
            observations, rewards, dones, infos = env.step({AGENT_ID: action})
            next_state = observations[AGENT_ID]

            loss_output = agent.step(
                state=state,
                action=action,
                reward=rewards[AGENT_ID],
                next_state=next_state,
                done=dones[AGENT_ID],
            )
            episode.record_step(
                agent_id=AGENT_ID,
                infos=infos,
                rewards=rewards,
                total_step=total_step,
                loss_output=loss_output,
            )
            total_step += 1
            state = next_state

        episode.record_episode()
        episode.record_tensorboard()
        if finished:
            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("intersection-single-agent")
    parser.add_argument(
        "--task", help="Tasks available : [0, 1, 2]", type=str, default="1"
    )
    parser.add_argument(
        "--level",
        help="Levels available : [easy, medium, hard, no-traffic]",
        type=str,
        default="easy",
    )
    parser.add_argument(
        "--policy",
        help="Policies available : [ppo, sac, ddpg, dqn, bdqn]",
        type=str,
        default="dqn",
    )
    parser.add_argument(
        "--episodes", help="Number of training episodes", type=int, default=1000000
    )
    parser.add_argument(
        "--max-episode-steps",
        help="Maximum number of steps per episode",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--timestep", help="Environment timestep (sec)", type=float, default=0.1
    )
    parser.add_argument(
        "--headless", help="Run without envision", type=bool, default=False
    )
    parser.add_argument(
        "--eval-episodes", help="Number of evaluation episodes", type=int, default=200
    )
    parser.add_argument(
        "--eval-rate",
        help="Evaluation rate based on number of observations",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--seed",
        help="Environment seed",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--log-dir",
        help="Log directory location",
        default="logs",
        type=str,
    )

    base_dir = os.path.dirname(__file__)
    pool_path = os.path.join(base_dir, "agent_pool.json")
    args = parser.parse_args()

    with open(pool_path, "r") as f:
        data = json.load(f)
        if args.policy in data["agents"].keys():
            policy_path = data["agents"][args.policy]["path"]
            policy_locator = data["agents"][args.policy]["locator"]
        else:
            raise ImportError("Invalid policy name. Please try again")

    # Required string for smarts' class registry
    policy_class = str(policy_path) + ":" + str(policy_locator)

    if not "_tf" in policy_locator:
        raise Exception("Tensorflow training only supports `tf_agent`-compatible \
agents from the registry. Valid `tf_agent`-compatible agents contain `_tf` in their \
policy locator. Currently, only `dqn_tf` is available from the registry.")


    # Suppress tensorflow deprecation warning messages 
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Setup GPU, if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Setup random number seeds
    tf.random.set_seed(42)
    np.random.seed(42)

    tf_agents.system.multiprocessing.handle_main(
        main(
            scenario_info=(args.task, args.level),
            num_episodes=int(args.episodes),
            max_episode_steps=int(args.max_episode_steps),
            eval_info={
                "eval_rate": int(args.eval_rate),
                "eval_episodes": int(args.eval_episodes),
            },
            timestep_sec=float(args.timestep),
            headless=args.headless,
            policy_class=policy_class,
            seed=args.seed,
            log_dir=args.log_dir,
        )
    )

# python ultra/train_tf.py --task 1 --level easy --episodes 10 --eval-episodes 5 --eval-rate 100 --policy dqn_tf --headless true
