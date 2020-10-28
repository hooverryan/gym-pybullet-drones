import os
import time
import argparse
import pdb
import math
import numpy as np
import pybullet as p
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env

from utils import *
from gym_pybullet_drones.envs.RLCrazyFlieAviary import RLCrazyFlieAviary

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##########################################
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionCtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DroneModel.CF2X,    type=lambda model: DroneModel[model],   help='Drone model (default: CF2X)', metavar='')
    parser.add_argument('--physics',            default=Physics.PYB,        type=lambda phy: Physics[phy],          help='Physics updates (default: PYB)', metavar='')
    parser.add_argument('--PID_Control',        default=False,              type=str2bool,                          help='Whether to use a PID Control (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,               type=str2bool,                          help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,              type=str2bool,                          help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=200,                type=int,                               help='Simulation frequency in Hz (default: 200)', metavar='')
    parser.add_argument('--test_duration',      default=10,                 type=int,                               help='Duration of the test simulation in seconds (default: 10)', metavar='')
    parser.add_argument('--run_name',           default='',                 type=str,                               help='Name for video files and frames (default: empty string)', metavar='')
    parser.add_argument('--training_steps',     default=500000,             type=int,                               help='Number of training steps before saving the model (default: 500000)', metavar='')
    parser.add_argument('--training_loops',     default=10,                 type=int,                               help='Number of training loops (default: 10)', metavar='')
    parser.add_argument('--max_wind_speed',     default=0,                  type=float,                             help='Maximum wind speed (requires Physics.PYB_WIND model) (default: 0)', metavar='')
    ARGS = parser.parse_args()

    #### Check the environment's spaces ################################################################
    #env = gym.make("rl-CrazyFlie-aviary-v0")
    env = RLCrazyFlieAviary(drone_model=ARGS.drone, Physics=ARGS.physics, freq=ARGS.simulation_freq_hz,
         gui=ARGS.gui, record=False, PID_Control=ARGS.PID_Control)
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)
    check_env(env, warn=True, skip_render_check=True) 

    #### Train the model ###############################################################################
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions), dt = 1/ARGS.simulation_freq_hz)
    model = DDPG(MlpPolicy, env, verbose=1, batch_size=64, action_noise=action_noise)
    
    training_timesteps = ARGS.training_steps
    
    for i in range(ARGS.training_loops):
    
        model.learn(total_timesteps=training_timesteps)
        model.save("ddpg"+str((i+1)*training_timesteps))
        model.save_replay_buffer(ARGS.run_name+str((i+1)*training_timesteps))

        #### Show (and record a video of) the model's performance ##########################################
        env_test = RLCrazyFlieAviary(drone_model=ARGS.drone, Physics=ARGS.physics, freq=ARGS.simulation_freq_hz,
            gui=ARGS.gui, record=ARGS.record_video, PID_Control=ARGS.PID_Control, run_name=ARGS.run_name)
        obs = env_test.reset()
        start = time.time()
        for i in range(ARGS.test_duration*env_test.SIM_FREQ):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_test.step(action)
            print(i)
            print(obs)
            print(done)
            env_test.render()
            if done: break
        env_test.close()

    env.close()