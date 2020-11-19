import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import os

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.RLCrazyFlieAviary import RLCrazyFlieAviary
from gym_pybullet_drones.utils.utils import *

from ddpg import DDPG

def train(agent, env, ARGS):

    max_episodes = ARGS.max_episodes
    max_episode_length = ARGS.max_episode_length
    run_name = ARGS.run_name
    num_model_saves = ARGS.num_model_saves
    
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    state = None
    while episode < max_episodes:
        # reset if it is the start of episode
        if state is None:
            state = env.reset()
            agent.reset()
            
        action = agent.select_action(state)
        # env response with next_observation, reward, terminate_info
        next_state, reward, done, info = env.step(action)

        if episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(state, action, reward, done)
        if step > agent.batch_size: agent.update_policy()
        
        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        state = next_state

        if done: # end of episode
            prGreen('#{}: episode_reward:{} episode_steps:{}'.format(episode,episode_reward,episode_steps))
            
            # save intermediate model
            if (episode+1) % int(max_episodes/num_model_saves) == 0:
                agent.save_model(run_name+str(episode+1))
                test(agent,run_name+str(episode+1)+'_training',ARGS)

            # reset
            state = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

def test(agent, run_name, ARGS):

    num_episodes = ARGS.validate_episodes
    max_episode_length = ARGS.test_episode_length
    record = ARGS.record_video

    env = RLCrazyFlieAviary(drone_model=ARGS.drone, physics=ARGS.physics, freq=ARGS.simulation_freq_hz,
        gui=False, record=record, PID_Control=ARGS.PID_Control, maxWindSpeed=ARGS.max_wind_speed, run_name=run_name)

    agent.is_training = False
    total_reward = 0.
    
    for episode in range(num_episodes):

        episode_reward = 0.
        episode_steps = 0
        state = env.reset()
        agent.reset()
        
        done = False
        while not done:
            action = agent.select_action(state)
            # env response with next_observation, reward, terminate_info
            next_state, reward, done, info = env.step(action)

            if episode_steps >= max_episode_length -1:
                done = True

            # update 
            episode_steps += 1
            episode_reward += reward
            state = next_state

            if done: # end of episode
                prCyan('#{}: episode_steps:{}\tepisode_reward:{}'.format(episode,episode_steps,episode_reward))

                # reset
                total_reward +=episode_reward
                episode_reward = 0.
                if record:
                    os.system('ffmpeg -r 24 -f image2 -s 640x480 -i '+env.IMG_PATH+'frame_%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p '+run_name+'_run'+str(episode)+'.mp4')
    
    prYellow('[Evaluation] mean_reward:{}'.format(total_reward/num_episodes))

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DDPG flight script using RLCrazyFlieAviary')
    parser.add_argument('--drone',              default=DroneModel.CF2X, type=lambda model: DroneModel[model], help='Drone model (default: CF2X)')
    parser.add_argument('--physics',            default=Physics.PYB_DRAG,type=lambda phy: Physics[phy],        help='Physics updates (default: PYB_DRAG)')
    parser.add_argument('--PID_Control',        default=False,           type=str2bool,                        help='Whether to use a PID Control (default: False)')
    parser.add_argument('--gui',                default=True,            type=str2bool,                        help='Whether to use PyBullet GUI (default: True)')
    parser.add_argument('--record_video',       default=False,           type=str2bool,                        help='Whether to record a video (default: False)')
    parser.add_argument('--simulation_freq_hz', default=200,             type=int,                             help='Simulation frequency in Hz (default: 200)')
    parser.add_argument('--test_episode_length',default=3000,            type=int,                             help='Maximum test episode length (default: 3000)')
    parser.add_argument('--validate_episodes',  default=10,              type=int,                             help='Number of test simulations (default: 10)')
    parser.add_argument('--run_name',           default='',              type=str,                             help='Name for video files and frames (default: None)')
    parser.add_argument('--max_wind_speed',     default=0,               type=float,                           help='Maximum wind speed (requires Physics.PYB_WIND model) (default: 0)')
    parser.add_argument('--train',              default=True,            type=str2bool,                        help='Whether to do training (default: True)')
    parser.add_argument('--num_model_saves',    default=5,               type=int,                             help='Number of intermediate model saves during training (default: 5)')
    parser.add_argument('--test',               default=True,            type=str2bool,                        help='Whether to do testing (default: True)')
    parser.add_argument('--max_episodes',       default=10000,           type=int,                             help='How many episodes to perform during training (default: 10000)')
    parser.add_argument('--max_episode_length', default=3000,            type=int,                             help='Maximum training episode length (default: 3000)')
    parser.add_argument('--load_model',         default=None,            type=str,                             help='Model name to load')
    parser.add_argument('--load_experience',    default=None,            type=str,                             help='Experience to load')
    ARGS = parser.parse_args()

    env = RLCrazyFlieAviary(drone_model=ARGS.drone, physics=ARGS.physics, freq=ARGS.simulation_freq_hz,
        gui=ARGS.gui, record=False, PID_Control=ARGS.PID_Control, maxWindSpeed=ARGS.max_wind_speed)
    prBlack("[INFO] Action space:{}".format(env.action_space))
    prBlack("[INFO] Observation space:{}".format(env.observation_space))

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    agent = DDPG(nb_states, nb_actions)
    
    if ARGS.load_model is not None:agent.load_weights(ARGS.load_model)
    
    if ARGS.load_experience is not None: agent.load_experience(ARGS.load_experience)

    if ARGS.train: train(agent, env, ARGS)

    if ARGS.test: test(agent, ARGS.run_name, ARGS)
    
    env.close()