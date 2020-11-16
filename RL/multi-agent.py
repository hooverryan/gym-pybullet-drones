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

def reward_calculation(state,control,env):
    penaltyVelocityWeight = 5
    penaltyAngularVelocityWeight = 5
    penaltyFlagWeight = 1000

    positionThreshold = 0.1
    angleThreshold = np.pi/18
    angularVelocityThreshold = 0.05
    velocityThreshold = 0.1

    rewardGoal = 20

    pos = state[0:3]
    v = state[6:9]
    angle = state[3:6]
    omega = state[9:12]
    
    target_position = env.target_pos
    
    constant_reward = 0
    
    rewards = []
    
    # position based reward
    rewards.append(np.sum(np.square(pos-target_position))+np.square(angle[2])+constant_reward)
    
    # stable flight reward
    rewards.append(np.sum(np.square(omega)))
    
    # total reward function
    errorPosition = np.sum(np.square(pos-target_position))
    errorVelocity = np.sum(np.square(v))
    errorAngularVelocity = np.sum(np.square(omega))
    
    penaltyPosition = errorPosition
    penaltyAngle = np.square(angle[2])
    penaltyVelocity = errorVelocity*penaltyVelocityWeight
    penaltyAngularVelocity = errorAngularVelocity*penaltyAngularVelocityWeight
    
    outOfGeoFence = any(np.abs(pos) > env.geoFenceMax)
    crashed = True if pos[2]<env.COLLISION_H else False
    penaltyFlag = penaltyFlagWeight if outOfGeoFence or crashed else 0
    
    rewardPosition = np.sqrt(errorPosition)>positionThreshold
    rewardYaw = angle[2]>angleThreshold
    rewardVelocity = np.sqrt(errorVelocity)>velocityThreshold
    rewardAngularVelocity = np.sqrt(errorAngularVelocity)>angularVelocityThreshold
    
    rewards.append(rewardGoal-penaltyPosition-penaltyAngle-penaltyVelocity-penaltyAngularVelocity-penaltyFlag)

    return rewards

def train(agents, env, ARGS):

    max_episodes = ARGS.max_episodes
    max_episode_length = ARGS.max_episode_length
    run_name = ARGS.run_name
    num_model_saves = ARGS.num_model_saves
    
    for a in agents:
        a.is_training = True

    step = episode = episode_steps = 0
    episode_reward = 0.
    state = None
    while episode < max_episodes:
        # reset if it is the start of episode
        if state is None:
            state = env.reset()
            for a in agents:
                a.reset()
            selected_agent=np.random.choice(len(agents))
            
        action = agents[selected_agent].select_action(state) #based on selected agent
        # env response with next_observation, reward, terminate_info
        next_state, _, done, info = env.step(action)
        
        if episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        reward = reward_calculation(next_state,action,env)
        for (r,a) in zip(reward,agents):
            a.observe(state,action,r,done)

        # add the state, action, reward to the memory for each agent
        if step > agents[0].batch_size:
            for a in agents:
                a.update_policy() #update each agent
        
        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward[selected_agent]
        state = next_state

        if done: # end of episode
            prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))
            
            # save intermediate model
            if (episode+1) % int(max_episodes/num_model_saves) == 0:
                
                for i in range(len(agents)):
                    agents[i].save_model(run_name+'_agent'+str(i)+'_'+str(episode+1))
                test(agents[i],run_name+'_agent'+str(i)+'_'+str(episode+1)+'_training',ARGS)

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
    #parser.add_argument('--load_model',         default=None,            type=str,                             help='Model name to load')
    #parser.add_argument('--load_experience',    default=None,            type=str,                             help='Experience to load')
    ARGS = parser.parse_args()

    env = RLCrazyFlieAviary(drone_model=ARGS.drone, physics=ARGS.physics, freq=ARGS.simulation_freq_hz,
        gui=ARGS.gui, record=False, PID_Control=ARGS.PID_Control, maxWindSpeed=ARGS.max_wind_speed)
    prBlack("[INFO] Action space:{}".format(env.action_space))
    prBlack("[INFO] Observation space:{}".format(env.observation_space))

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    
    agents = [DDPG(nb_states, nb_actions) for _ in range(3)]
    
    #if ARGS.load_model is not None:agent.load_weights(ARGS.load_model)
    
    #if ARGS.load_experience is not None: agent.load_experience(ARGS.load_experience)

    if ARGS.train: train(agents, env, ARGS)

    if ARGS.test: test(agents[0], ARGS.run_name, ARGS)
    
    env.close()