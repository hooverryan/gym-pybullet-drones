import os
import time
import datetime
import sys
import argparse
import pdb
import math
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p

from torch import nn

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.RLCrazyFlieAviary import RLCrazyFlieAviary

from gym_pybullet_drones.utils.utils import *
from gym_pybullet_drones.utils.Logger import Logger

from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor

from TD_multiagent import TD3_MultiAgent

def reward_calculation(old_state=None,new_state=None,action=None,prev_action=None,env=None):
    if old_state is None:
        return 3

    old_state = old_state.flatten()
    pos_old = old_state[0:3]
    v_old = old_state[6:9]
    angle_old = old_state[3:6]
    omega_old = old_state[9:12]
    
    new_state = new_state.flatten()
    pos_new = new_state[0:3]
    v_new = new_state[6:9]
    angle_new = new_state[3:6]
    omega_new = new_state[9:12]
    
    target_pos = env.target_pos
    
    constant_reward = 0
    
    rewards = []
    
    # position based reward
    pos_error_new = np.sqrt(np.sum(np.square(pos_new-target_pos)))
    yaw_error = np.abs(angle_new[2])
    rewards.append(env.constantReward - pos_error_new-yaw_error)
    
    # stable flight reward
    omega_error_new = np.sqrt(np.sum(np.square(omega_new)))
    v_error_new = np.sqrt(np.sum(np.square(v_new)))
    rewards.append(env.constantReward-omega_error_new-v_error_new)
    
    # control reward
    control_penalty = -(np.max(action)-np.min(action))
    control_penalty -= np.sqrt(np.sum(np.square(action-prev_action)))
    control_penalty -= np.sqrt(np.sum(np.square(2*env.HOVER_RPM/env.MAX_RPM-1-action)))
    rewards.append(env.constantReward+control_penalty)
    
    return np.array(rewards)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


def test(agent, env, run_name, ARGS, log_dir, agent_number):

    num_episodes = ARGS.validate_episodes
    max_episode_length = ARGS.test_episode_length
    record = ARGS.record_video
    
    test_env = RLCrazyFlieAviary(drone_model=env.DRONE_MODEL, physics=env.PHYSICS, freq=env.SIM_FREQ,
        gui=False, record=record, PID_Control=env.usePID, maxWindSpeed=env.MAXWINDSPEED, run_name=run_name)
        
    total_reward = 0.
    
    for episode in range(num_episodes):
            
        episode_reward = 0.
        episode_steps = 0
        state = test_env.reset()
        
        test_env.IMG_PATH = os.path.join(os.getcwd(), log_dir,'frames',"videos-"+run_name+"-%s" % (episode))+'\\'
        os.makedirs(os.path.dirname(test_env.IMG_PATH), exist_ok=True)
        logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),log_name=run_name+'-'+str(episode+1))
        
        done = False
        while not done:
            action = agent.predict(state,deterministic=True)[0]
            # env response with next_observation, reward, terminate_info
            next_state, reward, done, _ = test_env.step(action)
            if agent_number!=0:
                reward = reward_calculation(next_state)[agent_number-1]

            if episode_steps >= max_episode_length -1:
                done = True

            # log and update
            logger.log(drone=0, timestamp=episode_steps/env.SIM_FREQ, state=np.hstack([next_state[0:3],np.zeros(4),next_state[3:12],test_env._preprocessAction(action)]), control=np.hstack([test_env.target_pos, np.zeros(9)]))
            episode_steps += 1
            episode_reward += reward
            state = next_state

            if done: # end of episode
                prCyan('#{}:\tepisode_reward:{}\tepisode_steps:{}'.format(episode,episode_reward,episode_steps))

                # reset
                total_reward += episode_reward
                episode_reward = 0.
                
        if record: os.system('ffmpeg -r 24 -f image2 -s 640x480 -i '+test_env.IMG_PATH+'frame_%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p '+log_dir+'\\videos\\'+run_name+'-'+str(episode+1)+'.mp4')
        try:
            logger.save(log_dir+"/flightlogs/")
            logger.plot(path=log_dir+"/flightlogs/")
        except:
            prRed('Error in the flight log')
    
    test_env.close()

    mean_rewards = total_reward/num_episodes
    prYellow('[Evaluation] mean_reward:{}'.format(mean_rewards))
    return mean_rewards
    
def main(ARGS):

    #### Define and parse (optional) arguments for the script ##########################################
    parser = argparse.ArgumentParser(description='RL flight script using RLCrazyFlieAviary')
    parser.add_argument('--drone',              default=DroneModel.CF2X, type=lambda model: DroneModel[model], help='Drone model (default: CF2X)')
    parser.add_argument('--physics',            default=Physics.PYB_DRAG,type=lambda phy: Physics[phy],        help='Physics updates (default: PYB_DRAG)')
    parser.add_argument('--gui',                dest='gui',              action='store_true',                  help='Enable PyBullet GUI (default: False)')
    parser.add_argument('--record_video',       dest='record_video',     action='store_true',                  help='Enable video recording (default: True)')
    parser.add_argument('--no_record',          dest='record_video',     action='store_false',                 help='Disable video recording')
    parser.add_argument('--simulation_freq_hz', default=200,             type=int,                             help='Simulation frequency in Hz (default: 200)')
    parser.add_argument('--run_name',           default='untitled',      type=str,                             help='Name for video files, frames, log files, etc.')
    parser.add_argument('--max_wind_speed',     default=0,               type=float,                           help='Maximum wind speed (requires Physics.PYB_WIND model) (default: 0)')
    parser.add_argument('--test_episode_length',default=6000,            type=int,                             help='Maximum test episode length (default: 6000)')
    parser.add_argument('--validate_episodes',  default=5,               type=int,                             help='Number of test simulations (default: 5)')
    parser.add_argument('--step_iters',         default=10,              type=int,                             help='Number of intermediate model saves during training (default: 5)')
    parser.add_argument('--training_timesteps', default=100000,          type=int,                             help='How many timesteps to perform during training (default: 500000)')
    parser.add_argument('--max_episode_length', default=2000,            type=int,                             help='Maximum training episode length (default: 2000)')
    parser.set_defaults(gui=False,record_video=True)
    ARGS = parser.parse_args()

    #### Set up the log directories
    log_dir = os.path.join("logs", ARGS.run_name + "-" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(log_dir)
    os.makedirs(os.path.join(log_dir,'videos'))
    os.makedirs(os.path.join(log_dir,'flightlogs'))
    os.makedirs(os.path.join(log_dir,'models'))
    os.makedirs(os.path.join(log_dir,'experiences'))
    
    #### Set up Environments, and action and observation spaces
    orig_env = RLCrazyFlieAviary(drone_model=ARGS.drone, physics=ARGS.physics, freq=ARGS.simulation_freq_hz,
        gui=ARGS.gui, record=False, maxWindSpeed=ARGS.max_wind_speed)
        
    env = Monitor(orig_env,log_dir)
    prBlack("[INFO] Action space:{}".format(env.action_space))
    prBlack("[INFO] Observation space:{}".format(env.observation_space))
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    
    #### Set up Agent
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[512,512,512])
    agent = TD3_MultiAgent(TD3Policy, env, verbose = 1, reward_function=reward_calculation,policy_kwargs=policy_kwargs)
    
    #### Training Loop
    for i in range(ARGS.step_iters):     # run for step_iters * training_timesteps

        agent.learn(total_timesteps=ARGS.training_timesteps)

        for j in range(len(agent.agents)):
            agent.agents[j].save(log_dir+"/models/"+ARGS.run_name+"_agent_"+str(j)+"_"+str((i+1)*ARGS.training_timesteps))
            agent.agents[j].save_replay_buffer(log_dir+"/experiences/"+ARGS.run_name+"_experience_agent_"+str(j)+"_"+str((i+1)*ARGS.training_timesteps))

        #### Show (and record a video of) the model's performance ##########################################
            test(agent.agents[j], orig_env, ARGS.run_name+"_agent_"+str(j)+"_"+str((i+1)*ARGS.training_timesteps), ARGS, log_dir, j)

    env.close()

if __name__ == "__main__":
    main(sys.argv)