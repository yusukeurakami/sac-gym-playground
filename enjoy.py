import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
import doorenv2

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--pos_control', action="store_true",
                    help='use position control of joints (default: False)')
parser.add_argument('--ik_control', action="store_true",
                    help='use inverse kinematic position control (default: False)')
parser.add_argument('--load_name', type=str,
                    help='policy to inference')
parser.add_argument('--render', action="store_false",
                    help='render (default: True)')
args = parser.parse_args()



# Environment
if args.pos_control:
    print("Using: position control")
else:
    print("Using: torque control")
############
actuator = 'motor'
if args.pos_control: actuator='position'

env_kwargs = dict(port = 1050,
                visionnet_input = False,
                unity = False,
                world_path = '/home/demo/DoorGym/world_generator/world/pull_blue_right_v2_gripper_{}_lefthinge_single/'.format(actuator),
                pos_control = args.pos_control)
env = gym.make(args.env_name, **env_kwargs)
print(env.xml_path)
env._max_episode_steps = 512
############
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Action space trick for the IK control
if not args.ik_control:
    env_action_space = env.action_space
    action_size = env.action_space.shape[0]
else:
    print("ik action space")
    low = np.zeros(7)
    env_action_space = gym.spaces.Box(low=low, high=low, dtype=np.float32)
    action_size = env_action_space.shape[0]

# Agent
agent = SAC(env.observation_space.shape[0], env_action_space, args)
## Load
actor_path = args.load_name
critic_path = args.load_name.replace('actor','critic')
agent.load_model(actor_path, critic_path)


# Evaluate
print('Evaluation')
avg_reward = 0.
episodes = 20
succeeded = 0
for _  in range(episodes):
    state = env.reset()
    ##########
    current_pos = state[:env.action_space.shape[0]]
    ##########
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state, evaluate=True)

        ##############
        if args.pos_control:
            next_a = action
            next_a += current_pos
            next_state, reward, done, _ = env.step(next_a) # Step
            current_pos = next_state[:env.action_space.shape[0]]
        else:
            next_state, reward, done, _ = env.step(action) # Step
        ##############

        if args.render:
            env.env.render()

        episode_reward += reward
        state = next_state

    avg_reward += episode_reward
    if abs(env.env.get_doorangle())>=0.2:
        succeeded += 1
avg_reward /= episodes
success_rate = succeeded/episodes

print("----------------------------------------")
print("Test Episodes: {}, Avg. Reward: {}, Success rate {}% per {} trials".format(episodes, round(avg_reward, 2), round(success_rate, 2), episodes))
print("----------------------------------------")

# env.close()

