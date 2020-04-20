import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory

from gym.envs.robotics.rotations import quat2euler, euler2quat, mat2euler
import doorenv2
import zmq

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
parser.add_argument('--id', type=str, default="test",
                    help='name for the experiments')
parser.add_argument('--pos_control', action="store_true",
                    help='use position control of joints (default: False)')
parser.add_argument('--ik_control', action="store_true",
                    help='use inverse kinematic position control (default: False)')
parser.add_argument('--action_repeat', type=int, default=1, metavar='N',
                    help='number of times to give a same action command (default: 1)')
args = parser.parse_args()

class Client():
    def __init__(self):
        port = "5556"
        context = zmq.Context()
        print ("Connecting to the policy server...")
        self.socket = context.socket(zmq.REQ)
        self.socket.connect ("tcp://localhost:%s" % port)

    def IK(self, ee_pos):
        # print(ee_pos)
        ee_pos = [float(x) for x in list(ee_pos)]
        self.socket.send_json(ee_pos)
        message = self.socket.recv_json()
        joint_cmd = message
        return joint_cmd

# Environment
# env = NormalizedActions(gym.make(args.env_name))
# env = gym.make(args.env_name)
if args.pos_control:
    print("Using: position control")
if args.ik_control:
    print("Using: IK control")
    c = Client()
else:
    print("Using: torque control")
############
actuator = 'motor'
if args.pos_control: actuator='position'

env_kwargs = dict(port = 1050,
                visionnet_input = False,
                unity = False,
                world_path = '/home/demo/DoorGym/world_generator/world/pull_blue_right_v2_gripper_{}_lefthinge_single/'.format(actuator),
                pos_control = args.pos_control,
                ik_control = args.ik_control)
env = gym.make(args.env_name, **env_kwargs)
print(env.xml_path)
env._max_episode_steps = 512
############
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Action space trick for the IK control
if not args.ik_control:
    low, high = -np.ones(env.action_space.shape[0]), np.ones(env.action_space.shape[0])
    # env_action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    # action_size = env.action_space.shape[0]
    # print(env_action_space.low, env_action_space.high)
else:
    print("ik action space")
    low, high = -np.ones(7), np.ones(7)
env_action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
action_size = env_action_space.shape[0]

# action_size = env.action_space.shape[0]

# Agent
agent = SAC(env.observation_space.shape[0], env_action_space, args)

#TesnorboardX
logdir='runs/{}_SAC_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""
                                                             , args.id)
writer = SummaryWriter(logdir)

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    ##########
    current_pos = state[:action_size]
    ##########

    while not done:
        if args.start_steps > total_numsteps:
            # print(env_action_space.sample(), env_action_space.high)
            # action = env.action_space.sample()  # Sample random action
            action = env_action_space.sample()  # Sample random action
            if args.ik_control:
                action = action[:action_size]
            print("random action ", action)
        else:
            action = agent.select_action(state)  # Sample action from policy
            print("actor action ", action)
        next_a = action

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, total_numsteps)
                writer.add_scalar('loss/critic_2', critic_2_loss, total_numsteps)
                writer.add_scalar('loss/policy', policy_loss, total_numsteps)
                writer.add_scalar('loss/entropy_loss', ent_loss, total_numsteps)
                writer.add_scalar('entropy_temprature/alpha', alpha, total_numsteps)
                updates += 1


        ##############
        if args.pos_control:
            print("current pos: ",current_pos)
            next_a += current_pos
        elif args.ik_control:
            print("orig ", next_a)
            print("euler ", next_a[3:-1])
            quat = euler2quat(next_a[3:-1])
            print("quat: ", quat)
            ee_pos = np.concatenate((next_a[:3], quat))
            print("ee_pos ", ee_pos)
            joint_pos = c.IK(ee_pos)
            if joint_pos == 0:
                next_a = current_pos
            else:
                next_a = joint_pos
            print("joint_pos ", joint_pos)
        for _ in range(args.action_repeat):
            next_state, reward, done, _ = env.step(next_a) # Step
            if done:
                break
        current_pos = next_state[:action_size]
        ##############

        # next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

        # print(">>>>>>>>>>>> episode steps", episode_steps)

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('train/reward', episode_reward, total_numsteps)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 20
        succeeded = 0
        for _  in range(episodes):
            state = env.reset()
            ##########
            current_pos = state[:action_size]
            ##########
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                next_a = action

                ##############
                if args.pos_control:
                    next_a += current_pos
                for _ in range(args.action_repeat):
                    next_state, reward, done, _ = env.step(next_a) # Step
                    if done:
                        break
                current_pos = next_state[:action_size]
                ##############

                # next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
            ###############
            if abs(env.env.get_doorangle())>=0.2:
                succeeded += 1
            ###############
        avg_reward /= episodes
        writer.add_scalar('test/avg_reward', avg_reward, i_episode)

        #########
        success_rate = succeeded/episodes
        writer.add_scalar('test/success_rate', success_rate, i_episode)
        #########

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}, Success rate {}% per {} trials".format(episodes, round(avg_reward, 2), round(success_rate, 2), episodes))
        print("----------------------------------------")

    #########
    # save agent
    if i_episode % 100 == 0:
        agent.save_model(args.env_name, logdir, suffix=i_episode)
    #########

env.close()

