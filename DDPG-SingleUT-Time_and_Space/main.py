import sys
import gym
import gym_foo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPG.ddpg import DDPGagent
from DDPG.utils import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
env = gym.make('foo-v0')
save_model = True
Train = False # True for tranining, and False for testing.

outdir = "checkpoints"
if save_model and not os.path.exists("{}/models".format(outdir)):
    os.makedirs("{}/models".format(outdir))

agent = DDPGagent(env, hidden_size=[64, 64], actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=0.99, tau=5e-3, max_memory_size = pow(2, 28))
noise = OUNoise(env.action_space)
batch_size = pow(2, 9)
rewards = []
avg_rewards = []
start_steps = 3e5

if Train:
    for episode in range(14000):
        state = env.reset()
        episode_reward = 0
        
        for step in range(81):
            if step+(81*episode) < start_steps:
                action = np.random.random(env.action_space.shape[0])
            else:
                action = np.zeros(env.action_space.shape[0]) + agent.get_action(state)
                action = noise.get_action(action)

            new_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, new_state, done)
            
            if (len(agent.memory) > batch_size) and (step+(81*episode))>start_steps:
                agent.update(batch_size)        
            
            state = new_state
            episode_reward += reward

            if done:
                sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=6), np.mean(rewards)))
                break

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards))

    agent.save('{}/models/model'.format(outdir))
else:
    agent.load('{}/models/model'.format(outdir))
    episode_reward = 0
    state = env.reset()
    for step in range(81):
        action = np.zeros(env.action_space.shape[0]) + agent.get_action(state)
        new_state, reward, done, _ = env.step(action)
        state = new_state
        episode_reward += reward
        rewards.append(reward)
        if done:
            sys.stdout.write("reward: {}\n".format(np.round(episode_reward, decimals=6)))
            break
    np.savetxt("Test_Rewards_Records.csv", rewards, delimiter=',')
    np.savetxt("Total_Reward_Test.txt", [episode_reward])