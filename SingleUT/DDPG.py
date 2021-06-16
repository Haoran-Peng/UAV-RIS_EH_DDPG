
"""
Many thanks for the contribution of Mofan Zhou's Reinforcement Learning tutorial
The DDPG part of this project base on the github Repositories of Dr. Zhou.
The original information of Dr. Zhou's tutorials as follows.
"""


"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time
import ARIS_ENV as arenv
import globe
import matplotlib.pyplot as plt
import sys
import csv
#####################  hyper parameters  ####################

MAX_EPISODES = 2000
MAX_EP_STEPS = 600
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.99     # reward discount #Reward discount should be smaller since our situation is different from the game.
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 100000
BATCH_SIZE = 128


##########################################################################
globe._init()

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.epsilon = 1
        self.epsilon_decay = 1/2000

        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):

            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):

        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]


    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################
def ou_noise(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.rand(2)

def plot(idx, frame_idx, rewards):
    plt.figure()
    plt.title('Dataset %i, frame %s. reward: %s' % (int(idx), frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    plt.savefig('DDPG_Result/Dataset_'+str(idx)+'.png', format='png')
    plt.close()

observation_space = arenv._observation_space()
action_space = arenv._action_space()
s_dim = observation_space.shape[0]
a_dim = action_space.shape[0]
a_bound = [0.5, 0.5] 


ddpg = DDPG(a_dim, s_dim, a_bound)


total_record = []

def train(filename_idx):
    arenv._init(globe, True, MAX_EP_STEPS, "Dataset/Distance_"+str(filename_idx)+".csv")
    action_epsilon = 1
    action_epsilon_decay = 1/2000
    rewards = []
    t1 = time.time()
    var = 1  # control exploration

    for i in range(MAX_EPISODES):
        s = arenv.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):

            # Add exploration noise
            a = ddpg.choose_action(s)
            
            # noise
            action_epsilon -= action_epsilon_decay
            noise = max(action_epsilon, 0) * ou_noise(a, 0, 0.6, 0.3)

            a = np.clip(np.random.normal(a+noise, var, size=2), 0.1, 0.99)

            s_, r, done = arenv.Step(a)

            ddpg.store_transition(s, a, r/10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %.6f' % ep_reward, 'Explore: %.6f' % var, )
                # if ep_reward > -300:RENDER = True
                break

        rewards.append(ep_reward)
    np.savetxt("DDPG_Result/Dataset_"+str(filename_idx)+"_Rewards.csv", rewards, delimiter=',')
    plot(filename_idx, i, rewards)
    total_record.append(rewards[-1])

    with open(r'DDPG_Result/total_rewards.csv',mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        wf.writerow(total_record)

    print('Running time: ', time.time() - t1)

train(sys.argv[1])