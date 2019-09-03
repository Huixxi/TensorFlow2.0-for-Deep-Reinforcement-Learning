"""
Deep Q-Network(DQN) for Atari Game, which has convolutional layers to handle images input and other preprocessings.

Using:
TensorFlow 2.0
Numpy 1.16.2
Gym 0.12.1
"""

import tensorflow as tf
print(tf.__version__)

import gym
import time
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko

np.random.seed(1)
tf.random.set_seed(1)

# Minor change from cs234:reinforcement learning, assignment 2 -> utils/preprocess.py
def greyscale(state):
    """
    Preprocess state (210, 160, 3) image into
    a (80, 80, 1) image in grey scale
    """
    state = np.reshape(state, [210, 160, 3]).astype(np.float32)
    # grey scale
    state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114
    # karpathy
    state = state[35:195]  # crop
    state = state[::2,::2]  # downsample by factor of 2
    state = state[:, :, np.newaxis]
    return state.astype(np.float32)


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__(name='dqn')
        self.conv1 = kl.Conv2D(32, kernel_size=(8, 8), strides=4, activation='relu')
        self.conv2 = kl.Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu')
        self.conv3 = kl.Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu')
        self.flat = kl.Flatten()
        self.fc1 = kl.Dense(512, activation='relu')
        self.fc2 = kl.Dense(num_actions)

    def call(self, inputs):
        # x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action[0], q_values[0]


class DQNAgent:
    def __init__(self, model, target_model, env, buffer_size=1000, learning_rate=.001, epsilon=.1, gamma=.9,
                 batch_size=4, target_update_iter=20, train_nums=100, start_learning=10):
        self.model = model
        self.target_model = target_model
        self.model.compile(optimizer=ko.Adam(), loss='mse')

        # parameters
        self.env = env  # gym environment
        self.lr = learning_rate  # learning step
        self.epsilon = epsilon  # e-greedy when exploring
        self.gamma = gamma  # discount rate
        self.batch_size = batch_size  # batch_size
        self.target_update_iter = target_update_iter  # target update period
        self.train_nums = train_nums  # total training steps
        self.num_in_buffer = 0  # transitions num in buffer
        self.buffer_size = buffer_size  # replay buffer size
        self.start_learning = start_learning  # step to begin learning(save transitions before that step)

        # replay buffer
        self.obs = np.empty((self.buffer_size,) + greyscale(self.env.reset()).shape)
        self.actions = np.empty((self.buffer_size), dtype=np.int8)
        self.rewards = np.empty((self.buffer_size), dtype=np.float32)
        self.dones = np.empty((self.buffer_size), dtype=np.bool)
        self.next_states = np.empty((self.buffer_size,) + greyscale(self.env.reset()).shape)
        self.next_idx = 0


    # To test whether the model works
    def test(self, render=True):
        obs, done, ep_reward = self.env.reset(), False, 0
        while not done:
            obs = greyscale(obs)
            # Using [None] to extend its dimension [80, 80, 1] -> [1, 80, 80, 1]
            action, _ = self.model.action_value(obs[None])
            obs, reward, done, info = self.env.step(action)
            ep_reward += reward
            if render:  # visually
                self.env.render()
            time.sleep(0.05)
        self.env.close()
        return ep_reward

    def train(self):
        obs = self.env.reset()
        obs = greyscale(obs)[None]
        for t in range(self.train_nums):
            best_action, q_values = self.model.action_value(obs)
            action = self.get_action(best_action)
            next_obs, reward, done, info = self.env.step(action)
            next_obs = greyscale(next_obs)[None]
            self.store_transition(obs, action, reward, next_obs, done)
            self.num_in_buffer += 1

            if t > self.start_learning:  # start learning
                losses = self.train_step(t)

            if t % self.target_update_iter == 0:
                self.update_target_model()

            obs = next_obs

    def train_step(self, t):
        idxes = self.sample(self.batch_size)
        self.s_batch = self.obs[idxes]
        self.a_batch = self.actions[idxes]
        self.r_batch = self.rewards[idxes]
        self.ns_batch = self.next_states[idxes]
        self.done_batch = self.dones[idxes]

        target_q = self.r_batch + self.gamma * \
                   np.amax(self.get_target_value(self.ns_batch), axis=1) * (1 - self.done_batch)
        target_f = self.model.predict(self.s_batch)
        for i, val in enumerate(self.a_batch):
            target_f[i][val] = target_q[i]

        losses = self.model.train_on_batch(self.s_batch, target_f)

        return losses



    # def loss_function(self, q, target_q):
    #     n_actions = self.env.action_space.n
    #     print('action in loss', self.a_batch)
    #     actions = to_categorical(self.a_batch, n_actions)
    #     q = np.sum(np.multiply(q, actions), axis=1)
    #     self.loss = kls.mean_squared_error(q, target_q)


    def store_transition(self, obs, action, reward, next_state, done):
        n_idx = self.next_idx % self.buffer_size
        self.obs[n_idx] = obs
        self.actions[n_idx] = action
        self.rewards[n_idx] = reward
        self.next_states[n_idx] = next_state
        self.dones[n_idx] = done
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    def sample(self, n):
        assert n < self.num_in_buffer
        res = []
        while True:
            num = np.random.randint(0, self.num_in_buffer)
            if num not in res:
                res.append(num)
            if len(res) == n:
                break
        return res

    def get_action(self, best_action):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return best_action

    def update_target_model(self):
        print('update_target_mdoel')
        self.target_model.set_weights(self.model.get_weights())

    def get_target_value(self, obs):
        return self.target_model.predict(obs)

if __name__ == '__main__':
    env = gym.make("Pong-v0")
    obs = env.reset()
    num_actions = env.action_space.n
    model = Model(num_actions)
    target_model = Model(num_actions)
    agent = DQNAgent(model, target_model, env)
    # reward = agent.test()
    agent.train()
