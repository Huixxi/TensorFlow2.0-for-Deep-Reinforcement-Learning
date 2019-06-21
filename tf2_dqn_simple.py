"""
A simple version Deep Q-Network(DQN) including the main tactics mentioned in DeepMind's original paper:
- Experience Replay
- Target Network
To play CartPole-v0.

> Note: DQN can only handle discrete-env which have a discrete action space, like up, down, left, right.
        As for the CartPole-v0 environment, its state(the agent's observation) is a 1-D vector not a 3-D image like
        Atari, so in that simple example, there is no need to use the convolutional layer, just fully-connected layer.

Using:
TensorFlow 2.0
Numpy 1.16.2
Gym 0.12.1
"""

import tensorflow as tf
import gym
import time
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko

# Neural Network Model Defined at Here.
class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__(name='dqn_simple')
        # you can try different kernel initializer
        self.fc1 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.fc2 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.logits = kl.Dense(num_actions, name='q_values')

    # forward propagation
    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.logits(x)
        return x

    # pi* = argmax_a' Q(s, a')
    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action[0], q_values[0]

# To test whether the model works
def test_model():
    env = gym.make('CartPole-v0')
    print('num_actions: ', env.action_space.n)
    model = Model(env.action_space.n)

    obs = env.reset()
    print('obs_shape: ', obs.shape)

    # tensorflow 2.0: no feed_dict or tf.Session() needed at all
    best_action, q_values = model.action_value(obs[None])
    print('res of test model: ', best_action, q_values)  # 0 [ 0.00896799 -0.02111824]


class DQNAgent:
    def __init__(self, model, env, buffer_size=100, learning_rate=.00001, epsilon=.1, epsilon_dacay=0.995, min_epsilon=.01,
                 gamma=.95, batch_size=4, target_update_iter=400, train_nums=10000, start_learning=10):
        self.model = model
        self.target_model = model
        # gradient clip
        opt = ko.Adam(clipvalue=10.0)
        self.model.compile(optimizer=opt, loss='mse')

        # parameters
        self.env = env                              # gym environment
        self.lr = learning_rate                     # learning step
        self.epsilon = epsilon                      # e-greedy when exploring
        self.epsilon_decay = epsilon_dacay          # epsilon decay rate
        self.min_epsilon = min_epsilon              # minimum epsilon
        self.gamma = gamma                          # discount rate
        self.batch_size = batch_size                # batch_size
        self.target_update_iter = target_update_iter    # target network update period
        self.train_nums = train_nums                # total training steps
        self.num_in_buffer = 0                      # transition's num in buffer
        self.buffer_size = buffer_size              # replay buffer size
        self.start_learning = start_learning        # step to begin learning(no update before that step)

        # replay buffer params [(s, a, r, ns, done), ...]
        self.obs = np.empty((self.buffer_size,) + self.env.reset().shape)
        self.actions = np.empty((self.buffer_size), dtype=np.int8)
        self.rewards = np.empty((self.buffer_size), dtype=np.float32)
        self.dones = np.empty((self.buffer_size), dtype=np.bool)
        self.next_states = np.empty((self.buffer_size,) + self.env.reset().shape)
        self.next_idx = 0

    def train(self):
        # initialize the initial observation of the agent
        obs = self.env.reset()
        for t in range(1, self.train_nums):
            best_action, q_values = self.model.action_value(obs[None])  # input the obs to the network model
            action = self.get_action(best_action)   # get the real action
            next_obs, reward, done, info = self.env.step(action)    # take the action in the env to return s', r, done
            self.store_transition(obs, action, reward, next_obs, done)  # store that transition into replay butter
            self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)

            if t > self.start_learning:  # start learning
                losses = self.train_step()
                if t % 1000 == 0:
                    print('losses each 1000 steps: ', losses)

            if t % self.target_update_iter == 0:
                self.update_target_model()
            if done:
                obs = env.reset()
            else:
                obs = next_obs

    def train_step(self):
        idxes = self.sample(self.batch_size)
        s_batch = self.obs[idxes]
        a_batch = self.actions[idxes]
        r_batch = self.rewards[idxes]
        ns_batch = self.next_states[idxes]
        done_batch = self.dones[idxes]

        target_q = r_batch + self.gamma * np.amax(self.get_target_value(ns_batch), axis=1) * (1 - done_batch)
        target_f = self.model.predict(s_batch)
        for i, val in enumerate(a_batch):
            target_f[i][val] = target_q[i]

        losses = self.model.train_on_batch(s_batch, target_f)

        return losses

    def evalation(self, render=True):
        obs, done, ep_reward = self.env.reset(), False, 0
        # one episode until done
        while not done:
            action, q_values = self.model.action_value(obs[None])  # Using [None] to extend its dimension (4,) -> (1, 4)
            obs, reward, done, info = self.env.step(action)
            ep_reward += reward
            if render:  # visually show
                self.env.render()
            time.sleep(0.1)
        self.env.close()
        return ep_reward

    # store transitions into replay butter
    def store_transition(self, obs, action, reward, next_state, done):
        n_idx = self.next_idx % self.buffer_size
        self.obs[n_idx] = obs
        self.actions[n_idx] = action
        self.rewards[n_idx] = reward
        self.next_states[n_idx] = next_state
        self.dones[n_idx] = done
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    # sample n different indexes
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

    # e-greedy
    def get_action(self, best_action):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return best_action

    # assign the current network parameters to target network
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_target_value(self, obs):
        return self.target_model.predict(obs)

    def e_decay(self):
        self.epsilon *= self.epsilon_decay

if __name__ == '__main__':
    test_model()

    env = gym.make("CartPole-v0")
    num_actions = env.action_space.n
    model = Model(num_actions)
    agent = DQNAgent(model, env)
    # test before
    rewards_sum = agent.evalation()
    print("Before Training: %d out of 200" % rewards_sum) # 9 out of 200

    agent.train()
    # test after
    rewards_sum = agent.evalation()
    print("After Training: %d out of 200" % rewards_sum) # 200 out of 200
