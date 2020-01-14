"""
A simple version of Multi-Step TD Learning Based on Dueling Double DQN with Prioritized Experience Replay.
To play CartPole-v0.

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

from collections import deque

np.random.seed(1)
tf.random.set_seed(1)

# Neural Network Model Defined at Here.
class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__(name='basic_nstepTD')
        # you can try different kernel initializer
        self.shared_fc1 = kl.Dense(16, activation='relu', kernel_initializer='he_uniform')
        self.shared_fc2 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        # there is a trick that combining the two streams' fc layer, then
        # the output of that layer is a |A| + 1 dimension tensor: |V|A1|A2| ... |An|
        # output[:, 0] is state value, output[:, 1:] is action advantage
        self.val_adv_fc = kl.Dense(num_actions + 1, activation='relu', kernel_initializer='he_uniform')

    # forward propagation
    def call(self, inputs):
        x = self.shared_fc1(inputs)
        x = self.shared_fc2(x)
        val_adv = self.val_adv_fc(x)
        # average version, you can also try the max version.
        outputs = tf.expand_dims(val_adv[:, 0], -1) + (val_adv[:, 1:] - tf.reduce_mean(val_adv[:, 1:], -1, keepdims=True))
        return outputs

    # a* = argmax_a' Q(s, a')
    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action if best_action.shape[0] > 1 else best_action[0], q_values[0]


# To test whether the model works
def test_model():
    env = gym.make('CartPole-v0')
    print('num_actions: ', env.action_space.n)
    model = Model(env.action_space.n)

    obs = env.reset()
    print('obs_shape: ', obs.shape)

    # tensorflow 2.0 eager mode: no feed_dict or tf.Session() needed at all
    best_action, q_values = model.action_value(obs[None])
    print('res of test model: ', best_action, q_values)  # 0 [ 0.00896799 -0.02111824]


# replay buffer
class SumTree:
    # little modified from https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    def __init__(self, capacity):
        self.capacity = capacity    # N, the size of replay buffer, so as to the number of sum tree's leaves
        self.tree = np.zeros(2 * capacity - 1)  # equation, to calculate the number of nodes in a sum tree
        self.transitions = np.empty(capacity, dtype=object)
        self.next_idx = 0

    @property
    def total_p(self):
        return self.tree[0]

    def add(self, priority, transition):
        idx = self.next_idx + self.capacity - 1
        self.transitions[self.next_idx] = transition
        self.update(idx, priority)
        self.next_idx = (self.next_idx + 1) % self.capacity

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)    # O(logn)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, s):
        idx = self._retrieve(0, s)   # from root
        trans_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.transitions[trans_idx]

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])


class MSTDAgent:  # Multi-Step TD Learning Based on Dueling Double DQN with Proportional Prioritization
    def __init__(self, model, target_model, env, learning_rate=.0008, epsilon=.1, epsilon_dacay=0.995, min_epsilon=.01,
                 gamma=.9, batch_size=8, target_update_iter=400, train_nums=5000, buffer_size=300, replay_period=20,
                 alpha=0.4, beta=0.4, beta_increment_per_sample=0.001, n_step=3):
        self.model = model
        self.target_model = target_model
        # gradient clip
        opt = ko.Adam(learning_rate=learning_rate, clipvalue=10.0)  # , clipvalue=10.0
        self.model.compile(optimizer=opt, loss=self._per_loss)  # loss=self._per_loss

        # parameters
        self.env = env                              # gym environment
        self.lr = learning_rate                     # learning step
        self.epsilon = epsilon                      # e-greedy when exploring
        self.epsilon_decay = epsilon_dacay          # epsilon decay rate
        self.min_epsilon = min_epsilon              # minimum epsilon
        self.gamma = gamma                          # discount rate
        self.batch_size = batch_size                # minibatch k
        self.target_update_iter = target_update_iter    # target network update period
        self.train_nums = train_nums                # total training steps

        # replay buffer params [(s, a, r, ns, done), ...]
        self.b_obs = np.empty((self.batch_size,) + self.env.reset().shape)
        self.b_actions = np.empty(self.batch_size, dtype=np.int8)
        self.b_rewards = np.empty(self.batch_size, dtype=np.float32)
        self.b_next_states = np.empty((self.batch_size,) + self.env.reset().shape)
        self.b_dones = np.empty(self.batch_size, dtype=np.bool)

        self.replay_buffer = SumTree(buffer_size)   # sum-tree data structure
        self.buffer_size = buffer_size              # replay buffer size N
        self.replay_period = replay_period          # replay period K
        self.alpha = alpha                          # priority parameter, alpha=[0, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.beta = beta                            # importance sampling parameter, beta=[0, 0.4, 0.5, 0.6, 1]
        self.beta_increment_per_sample = beta_increment_per_sample
        self.num_in_buffer = 0                      # total number of transitions stored in buffer
        self.margin = 0.01                          # pi = |td_error| + margin
        self.p1 = 1                                 # initialize priority for the first transition
        # self.is_weight = np.empty((None, 1))
        self.is_weight = np.power(self.buffer_size, -self.beta)  # because p1 == 1
        self.abs_error_upper = 1

        # multi step TD learning
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)

    def _per_loss(self, y_target, y_pred):
        return tf.reduce_mean(self.is_weight * tf.math.squared_difference(y_target, y_pred))

    def train(self):
        # initialize the initial observation of the agent
        obs = self.env.reset()
        for t in range(1, self.train_nums):
            best_action, q_values = self.model.action_value(obs[None])  # input the obs to the network model
            action = self.get_action(best_action)   # get the real action
            next_obs, reward, done, info = self.env.step(action)   # take the action in the env to return s', r, done

            # n-step replay buffer
            # minor modified from github.com/medipixel/rl_algorithms/blob/master/algorithms/common/helper_functions.py
            temp_transition = [obs, action, reward, next_obs, done]
            self.n_step_buffer.append(temp_transition)
            if len(self.n_step_buffer) == self.n_step:  # fill the n-step buffer for the first translation
                # add a multi step transition
                reward, next_obs, done = self.get_n_step_info(self.n_step_buffer, self.gamma)
                obs, action = self.n_step_buffer[0][:2]

            if t == 1:
                p = self.p1
            else:
                p = np.max(self.replay_buffer.tree[-self.replay_buffer.capacity:])
            self.store_transition(p, obs, action, reward, next_obs, done)  # store that transition into replay butter
            self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)

            if t > self.buffer_size:
                # if t % self.replay_period == 0:  # transition sampling and update
                losses = self.train_step()
                if t % 1000 == 0:
                    print('losses each 1000 steps: ', losses)

            if t % self.target_update_iter == 0:
                self.update_target_model()
            if done:
                obs = self.env.reset()   # one episode end
            else:
                obs = next_obs

    def train_step(self):
        idxes, self.is_weight = self.sum_tree_sample(self.batch_size)
        assert len(idxes) == self.b_next_states.shape[0]

        # Double Q-Learning
        best_action_idxes, _ = self.model.action_value(self.b_next_states)  # get actions through the current network
        target_q = self.get_target_value(self.b_next_states)    # get target q-value through the target network
        # get td_targets of batch states
        td_target = self.b_rewards + \
            self.gamma * target_q[np.arange(target_q.shape[0]), best_action_idxes] * (1 - self.b_dones)
        predict_q = self.model.predict(self.b_obs)
        td_predict = predict_q[np.arange(predict_q.shape[0]), self.b_actions]
        abs_td_error = np.abs(td_target - td_predict) + self.margin
        clipped_error = np.where(abs_td_error < self.abs_error_upper, abs_td_error, self.abs_error_upper)
        ps = np.power(clipped_error, self.alpha)
        # priorities update
        for idx, p in zip(idxes, ps):
            self.replay_buffer.update(idx, p)

        for i, val in enumerate(self.b_actions):
            predict_q[i][val] = td_target[i]

        target_q = predict_q  # just to change a more explicit name
        losses = self.model.train_on_batch(self.b_obs, target_q)

        return losses

    # proportional prioritization sampling
    def sum_tree_sample(self, k):
        idxes = []
        is_weights = np.empty((k, 1))
        self.beta = min(1., self.beta + self.beta_increment_per_sample)
        # calculate max_weight
        min_prob = np.min(self.replay_buffer.tree[-self.replay_buffer.capacity:]) / self.replay_buffer.total_p
        max_weight = np.power(self.buffer_size * min_prob, -self.beta)
        segment = self.replay_buffer.total_p / k
        for i in range(k):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, t = self.replay_buffer.get_leaf(s)
            idxes.append(idx)
            self.b_obs[i], self.b_actions[i], self.b_rewards[i], self.b_next_states[i], self.b_dones[i] = t
            # P(j)
            sampling_probabilities = p / self.replay_buffer.total_p     # where p = p ** self.alpha
            is_weights[i, 0] = np.power(self.buffer_size * sampling_probabilities, -self.beta) / max_weight
        return idxes, is_weights

    def evaluation(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        # one episode until done
        while not done:
            action, q_values = self.model.action_value(obs[None])  # Using [None] to extend its dimension (4,) -> (1, 4)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if render:  # visually show
                env.render()
            time.sleep(0.05)
        env.close()
        return ep_reward

    # store transitions into replay butter, now sum tree.
    def store_transition(self, priority, obs, action, reward, next_state, done):
        transition = [obs, action, reward, next_state, done]
        self.replay_buffer.add(priority, transition)

    # minor modified from https://github.com/medipixel/rl_algorithms/blob/master/algorithms/common/helper_functions.py
    def get_n_step_info(self, n_step_buffer, gamma):
        """Return n step reward, next state, and done."""
        # info of the last transition
        reward, next_state, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            reward = r + gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done


    # rank-based prioritization sampling
    def rand_based_sample(self, k):
        pass

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
    target_model = Model(num_actions)
    agent = MSTDAgent(model, target_model, env)
    # test before
    rewards_sum = agent.evaluation(env)
    print("Before Training: %d out of 200" % rewards_sum)  # 9 out of 200

    agent.train()
    # test after
    # env = gym.wrappers.Monitor(env, './recording', force=True)
    rewards_sum = agent.evaluation(env)
    print("After Training: %d out of 200" % rewards_sum)  # 200 out of 200
