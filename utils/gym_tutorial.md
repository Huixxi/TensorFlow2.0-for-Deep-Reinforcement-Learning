# OpenAI Gym An Introduction
Official Docs: http://gym.openai.com/docs/  
Github: https://github.com/openai/gym  

## Installation
* Simply install `gym` using `pip3`:  
  `pip3 install gym`
  
* Full installation containing all environments  
  `pip3 install gym[all]`  
  You can ignore the failed building message of `mujoco-py`, which needs a license.
  
## Environment
Check all environment in gym using:  
* `print(gym.envs.registry.all())`
* `print([env.id for env in gym.envs.registry.all()])  # list version`

`['Copy-v0', 'RepeatCopy-v0', 'ReversedAddition-v0', 'ReversedAddition3-v0', 'DuplicatedInput-v0', 'Reverse-v0', 'CartPole-v0', 'CartPole-v1', 'MountainCar-v0', ...`

## Basic Usage
Take "CartPole-v0" environment as an example:  
```python
import gym
import time

env = gym.make("CartPole-v0")  # setup a environment for the agent
initial_observation = env.reset()
done = False

# one episode, when done is True, break.
while not done:
  env.render()  # make the environment visiable
  action = env.action_space.sample()  # randomly select an action from total actions the agent can take 
  next_observation, reward, done, info = env.step(action)
  time.sleep(0.1)  # for better display effect
  
env.close()  # close the environment
```
Here, the agent is a random agent that just take a random action in each step. You can change it as a **linear agent** or a **neural network agent** which accept the observation and return an action not randomly select from the action space.  
Note, `env.step(action)` that takes an action and returns four different things:  
* **observation (object):** an environment-specific object representing your observation of the environment. 
* **reward (float):** amount of reward achieved by the previous action.
* **done (boolean):** whether it’s time to reset the environment again. 
* **info (dict):** diagnostic information useful for debugging.

![](https://github.com/Huixxi/TensorFlow2.0-for-Deep-Reinforcement-Learning/blob/master/images/sards.png)

## Spaces
(Just copy from the official docs. Still take "CartPole-v0" as example.)  
Every environment(discrete) comes with an `action_space` and an `observation_space`. These attributes are of type `Space`, and they describe the format of valid actions and observations:  
```python
import gym
env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
```
The `Discrete` space allows a fixed range of non-negative numbers, so in this case valid `actions` are either `0` or `1`. The `Box` space represents an `n`-dimensional box, so valid `observations` will be an array of `4` numbers. We can also check the `Box`’s bounds:
```python
print(env.observation_space.high)
#> array([ 2.4       ,         inf,  0.20943951,         inf])
print(env.observation_space.low)
#> array([-2.4       ,        -inf, -0.20943951,        -inf])
```
`Box` and `Discrete` are the most common `Space`s. You can sample from a `Space` or check that something belongs to it:
```c++
from gym import spaces
space = spaces.Discrete(8)  # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
assert space.contains(x)
assert space.n == 8
```
For `CartPole-v0` one of the actions applies force to the left, and one of them applies force to the right.
