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
