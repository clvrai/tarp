from gym.envs.registration import register
import os

register(
    id="VizdoomD3Battle-v0",
    entry_point='tarp.rl.envs.vizdoom:D3BattleDoomEnv',
)


register(
    id="VizdoomD3BattleState-v0",
    entry_point='tarp.rl.envs.vizdoom:D3BattleDoomStateEnv',
)

register(
    id='carla-v0',
    entry_point='tarp.rl.envs.carla:CarlaEnv',
    kwargs={}
)

register(
    id='carla-state-v0',
    entry_point='tarp.rl.envs.carla:CarlaStateEnv',
    kwargs={}
)

