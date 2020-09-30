from gym.envs.registration import register

register(
    id='res-env-v0',
    entry_point='resource_gen.envs:ResEnv',
)