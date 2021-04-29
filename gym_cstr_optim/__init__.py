from gym.envs.registration import register

register(
    id='awgn-v0',
    entry_point='gym_cstr_optim.envs.res_alloc.awgn:AWGN',
)
register(
    id='mai-v0',
    entry_point='gym_cstr_optim.envs.res_alloc.mai:MAI',
)