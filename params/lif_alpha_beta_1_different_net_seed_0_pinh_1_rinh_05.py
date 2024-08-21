import numpy as np
from brian2 import second, defaultclock

runtime = 60 * 60 * second
recording_period = 10 * second
iterations = 15

params = dict(
    N_nets=10,
    N=100,
    alpha=1.,
    beta=1.,
    delta_estdp=.01,
    delta_istdp=.01,
    gmax_exc=8.,
    gmax_inh=8.,

    layout='circle',
    radius_net=2.,
    radius_exc=1.3,
    radius_inh=.5,
    p_connection={'EE': .6, 'EI': .6, 'IE': 1., 'II': 1.},

    model='LIF',
    vnoise_mean=5.,
    vnoise_std=5.,
    rng=0,
    identical_nets=False,
    dt=1e-3*second
)

device = 'cpp_standalone'
