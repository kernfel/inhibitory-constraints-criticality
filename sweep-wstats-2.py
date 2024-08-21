#code for computing statistical quantities during maturation
import sys
import os
import numpy as np
import deepdish as dd
from brian2 import second

from SNN import SNN, directions
from sweep import get_script


if len(sys.argv) < 3:
    print(
        f'Usage: python {os.path.basename(__file__)} OUTDIR '
        'SAMPLING_INTERVAL[second] [ITERATIONS]')
    exit(1)
script, path = get_script(sys.argv[1])
sampling_interval = int(sys.argv[2])  # interval between two sampling points

if len(sys.argv) > 3:
    iterations = int(sys.argv[3])
else:
    iterations = script.iterations
N_nets = script.params['N_nets']

runtime = script.runtime
recording_period = script.recording_period


assert sampling_interval % (recording_period / second) == 0
assert (runtime / second) % sampling_interval == 0

sampling_freq = int(sampling_interval/(recording_period/second))

# define data containers
W_mean = {net: {d: [] for d in directions}
          for net in range(N_nets)}  # average of weights for each connection type
W_median = {net: {d: [] for d in directions}
            for net in range(N_nets)}  # meidan of weights for each connection type
W_std = {net: {d: [] for d in directions}
         for net in range(N_nets)}  # std of weights for each connection type
delta_W = {net: {d: [] for d in directions}
           for net in range(N_nets)}  # weight apdates between adjacent timepoints
# sum of exc inputs into each neurons
E_input_sum = {net: [] for net in range(N_nets)}
# sum of exc inputs into each neurons
I_input_sum = {net: [] for net in range(N_nets)}
# 2D weight matrix, diagonal element == None
W_matrix = {net: [] for net in range(N_nets)}

# instantiate SNN model to use get weight matrix
snn = SNN(init_weights=False, **script.params)
snn.initialize_with(dd.io.load(f'{path}/maturation_0.h5'))  # Align structure

for i in range(iterations):
    print(f'processing {i} generation')
    try:
        W = dd.io.load(f'{path}/W_{i}.h5')  # load weights in generation i
    except OSError:
        i -= 1
        break
    # store last weights of i generation to comppute delta_W in t_step = 0
    if i > 0:
        Wmat_prev = Wmat
    Wmat = {d: snn.get_weight_matrix(direction=d, force_square=True, Wdict=W,
                                     missing=np.nan, subnet=None)
            for d in directions}
    Wmat_full = np.nansum(np.stack(list(Wmat.values())), 0)
    Wmat_full[np.prod(np.stack([np.isnan(Wmat[d]) for d in directions]),
                      axis=0, dtype=bool)] = np.nan
    sampling_points = np.arange(0, runtime/recording_period, sampling_freq,
                                dtype=int)

    for t_step in sampling_points:
        for net in range(N_nets):
            # calculate mean, median, and delta
            for d in directions:
                W_d_t = Wmat[d][net, ..., t_step]
                W_mean[net][d].append(np.nanmean(W_d_t))
                W_median[net][d].append(np.nanmedian(W_d_t))
                W_std[net][d].append(np.nanstd(W_d_t))
                if ((t_step == 0) & (i == 0)):
                    delta_W[net][d].append(np.nanmean(W_d_t))
                elif ((t_step == 0) & (i != 0)):
                    W_d_t_1 = Wmat_prev[d][net, ..., -sampling_freq]
                    delta_W[net][d].append(np.nanmean(abs(W_d_t - W_d_t_1)))
                else:
                    W_d_t_1 = Wmat[d][net, ..., t_step-sampling_freq]
                    delta_W[net][d].append(np.nanmean(abs(W_d_t - W_d_t_1)))

            # load 2D matrix
            W_2D = Wmat_full[net, ..., t_step]
            W_matrix[net].append(W_2D)

            W_E_input_mean = np.nansum(W_2D[:snn.N_exc, :], axis=0)
            E_input_sum[net].append(W_E_input_mean)

            W_I_input_mean = np.nansum(W_2D[snn.N_exc:, :], axis=0)
            I_input_sum[net].append(W_I_input_mean)

t = np.arange(0, runtime * (i+1)/second, sampling_interval)  # time points
dd.io.save(f'{path}/W_stats2.h5', dict(
    time=t, mean=W_mean, median=W_median, std=W_std, delta=delta_W,
    matrix=W_matrix, E_sum_neuron=E_input_sum, I_sum_neuron=I_input_sum))
