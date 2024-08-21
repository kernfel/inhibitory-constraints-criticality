#code for computing statistical quantities of spikes in maturation
import sys
import os
import numpy as np
import deepdish as dd
from brian2 import second

from SNN import SNN
from sweep import get_script
from spike_utils import bin_spikes

if len(sys.argv) < 4:
    print(f'Usage: python {os.path.basename(__file__)} OUTDIR '
          'SAMPLING_INTERVAL[second] SPIKE_LENGTH[second] [ITERATIONS]')
    exit(1)
script, path = get_script(sys.argv[1])
sampling_interval = int(sys.argv[2])  # interval between two sampling points

spike_period = int(sys.argv[3])  # time length of each spike sequence

if len(sys.argv) > 4:
    iterations = int(sys.argv[4])
else:
    iterations = script.iterations
N_nets = script.params['N_nets']

runtime = script.runtime
recording_period = script.recording_period


assert sampling_interval % (recording_period / second) == 0
assert (runtime / second) % sampling_interval == 0
# guarantee no over lap between spike sequences
assert sampling_interval >= spike_period

sampling_freq = int(sampling_interval/(recording_period/second))

# mean, activity, avalanche
neuron_class = ['all', 'exc', 'inh']
t = []  # array of sampling time points
F_mean = {net: {n_type: [] for n_type in neuron_class}
          for net in range(N_nets)}  # mean firing rate
F_median = {net: {n_type: [] for n_type in neuron_class}
            for net in range(N_nets)}  # median firing rate
F_std = {net: {n_type: [] for n_type in neuron_class}
         for net in range(N_nets)}  # std firing rate
avalanche = {net: [] for net in range(N_nets)}  # avalanche size sequence

# instantiate SNN model to use get_rates
snn = SNN(init_weights=False, **script.params)
snn.initialize_with(dd.io.load(f'{path}/maturation_0.h5'))

netidx = np.empty(snn.N*snn.N_nets)
for net in range(snn.N_nets):
    netidx[net*snn.N_exc:(net+1)*snn.N_exc] = net
    netidx[snn.N_nets*snn.N_exc + net*snn.N_inh:
           snn.N_nets*snn.N_exc + (net+1)*snn.N_inh] = net
excitatory = np.ones(snn.N*snn.N_nets, bool)
excitatory[snn.N_exc*snn.N_nets:] = False

for i in range(iterations):
    print(f'processing {i} generation')
    try:
        spikes = dd.io.load(f'{path}/spikes_{i}.h5')
    except OSError:
        break
    spike_i = spikes['i']
    spike_t = spikes['t']

    sampling_points = np.arange(
        sampling_freq, runtime/recording_period + sampling_freq/2, sampling_freq)

    for t_step in sampling_points:
        # initial time of spike sequnce, 60s
        # t_start = t_step * (recording_period/second) - 10
        # last time of spike sequnce
        # t_end = t_step * (recording_period/second) + 10
        t.append((runtime/second) * i + t_step * (recording_period/second))

        # spike_i_partial = spike_i[(t_start < spike_t) & (spike_t < t_end)]
        # spike_t_partial = spike_t[(t_start < spike_t)
        #                           & (spike_t < t_end)] - t_start
        # spike_dict_test = dict(
        #     i=spike_i_partial, t=spike_t_partial, tmax=20*second)
        # rates_exc, rates_inh, rates_all, t_rates = snn.get_rates(
        #     spike_dict=spike_dict_test, sigma=20e-3*second)

        # compute avalanche, use spike_period[s] spike sequence
        # t_start = t_step * (recording_period/second) - \
        #     spike_period/2  # initial time of spike sequnce
        # t_end = t_step * (recording_period/second) + \
        #     spike_period/2  # last time of spike sequnce
        t_end = t_step * (recording_period/second)
        t_start = t_end - spike_period

        spike_i_partial = spike_i[(t_start < spike_t) & (spike_t < t_end)]
        spike_t_partial = spike_t[(t_start < spike_t)
                                  & (spike_t < t_end)] - t_start
        binned_spikes = bin_spikes(
            spike_i_partial, spike_t_partial, binsize=12e-3*second,
            N=snn.N*snn.N_nets)

        binsize = 1*second
        rate_spikes = bin_spikes(
            spike_i_partial, spike_t_partial, binsize=binsize,
            N=snn.N*snn.N_nets)

        for net in range(N_nets):
            rates_exc = rate_spikes[(netidx == net) & excitatory].mean(
                axis=0) / binsize * second
            rates_inh = rate_spikes[(netidx == net) & ~excitatory].mean(
                axis=0) / binsize * second
            rates_all = rate_spikes[netidx == net].mean(
                axis=0) / binsize * second
            F_mean[net]['exc'].append(rates_exc.mean())
            F_mean[net]['inh'].append(rates_inh.mean())
            F_mean[net]['all'].append(rates_all.mean())

            F_median[net]['exc'].append(np.median(rates_exc))
            F_median[net]['inh'].append(np.median(rates_inh))
            F_median[net]['all'].append(np.median(rates_all))

            F_std[net]['exc'].append(np.std(rates_exc))
            F_std[net]['inh'].append(np.std(rates_inh))
            F_std[net]['all'].append(np.std(rates_all))

            # compute avalanche
            binned_spike_i_sum = binned_spikes[netidx == net].sum(axis=0)

            avalanche_size = []
            spike_count = 0
            pre_nonzero_flag = False  # True if the number of spikes in previous bin is nonzero
            for bin_count in binned_spike_i_sum:
                if bin_count != 0:
                    spike_count += bin_count
                    pre_nonzero_flag = True
                if (bin_count == 0) & pre_nonzero_flag:
                    avalanche_size.append(spike_count)
                    spike_count = 0
                    pre_nonzero_flag = False
            avalanche[net].append(avalanche_size)

dd.io.save(f'{path}/spike_stats_12ms.h5', dict(time=t, mean=F_mean, median=F_median,
                                          std=F_std, avalanche=avalanche))
