from collections.abc import Iterable
import numpy as np
from scipy.optimize import curve_fit
from brian2 import second, Quantity
from SNN import ensure_unit


def get_clean_tbounds(tstart, tstop, interval):
    tstart = 0 if tstart is None else tstart
    tstop = interval if tstop is None else tstop
    if not isinstance(tstart, Quantity):
        tstart = tstart * second
    if not isinstance(tstop, Quantity):
        tstop = tstop * second
    if tstop <= 0 * second:
        tstop = interval
    if tstart < 0 * second:
        tstart = 0 * second
    return tstart, tstop


def iterspikes(spike_i, spike_t, n, interval, tstart=None, tstop=None):
    '''
    Iterates over `spike_i`/`spike_t` in `n` chunks of duration `interval`.
    Each iteration provides the chunk_i in the current chunk, as well as
    chunk_t relative to the start of the chunk. tstart and tstop can be used
    to exclude early/late parts of each chunk. Note that chunk_t is relative
    to the start of the returned chunk.
    '''
    if not isinstance(spike_t, Quantity):
        spike_t = spike_t * second
    tstart, tstop = get_clean_tbounds(tstart, tstop, interval)
    istart, iend = 0, 0
    for ipulse in range(n):
        ltstart = (ipulse * interval + tstart)
        ltstop = (ipulse * interval + tstop)
        while istart < len(spike_t) and spike_t[istart] < ltstart:
            istart += 1
        iend = istart
        while iend < len(spike_t) and spike_t[iend] < ltstop:
            iend += 1
        yield spike_i[istart:iend], spike_t[istart:iend] - ltstart
        istart = iend


def bin_spikes(spike_i, spike_t, binsize=1e-3*second, N=None, tmax=None,
               leading_shape=()):
    '''
    Discretizes spike trains into a neuron*bins matrix, preserving leading
        dimensions.
    Params:
        spike_i: (*, N) array-like of int, indices of spiking neurons
        spike_t: (*, N) array-like of seconds (explicit or implicit),
            time stamps of spikes. Must be the same shape as spike_i.
        binsize: Implicit or explicit seconds, temporal width of bins
        N: int, number of neurons. If None (default), this is deduced from the
            maximum entry in spike_i. Dictates the `N` dimension of the return
            value.
        tmax: Implicit or explicit seconds, total duration of the spike trains.
            If None (default), this is deduced from the maximum entry in
            spike_t. Dictates the `T` dimension of the return value after
            discretization.
        leading_shape: tuple of int, shape of leading dimensions of spike_i,
            spike_t, if any. Defaults to (), indicating no nested structure.
    Returns:
        (*, N, T) ndarray of int. The value of each entry (*, n, t) corresponds
            to the number of spikes fired by the n:th neuron in the t:th bin.
    '''
    spike_t = np.asarray(spike_t)
    spike_i = np.asarray(spike_i)
    binsize = ensure_unit(binsize, second)
    if N is None:
        N = np.max(spike_i) + 1
    if tmax is None:
        tmax = np.max(spike_t)
    tmax = ensure_unit(tmax, second)
    T = int(np.ceil(tmax/binsize)) + 1
    hist = np.zeros((*leading_shape, N, T))

    def bin_recursive(H, si, st, shape):
        if len(shape) > 0:
            for outer_dim_idx, (i, t) in enumerate(zip(si, st)):
                bin_recursive(H[outer_dim_idx], i, t, shape[1:])
        else:
            bin_idx = (ensure_unit(st.astype(float), second)
                       // binsize).astype(int)
            np.add.at(H, (si.astype(int), bin_idx), 1)

    bin_recursive(hist, spike_i, spike_t, leading_shape)
    return hist


def compute_delta_p_array(spike_stats):
    '''
    Computes a (net, time) tensor of delta_p.
    Returns t_spike, i.e., the time stamps for the second dimension, and delta_p.
    '''
    def power_law(x,a,b):
        y=a*pow(x,b)
        return y
    lists_delta_p_array=[]
    t_spike=np.array(spike_stats['time'])/3600
    for net in range(10):
        delta_p_array=[]
        for i in range(len(t_spike)):
            avalanche_size = spike_stats['avalanche'][net][i]
            avalanche_size = np.array(avalanche_size, dtype=int)
            freq=np.zeros(avalanche_size.max())
            for s in avalanche_size:
                freq[s-1] +=1
            size=np.arange(1,avalanche_size.max()+1)
            freq_nonzero=freq[freq!=0]
            size=size[freq!=0]
            prob=freq_nonzero/freq_nonzero.sum()
            popt,pcov = curve_fit(power_law, size, prob)
            delta_p = np.mean(np.log(prob)-np.log(power_law(size, popt[0], popt[1])))
            delta_p_array.append(delta_p)
        lists_delta_p_array.append(delta_p_array)
    return t_spike, np.asarray(lists_delta_p_array)
