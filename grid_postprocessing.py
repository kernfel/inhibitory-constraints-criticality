import numpy as np
import deepdish as dd
import scipy.stats as stats
import pandas as pd

from grid_criticality import analysis_path, path_pr, get_distances
from sweep import get_script


nbins = 4
r_bins = np.arange(1,nbins,1)

dt_max = 50e-3  # s
velocity = .15  # m/s
slice_size = 5  # minutes

DT = 1e-3
y = np.arange(-dt_max, dt_max+1e-6, DT)
y[np.argmin(np.abs(y))] = 0  # rather than some floating point error.
nearly_sync = np.abs(y) < 5.5*DT

p_list = np.array([1.0, 0.7, 0.5, 0.3, 0.1])
r_list = np.array([0.5, 1.0, 2.0, 3.0, 4.0])


def get_global_idx(script, net, local_idx):
    N, N_nets = script.params['N'], script.params['N_nets']
    inh_ratio = script.params.get('inhibitory_ratio', 0.2)
    N_exc = int(N*(1-inh_ratio))
    N_inh = N - N_exc

    if local_idx < N_exc:
        return net*N_exc + local_idx
    else:
        return N_nets*N_exc + net*N_inh + local_idx-N_exc


def get_coincidence_single(tpre, tpost, delay, dt_max):
    ipost = 0
    t0 = int(dt_max/1e-3)
    coincidence = np.zeros(1 + 2*t0, int)
    for t in tpre + delay:  # Presynaptic spike arrival times

        # Forward ipost through old spikes
        while ipost < len(tpost) and tpost[ipost] < t - dt_max:
            ipost += 1

        # Run jpost through all spikes in the coincidence window (+- dt_max)
        for jpost in range(ipost, len(tpost)):
            if tpost[jpost] > t + dt_max:
                break
            dt = tpost[jpost] - t  # dt := post minus pre
            coincidence[t0 + int(dt/1e-3)] += 1
    return coincidence


def get_binned_coincidences(script, all_spikes, net, distances, W_matrix):
    nbins = len(r_bins) + 1
    binned_dist = np.digitize(distances[net], r_bins, right=True)
    prepost = []
    for bin in range(nbins):
        pre, post = np.nonzero((binned_dist==bin) & ~np.isnan(W_matrix[0]))
        mask = pre >= 80
        prepost.append((pre[mask], post[mask]))
    coincidences = np.zeros((nbins, len(y)), int)
    gindex = [get_global_idx(script, net, i) for i in range(100)]
    for spikes in all_spikes:
        tindex = [spikes['t'][spikes['i'] == i] for i in gindex]
        for bin in range(nbins):
            coinc = []
            for pre, post in zip(*prepost[bin]):
                if pre < 80:
                    continue

                tpre, tpost = tindex[pre], tindex[post]
                for islice in range(60//slice_size):
                    tpre_slice = tpre[(tpre >= islice*5*60) & (tpre < (islice+1)*5*60)]
                    tpost_slice = tpost[(tpost >= islice*5*60 - dt_max) & (tpost < (islice+1)*5*60 + dt_max)]

                    coincidence = get_coincidence_single(tpre_slice, tpost_slice, 0, dt_max)
                    coinc.append(coincidence)
            coincidences[bin] += np.sum(coinc, axis=0, dtype=coincidences.dtype)
    return coincidences


def get_centrality_indicators(coincidences):
    maxima = [y[np.argmax(x)] if np.ptp(x) > 0 else np.nan for x in coincidences]
    nearly_sync_ratios = [x[nearly_sync].sum() / x.sum() for x in coincidences]
    return maxima, nearly_sync_ratios




if __name__ == '__main__':
    import sys

    try:
        runseeds = [int(arg) for arg in sys.argv[1:]]
    except ValueError:
        runseeds = []
    if len(runseeds) == 0:
        print('Usage: python {__file__} [runseed[s] (integer[s])]')
        exit(1)

    print('Digesting weight matrices...')
    print('p_inh\tr_inh\tseed\tnet')
    rp_maxima, rp_sync_ratios = np.zeros((2, 5, 5, 50, nbins))
    correlations = []
    for ip, p_inh in enumerate(p_list):
        for ir, r_inh in enumerate(r_list):
            distances = None
            inet = 0
            for runseed in runseeds:
                script, path = get_script(f'lif_alpha_beta_1_different_net_seed_0_pinh_{path_pr(p_inh)}_rinh_{path_pr(r_inh)}_runseed_{runseed}')
                all_spikes = [dd.io.load(f'{path}/spikes_{iteration}.h5') for iteration in range(script.iterations)]

                N = script.params['N']
                N_exc = int(N * (1 - script.params.get('inhibitory_ratio', 0.2)))

                if distances is None:
                    distances = get_distances(script, path)
                    lsyn = [np.asarray([d[pre,post] for pre in range(N_exc,N) for post in range(N)]) for d in distances]
                
                W_stats = dd.io.load(f'{path}/W_stats2.h5')
                
                # Final weights
                dd.io.save(f'{path}/W_final.h5', {net: m[-1] for net, m in W_stats['matrix'].items()})
                
                for net, W_matrix in W_stats['matrix'].items():
                    
                    # Distance x weight correlations
                    Wsyn = np.asarray([np.mean([w[pre,post] for w in W_matrix]) for pre in range(N_exc,N) for post in range(N)])
                    mask = ~np.isnan(Wsyn)
                    reg = stats.linregress(lsyn[net][mask], Wsyn[mask])
                    correlations.append({'p_inh': p_inh, 'r_inh': r_inh, 'runseed': runseed, 'net': net,
                                        'rvalue': reg.rvalue, 'pvalue': reg.pvalue, 'slope': reg.slope, 'intercept': reg.intercept})
                    
                    # Spike coincidences
                    coinc = get_binned_coincidences(script, all_spikes, net, distances, W_matrix)
                    maxima, sync_ratios = get_centrality_indicators(coinc)
                    rp_maxima[ip, ir, inet] = maxima
                    rp_sync_ratios[ip, ir, inet] = sync_ratios

                    # Print
                    print(f'{ip}/4\t{ir}/4\t{runseed}\t{net}/9 (inet={inet}/49)')
                    inet += 1

    pd.DataFrame(correlations).to_csv(f'{analysis_path}/length_weight_correlations.csv')

    np.save(f'{analysis_path}/rp_maxima.npy', rp_maxima)
    np.save(f'{analysis_path}/rp_sync_ratios.npy', rp_sync_ratios)

    print('Done.')
