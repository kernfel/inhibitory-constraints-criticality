import sys
import os
import warnings
import numpy as np
import deepdish as dd
import pandas as pd
from scipy.spatial import distance
from brian2 import second

from SNN import SNN
from sweep import get_script
from spike_utils import compute_delta_p_array


analysis_path = 'results/lif_alpha_beta_1_different_net_seed_0_analysis'


def path_pr(pr):
    return str(int(pr)) if pr >= 1 else f'0{int(10*pr)}'


def compute_final_delta_p(spike_stats, final_hours):
    t_spike, delta_p = compute_delta_p_array(spike_stats)
    tmask = t_spike >= t_spike[-1] - final_hours
    delta_p_mean = np.mean(delta_p[:, tmask], axis=1)  # (net,)
    return delta_p_mean


def compute_final_degrees(W_stats, final_i):
    lists_degree=[]
    for W_matrix in W_stats['matrix'].values():
        k_exc_means=[]
        k_inh_means=[]
        for i in final_i:
            network = W_matrix[i]>0.5
            out_degrees=network.sum(axis=1)
            k_exc_means.append(out_degrees[:80].mean())
            k_inh_means.append(out_degrees[80:].mean())
        k_exc_means=np.array(k_exc_means)
        k_inh_means=np.array(k_inh_means)
        lists_degree.append({'k_exc_steady': k_exc_means.mean(), 'k_inh_steady': k_inh_means.mean()})
    return np.asarray(lists_degree)  # (net, {})


def compute_initial_degrees(W_stats):
    lists_degree=[]
    for W_matrix in W_stats['matrix'].values():
        network = ~np.isnan(W_matrix[0])
        out_degrees=network.sum(axis=1)
        k_exc_mean = out_degrees[:80].mean()
        k_inh_mean = out_degrees[80:].mean()
        lists_degree.append({'k_exc_init': k_exc_mean, 'k_inh_init': k_inh_mean})
    return np.asarray(lists_degree)  # (net, {})


def get_distances(script, path):
    distances = []

    N = script.params['N']
    N_nets = script.params['N_nets']
    N_exc = int(N * (1 - script.params.get('inhibitory_ratio', 0.2)))
    N_inh = N - N_exc

    #positionsベクトルのnetworkごとの割り振り
    netidx = np.empty(N*N_nets)
    for net in range(N_nets):
        netidx[net*N_exc:(net+1)*N_exc] = net
        netidx[N_nets*N_exc + net*N_inh:N_nets*N_exc + (net+1)*N_inh] = net

    netstate = dd.io.load(f'{path}/maturation_0.h5')

    for net in range(N_nets):
        x_network=netstate['xloc'][netidx==net]
        y_network=netstate['yloc'][netidx==net]
        positions_network=np.stack([x_network,y_network],1)
        dist_M = distance.cdist(positions_network, positions_network, metric='euclidean')#距離行列
        distances.append(dist_M)
    
    return distances


def get_radii(distances, W_stats):
    radii = []
    for net, dist_M in enumerate(distances):
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=RuntimeWarning, message='invalid value encountered in divide')
            
            #initial でのradius meanを計算
            network_initial= ~np.isnan(W_stats['matrix'][net][-1])
            dist_M_initial=dist_M*network_initial
            r_initial_exc_mean=(dist_M_initial.sum(axis=1)/network_initial.sum(axis=1))[:80].mean()
            r_initial_inh_mean=np.nanmean((dist_M_initial.sum(axis=1)/network_initial.sum(axis=1))[80:])
            
            #lastでのradius meanを計算
            network_last= W_stats['matrix'][net][-1]>0.5 #saigonoweightをbinalize
            dist_M_last=dist_M*network_last
            r_last_exc_mean=np.nanmean((dist_M_last.sum(axis=1)/network_last.sum(axis=1))[:80])
            r_last_inh_mean=np.nanmean((dist_M_last.sum(axis=1)/network_last.sum(axis=1))[80:])

        radii.append({'r_exc_mean_init': r_initial_exc_mean, 'r_inh_mean_init': r_initial_inh_mean,
                      'r_exc_mean_last': r_last_exc_mean, 'r_inh_mean_last': r_last_inh_mean})
    return radii  # (net, {})


def get_final_weights(W_stats, final_i):
    final_weights = []
    for W_matrix in W_stats['matrix'].values():
        EE, EI, IE, II = [], [], [], []
        Ex, Ix = [], []
        for i in final_i:
            network=W_matrix[i]
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=RuntimeWarning, message='Mean of empty slice')
                EE.append(np.nanmean(network[:80,:80]))
                EI.append(np.nanmean(network[:80,80:]))
                IE.append(np.nanmean(network[80:,:80]))
                II.append(np.nanmean(network[80:,80:]))
                Ex.append(np.nanmean(network[:80, :]))
                Ix.append(np.nanmean(network[80:, :]))
        final_weights.append({
            'EE_mean': np.mean(EE), 'EI_mean': np.mean(EI),
            'IE_mean': np.mean(IE), 'II_mean': np.mean(II),
            'Ex_mean': np.mean(Ex), 'Ix_mean': np.mean(Ix)})
    return final_weights  # (net, {})


def get_df(results_path, final_iterations, p_list, r_list):
    rows = []
    for p in p_list:
        for r in r_list:
            distances = None  # Cache, because network structure is conserved across runs
            for run, runseed in enumerate(range(0,500,100)):
                print(f'Processing p={p}, r={r}, run={run}')
                script, path = get_script(results_path.format(p=path_pr(p), r=path_pr(r), runseed=runseed))
                p_inh=script.params['p_connection']['IE']
                r_inh=script.params['radius_inh']

                spike_stats = dd.io.load(f'{path}/spike_stats_12ms.h5')
                W_stats = dd.io.load(f'{path}/W_stats2.h5')
                if distances is None:
                    distances = get_distances(script, path)

                final_hours = int(script.runtime/3600/second) * final_iterations
                final_i = np.flatnonzero(W_stats['time'] >= W_stats['time'][-1]+W_stats['time'][1] - final_hours*3600)

                delta_p = compute_final_delta_p(spike_stats, final_hours)
                degrees_final = compute_final_degrees(W_stats, final_i)
                degrees_initial = compute_initial_degrees(W_stats)
                final_radii = get_radii(distances, W_stats)
                final_weights = get_final_weights(W_stats, final_i)

                for net, (dp, degf, degi, rad, w) in enumerate(zip(
                          delta_p, degrees_final, degrees_initial, final_radii, final_weights)):
                    rows.append({
                        'p_inh': p_inh, 'r_inh': r_inh, 'runseed': runseed, 'net': net,
                        'delta_p': dp, **degf, **degi, **rad, **w
                    })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: python {os.path.basename(__file__)} FINAL_ITERATIONS')
        exit(1)

    results_path = 'lif_alpha_beta_1_different_net_seed_0_pinh_{p}_rinh_{r}_runseed_{runseed}'
    final_iterations = int(sys.argv[1])

    p_list = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
    r_list = np.array([0.5, 1.0, 2.0, 3.0, 4.0])

    df = get_df(results_path, final_iterations, p_list, r_list)
    df.to_csv(f'{analysis_path}/criticality_data.csv')
