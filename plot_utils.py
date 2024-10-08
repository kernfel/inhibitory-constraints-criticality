import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import colorConverter
from brian2.units import msecond
import scipy.stats as stats
import seaborn as sns
import string
import styling


colors = {
    'exc': 'b',
    'inh': 'r'
}

def label_panel(ax, ord, lower=False, left=1):
    lb = ax.set_title(styling.panel_labels[ord], loc='left', y=1, va='top' if lower else 'baseline', **styling.label_kwargs, ha='left')
    bb_plotonly = ax.get_window_extent()
    bb_withdeco = ax.get_tightbbox()
    x = left*(bb_withdeco.xmin - bb_plotonly.xmin) / bb_plotonly.bounds[2]
    lb.set_position((x, 1))
    return lb


def fullwidth(height):
    return styling.fig_width, height * styling.fig_width / 6.29
def halfwidth(height):
    return styling.fig_halfwidth, height * styling.fig_halfwidth / 3


def grouped_bars(series, xlabels, slabels, ax, w0=0.7):
    x = np.arange(len(xlabels))  # the label locations
    n = len(series)
    width = w0/n  # the width of the bars

    for i, (s, label) in enumerate(zip(series, slabels)):
        ax.bar(x - w0/2 + i*width, s, width, label=label)

    ax.set_xticks(x, xlabels)
    ax.legend()


def plot_pulse_hist(histograms, index_N, index_t, dt, figsize=(10,15), grid=False, cmap='PiYG', vmin=None, vmax=None, symmetric=True, cscale=False):
    if type(index_t) == int:
        assert len(index_N.shape) == 1
        index_N = np.repeat(index_N.reshape(-1,1), index_t, 1)
        index_t = np.repeat(np.arange(index_t).reshape(1,-1), len(index_N), 0)
    histograms = np.asarray(histograms)
    x = np.arange(index_t.shape[-1] + 1)*dt/msecond
    y = np.arange(len(index_N)+1)
    if symmetric:
        if vmax is None:
            vmax = np.nanmax(np.abs(histograms))
        if vmin is None:
            vmin = -vmax
    else:
        if vmax is None:
            vmax = np.nanmax(histograms)
        if vmin is None:
            vmin = np.nanmin(histograms)

    fig, axs = plt.subplots(1, len(histograms), figsize=figsize, sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    axs = axs[0]
    orders = []
    for ax, hist in zip(axs, histograms):
        h = hist[index_N, index_t]
        order = 0
        if cscale:
            hmax, hmin = np.nanmax(h), np.nanmin(h)
            max_order = int(np.log10(vmax/hmax)) if hmax>0 else np.nan
            min_order = int(np.log10(vmin/hmin)) if hmin<0 else np.nan
            order = np.nanmin([max_order, min_order])
            if np.isnan(order):
                order = 1
        orders.append(order)
        m = ax.pcolormesh(x, y, h*10**order, vmin=vmin, vmax=vmax, cmap=cmap, shading='flat')
        ax.set_xlabel('Time after stimulus (ms)')
        if grid:
            ax.grid()
    axs[0].set_ylabel('Neuron #')
    cb = plt.colorbar(m, location='bottom', ax=axs, aspect=40, fraction=1/figsize[1], pad=.5/figsize[1])

    if cscale:
        return fig, axs, cb, orders
    else:
        return fig, axs, cb

def alpha_to_color(c, alpha, bg='white'):
    c = np.asarray(colorConverter.to_rgb(c))
    bg = np.asarray(colorConverter.to_rgb(bg))
    rgb = (1-alpha)*bg + alpha*c
    return rgb

def fill_ratios(*ratios, to=100):
    ratios = np.asarray(ratios)
    total = ratios[ratios>0].sum()
    remainder = to-total
    ratios[ratios<0] = remainder / (ratios<0).sum()
    return ratios

def inset_hist(ax, data, x=True, median_color='C1', rescale=True, **kwargs):
    if x:
        y = ax.get_ylim()
        if rescale:
            ax.set_ylim(top=y[1] + .1*(y[1]-y[0]))
        tx = ax.twinx()
    else:
        y = ax.get_xlim()
        if rescale:
            ax.set_xlim(right=y[1] + .1*(y[1]-y[0]))
        tx = ax.twiny()
    hargs = dict(color='grey', histtype='stepfilled', edgecolor='dimgrey', orientation='vertical' if x else 'horizontal')
    hargs.update(**kwargs)
    binned, bins, *_ = tx.hist(data, 20, **hargs)
    ci = stats.bootstrap([data], np.median, n_resamples=10000)
    lo, hi = ci.confidence_interval
    bsize = np.diff(bins)[0]
    mask = (bins > lo - bsize) & (bins < hi + bsize)
    y = bins[mask]
    y[0], y[-1] = lo, hi
    x_in_ci = binned[mask[1:]].copy()
    x_in_ci[0] = x_in_ci[1]

    if x:
        tx.fill_between(y, x_in_ci, step='pre', ec='dimgrey', fc=median_color)
        tx.set_ylim(bottom=-10*binned.max())
        tx.set_yticks([])
    else:
        tx.fill_betweenx(y, x_in_ci, step='pre', ec='dimgrey', fc=median_color)
        tx.set_xlim(left=-10*binned.max())
        tx.set_xticks([])
    sns.despine(ax=tx)
    return tx
