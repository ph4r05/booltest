import json
import collections
import itertools
import copy
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib
from scipy import stats
from scipy.misc import derivative


def unique_justseen(iterable, key=None):
    """List unique elements, preserving order. Remember only the element just seen."""
    # unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    # unique_justseen('ABBCcAD', str.lower) --> A B C A D
    return list(map(next, map(lambda x: x[1], itertools.groupby(iterable, key))))


def get_bins(iterable, nbins=1, key=lambda x: x, ceil_bin=False, full=False):
    vals = [key(x) for x in iterable]
    min_v = min(vals)
    max_v = max(vals)
    bin_size = ((1 + max_v - min_v) / float(nbins))
    bin_size = math.ceil(bin_size) if ceil_bin else bin_size
    bins = [[] for _ in range(nbins)]
    for c in iterable:
        cv = key(c)
        cbin = int((cv - min_v) / bin_size)
        bins[cbin].append(c)
    return bins if not full else (bins, bin_size, min_v, max_v, len(vals))


def get_distrib_fbins(iterable, bin_tup, non_zero=True):  # bins = bins, size, minv, maxv
    bins, size, minv, maxv, ln = bin_tup  # (idx+0.5)
    return [x for x in [(minv + (idx) * size, len(bins[idx]) / float(ln)) for idx in range(len(bins))] if
            not non_zero or x[1] > 0]


def get_distrib(iterable, nbins=100, key=lambda x: x, non_zero=True):
    inp = list(iterable)
    return get_distrib_fbins(iterable, get_bins(inp, nbins=nbins, key=key, full=True), non_zero=non_zero)


def get_bin_val(idx, bin_tup):
    return len(bin_tup[0][idx]) / float(bin_tup[4])


def get_bin_data(x, bin_tup):
    idx = binize(x, bin_tup)
    if idx is None:
        return 0, None
    return get_bin_val(idx, bin_tup), idx


def binize(x, bin_tup):  # returns bin idx where the x lies, None if out of region
    bins, size, minv, maxv, ln = bin_tup
    if x < minv or x > maxv + size:
        return None
    return int((x - minv) // size)


def binned_pmf(x, bin_tup):
    return get_bin_data(x, bin_tup)[0]


def integrate_pmf(x, bin_tup):  # idea: sum all samples below + scale current bin
    pass


def get_bin_start(idx, bin_tup):
    return bin_tup[2] + idx * bin_tup[1]


def build_integrator(bin_tup, sums=None):
    sums = bin_sums(bin_tup) if sums is None else sums

    def cf(x):
        idx = binize(x, bin_tup)
        if idx is None: return 0
        subsum = 0 if idx <= 0 else sums[idx - 1]
        curp = len(bin_tup[0][idx]) / float(bin_tup[4])
        binstart = bin_tup[2] + idx * bin_tup[1]
        return subsum + curp * ((x - binstart) / float(binsize))

    return cf


def bin_sums(bin_tup):
    binsums = []
    for ix in range(len(bin_tup[0])):
        cbin = len(bin_tup[0][ix]) / float(bin_tup[4])
        cbin = (cbin + binsums[ix - 1]) if ix > 0 else cbin
        binsums.append(cbin)
    return np.array(binsums)


def is_crossed(pt, bound, direction):
    return (bound >= pt) if direction > 0 else (bound <= pt)


def move_bound(bnd, dx, direction):
    return bnd + dx if direction > 0 else bnd - dx


def all_diffs(vals):
    diffs, ln = [], len(vals)
    for i in range(ln):
        for j in range(i, ln):
            diffs.append(vals[i] - vals[j])
    return diffs


def pvalue_comp(fnc, extremes, dx, bin_tup, by_bins=True):
    """Extremes = [(val, direction +1\-1)] """
    nints = len(extremes)
    areas = [0] * nints
    nbounds = [x[0] for x in extremes]
    nbins = [binize(x[0], bin_tup) for x in extremes]
    bmin = min(nbounds)
    bmax = max(nbounds)
    cp = 0
    iterc = 0
    results = []
    print('OK: ', nints, nbins, ' size: ', bin_tup[4])

    while cp <= 1.0:  # integration step
        iterc += 1
        if iterc > 10000:
            raise ValueError('exc')  # Hard-termination to avoid infinite cycle.

        # Integration by increasing pvalue and tabulating.
        # Each area grows at the same pace. pvalue is a sum of areas.
        # Termination - bounds are crossing / touching.

        # Integrate each area with one step but in such a way the area is the same.
        max_area = max(areas)
        min_area = min(areas)
        sum_area = sum(areas)
        err = max([abs(x) for x in all_diffs(areas)])
        areas_str = ['%.7f' % x for x in areas]
        # print('Main iter: %s, cp: %.7f, mina: %.7f, maxa: %.7f, suma: %.7f, err: %.7f, a: [%s], n: %s'
        #      % (iterc, cp, min_area, max_area, sum_area, err, ', '.join(areas_str), nbins))

        subit = 0
        while any([x <= min_area for x in areas]):
            subit += 1
            # print('.. subit: %s' % subit)

            for ix in range(nints):
                if areas[ix] > min_area:
                    continue
                if by_bins:
                    areas[ix] += get_bin_val(nbins[ix], bin_tup)
                    nbounds[ix] = get_bin_start(nbins[ix], bin_tup)
                    nbins[ix] = move_bound(nbins[ix], 1, extremes[ix][1])
                else:
                    areas[ix] += fnc(nbounds[ix])
                    nbounds[ix] = move_bound(nbounds[ix], dx, extremes[ix][1])
        cp = sum(areas)

        crit_int = [None] * nints
        for i in range(nints):
            crit_int[i] = (extremes[i][0], nbounds[i]) if extremes[i][1] > 0 else (nbounds[i], extremes[i][0])

        results.append((cp, crit_int, copy.deepcopy(areas), err))

    # print('Main iter: %s, cp: %s, mina: %s, maxa: %s, suma: %s, a: %s'
    #          % (iterc, cp, min(areas), max(areas), sum(areas), areas))
    # print('Total: %s' % (sum([get_bin_val(ix, bin_tup) for ix in range(len(bin_tup[0]))])))
    # print(json.dumps(results, indent=2))
    return results


def tabulate_pvals(val, nbins=200, abs_val=False, target_pvals=[0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01]):
    inp_iter = val['zscores']
    if abs_val:
        inp_iter = [abs(x) for x in inp_iter]

    bin_tup = get_bins(inp_iter, nbins=nbins, full=True)
    bb = get_distrib_fbins(inp_iter, bin_tup)

    bin_size = bin_tup[1]
    minv, maxv = bin_tup[2], bin_tup[3]
    bins = np.array([x[0] for x in bb])

    # Tabulate pvalues
    # build_integrator(bin_tup)
    extremes = [
        [minv, 1],
        [0, -1],
        [0, +1],
        [maxv, -1]
    ] if not abs_val else [
        [minv, 1],
        [maxv, -1]
    ]

    print('%s-%s-%s-%s-%s' % (val['method'], val['block'], val['deg'], val['comb_deg'], val['data_size']))
    pvals = pvalue_comp(lambda x: binned_pmf(x, bin_tup), extremes,
                        dx=1. / (nbins / 10.), bin_tup=bin_tup, by_bins=True)

    res_pdata = []
    for target in target_pvals:
        chosen = 0
        for i in range(len(pvals)):
            chosen = i
            if pvals[i][0] >= target:
                chosen = i - 1 if i > 0 else 0
                break

        cdata = pvals[chosen]
        res_pdata.append(collections.OrderedDict([
            ('pval_target', target),
            ('pval', cdata[0]),
            ('crit', cdata[1]),
            ('areas', cdata[2]),
            ('err', cdata[3]),
        ]))

    return collections.OrderedDict([
        ('method', val['method']),
        ('block', val['block']),
        ('deg', val['deg']),
        ('comb_deg', val['comb_deg']),
        ('data_size', val['data_size']),
        ('nsamples', len(inp_iter)),
        ('nbins', nbins),
        ('abs_val', abs_val),
        ('binsize', bin_size),
        ('minv', minv),
        ('maxv', maxv),
        ('extremes', extremes),
        ('pvals', res_pdata)
    ])


def main():
    js = json.load(open('ref_1554219251.json'))
    csv = open('ref_1554219251.csv').read()

    csv_data = []
    for rec in [x.strip() for x in csv.split("\n")]:
        p = rec.split(';')
        if len(p) < 6:
            continue
        cur = collections.OrderedDict([
            ('method', p[0]),
            ('block', int(p[1])),
            ('deg', int(p[2])),
            ('comb_deg', int(p[3])),
            ('data_size', int(p[4])),
            ('zscores', [float(x.replace(',','.')) for x in p[6:]])
        ])
        csv_data.append(cur)
    print(json.dumps(csv_data[0]))


    data = csv_data
    data_filt = [x for x in data if x and len(x['zscores']) > 1000]
    data_filt.sort(key=lambda x: (x['method'], x['block'], x['deg'], x['comb_deg'], x['data_size']))
    np.random.seed(87655677)


    pval_db = []
    for dix, val in enumerate(data_filt):
        res = tabulate_pvals(val, abs_val=True)
        pval_db.append(res)
        print('Dump %s' % dix)
    json.dump(pval_db, open('pval_db.json', 'w+'), indent=2)

    nbins = 200
    abs_val = True

    for dix, val in enumerate(data_filt):
        inp_iter = (val['zscores'])
        if abs_val:
            inp_iter = [abs(x) for x in inp_iter]

        print('%s[%s:%s:%s:%s]: %s %s'
              % (val['method'], val['block'], val['deg'], val['comb_deg'],
                 val['data_size'], len(val['zscores']),
                 '',  # dst.ppf([1-0.0001, 1-0.001, 1-0.01, 1-0.05, 1-0.10, 1-0.5, 0, 1, 0.0001, 0.001, 0.1, 0.9])
                 # dst.stats(moments='mvsk')
                 ))

        bin_tup = get_bins(inp_iter, nbins=nbins, full=True)
        bb = get_distrib_fbins(inp_iter, bin_tup)

        bin_size = bin_tup[1]
        minv, maxv = bin_tup[2], bin_tup[3]
        bins = np.array([x[0] for x in bb])
        dst = stats.rv_discrete(values=([x[0] for x in bb], [x[1] for x in bb]))
        print(stats.rv_discrete)

        x = np.array([bins[0], bins[1], bins[6]])
        print(dst.pmf(x))
        print(dst._pmf(x))

        # Tabulate pvalues
        build_integrator(bin_tup)
        extremes = [
            [minv, 1],
            [0, -1],
            [0, +1],
            [maxv, -1]
        ] if not abs_val else [
            [minv, 1],
            [maxv, -1]
        ]

        pvals = pvalue_comp(lambda x: binned_pmf(x, bin_tup), extremes,
                            dx=1. / (nbins / 10.), bin_tup=bin_tup, by_bins=True)

        n_sample = 100
        rvs = dst.rvs(size=n_sample)
        f, l = np.histogram(rvs, bins=bins)
        f = np.append(f, [0])
        probs = np.array([x[1] for x in bb])
        # print(bins, len(bins))
        # print(probs, len(probs))
        # print(f, len(f))
        # sfreq = np.vstack([np.array([x[0] for x in bb]), f, probs*n_sample]).T
        # print(sfreq)

        print('%s[%s:%s:%s:%s]: %s %s'
              % (val['method'], val['block'], val['deg'], val['comb_deg'],
                 val['data_size'], len(val['zscores']),
                 dst.ppf([1 - 0.0001, 1 - 0.001, 1 - 0.01, 1 - 0.05, 1 - 0.10, 1 - 0.5, 0, 1, 0.0001, 0.001, 0.1, 0.9])
                 # dst.stats(moments='mvsk')
                 ))

        x = np.linspace(min(bins), max(bins), 1000)
        plt.plot(x, dst.cdf(x))
        plt.show()

        cdf_dev = derivative(dst.cdf, x, dx=0.5)
        plt.plot(x, cdf_dev)

        sec_x = pvals[40]  # 49
        print('Plotting area under: ', sec_x)
        for ix in range(len(sec_x[1])):
            section = np.arange(sec_x[1][ix][0], sec_x[1][ix][1], 1 / 20.)
            plt.fill_between(section, derivative(dst.cdf, section, dx=0.5))
        plt.show()

        x = np.linspace(0, 100, 10000)
        plt.plot(x, dst.ppf(x))
        plt.show()

        x = np.linspace(minv, maxv, 10000)
        plt.plot(bins, dst._pmf(bins))
        plt.show()

        x = np.linspace(minv, maxv, 10000)
        plt.plot(x, [binned_pmf(y, bin_tup) for y in x])
        for ix in range(len(sec_x[1])):
            section = np.linspace(sec_x[1][ix][0], sec_x[1][ix][1],
                                  10000)  # np.arange(sec_x[1][ix][0], sec_x[1][ix][1], 1/20.)
            plt.fill_between(section, [binned_pmf(y, bin_tup) + 0.0005 for y in section])
        plt.show()

        # Idea: pvalue function = pms of the distribution.
        # If test returns z-score with p=0 then we reject the hypothesis as we didnt get such zscore
        # If test returns with p=0.3 we dont reject as we have our alpha set somehow...
        # Problem: number of bins. If too many, we have small probabilities -> some alphas not reachable.
        # if dix > 3:
        break


    a4_dims = (2*11.7, 8.27)
    fig, ax = pyplot.subplots(figsize=a4_dims)
    zs = data_filt[1]['zscores']

    for i in range(1):
        zs = [x for x in data_filt[i]['zscores']]
        print(len(zs))
        sns.distplot(a=zs, ax=ax, hist=True, norm_hist=False, bins='auto')


if __name__ == '__main__':
    main()
