import argparse
import logging, coloredlogs
import common
import os
import sys
import math
import random
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.misc
import scipy.stats

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


def bar_chart(sources=None, values=None, res=None, error=None, xlabel=None, title=None):
    if res is not None:
        sources = [x[0] for x in res]
        values = [x[1] for x in res]

    plt.rcdefaults()
    y_pos = np.arange(len(sources))
    plt.barh(y_pos, values, align='center', xerr=error, alpha=0.4)
    plt.yticks(y_pos, sources)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


# Main - argument parsing + processing
class App(object):
    def __init__(self, *args, **kwargs):
        self.args = None
        self.tester = None
        self.term_map = []

    def defset(self, val, default=None):
        return val if val is not None else default

    def independence_test(self, term_eval, ddeg=3, vvar=10):
        """
        Experimental verification of term independence.
        :param term_eval:
        :param ddeg:
        :param vvar:
        :return:
        """
        tterms = common.comb(vvar, ddeg)
        print('Independence test C(%d, %d) = %s' % (vvar, ddeg, tterms))
        ones = [0] * common.comb(vvar, ddeg, True)

        for val in common.pos_generator(dim=vvar, maxelem=1):
            for idx, term in enumerate(common.term_generator(ddeg, vvar - 1)):
                ones[idx] += term_eval.eval_term_raw_single(term, val)
        print('Done')
        print(ones)
        # TODO: test slight bias - in the allowed boundaries...

    def work(self):
        blocklen = int(self.defset(self.args.blocklen, 128))
        deg = int(self.defset(self.args.degree, 3))
        tvsize_orig = long(self.defset(self.args.tvsize, 1024*256))
        zscore_thresh = float(self.args.conf)
        rounds = int(self.args.rounds) if self.args.rounds is not None else None
        top_k = int(self.args.topk) if self.args.topk is not None else None
        top_comb = int(self.defset(self.args.combdeg, 2))
        reffile = self.defset(self.args.reffile)

        # prebuffer map 3deg terms
        logger.info('Precomputing term mappings')
        term_map = [[] for x in range(deg+1)]
        for dg in range(1, deg+1):
            for x in common.term_generator(deg, blocklen-1):
                term_map[dg].append(x)

        # specific polynomial testing
        poly_test = [
            [[0]],
            [[0,1]],
            [[0,1,2]],
            [[0,1,2],[0]],
            [[0,1,2],[0,1]],
            [[0,1,2],[3]],
            [[0,1,2],[2,3,4]],
            [[0,1,2],[1,2,3]],
            [[0,1,2],[3,4,5]],
            [[5,6,7],[8,9,10]],
            [[5,6,7],[7,8,9]],
            [[1,2],[2,3],[1,3]],
            [[0,1,2],[2,3,4],[5,6,7]],
            [[0,1,2],[2,3,4],[1,2,3]],
        ]
        poly_acc = [0] * len(poly_test)

        # test polynomials
        term_eval = common.TermEval(blocklen=blocklen, deg=deg)
        for idx, poly in enumerate(poly_test):
            print('Test poylnomial: %02d, %s' % (idx, poly))
            expp = term_eval.expp_poly(poly)
            print('  Expected probability: %s' % expp)

        # read file by file
        for file in self.args.files:
            tvsize = tvsize_orig

            if not os.path.exists(file):
                logger.error('File does not exist: %s' % file)

            size = os.path.getsize(file)
            logger.info('Testing file: %s, size: %d kB' % (file, size/1024.0))

            # size smaller than TV? Adapt tv then
            if size < tvsize:
                logger.warning('File size is smaller than TV, updating TV to %d' % size)
                tvsize = size

            term_eval = common.TermEval(blocklen=blocklen, deg=deg)

            total_terms = long(round(scipy.misc.comb(blocklen, deg)))
            logger.info('BlockLength: %d, deg: %d, terms: %d' % (blocklen, deg, total_terms))

            # read the file until there is no data.
            # TODO: sys.stdin
            with open(file, 'r') as fh:
                data_read = 0
                cur_round = 0
                total_fails = []
                total_hws = [0] * total_terms
                total_n = 0

                while data_read < size:
                    if rounds is not None and cur_round > rounds:
                        break

                    data = fh.read(tvsize)
                    bits = common.to_bitarray(data)
                    if len(bits) == 0:
                        logger.info('File read completely')
                        break

                    # pre-compute
                    logger.info('Pre-computing with TV, deg: %d, blocklen: %04d, tvsize: %08d, round: %d, avail: %d' %
                                (deg, blocklen, tvsize, cur_round, len(bits)))

                    term_eval.load(bits)
                    cur_round += 1

                    # evaluate all terms of the given degree
                    logger.info('Evaluating all terms')
                    probab = term_eval.expp_term_deg(deg)
                    exp_count = term_eval.cur_evals * probab
                    hws = term_eval.eval_terms(deg)

                    for idx, x in enumerate(hws):
                        total_hws[idx] += x
                    total_n += term_eval.cur_evals

                    difs = [(abs(x-exp_count), idx) for idx,x in enumerate(hws)]
                    difs.sort(key=lambda x: x[0], reverse=True)
                    zscores = [common.zscore(x, exp_count, term_eval.cur_evals) for x in hws]

                    # top x diffs
                    for x in difs[0:10]:
                        observed = hws[x[1]]
                        zscore = common.zscore(observed, exp_count, term_eval.cur_evals)
                        fail = 'x' if abs(zscore) > zscore_thresh else ' '
                        print(' - zscore: %+05.5f, observed: %08d, expected: %08d %s idx: %6d, term: %s'
                              % (zscore, observed, exp_count, fail, x[1], term_map[deg][x[1]]))

                    mean = sum(hws)/float(len(hws))
                    mean_zscore = sum(zscores)/float(len(zscores))

                    fails = sum([1 for x in zscores if abs(x) > zscore_thresh])
                    fails_fraction = float(fails)/len(zscores)
                    total_fails.append(fails_fraction)
                    print('Mean value: %s' % mean)
                    print('Mean zscore: %s' % mean_zscore)
                    print('Num of fails: %s = %02f.5%%' % (fails, 100.0*fails_fraction))

                    # take top X best polynomials
                    top_poly = []
                    if top_k is not None:
                        top_k_cur = len(difs) if top_k < 0 else top_k
                        top_terms_idx = set([x[1] for x in difs[:top_k_cur]])
                        top_terms = []
                        for idx, term in enumerate(term_eval.term_generator(deg)):
                            if idx in top_terms_idx:
                                top_terms.append(term)

                        # Combine & store the results - XOR
                        top_res = []
                        logger.info('Combining...')
                        Combined = collections.namedtuple('Combined', ['poly', 'expp', 'exp_cnt', 'obs_cnt', 'zscore'])

                        for top_comb_cur in range(2, top_comb+1):
                            for idx, places in enumerate(common.term_generator(top_comb_cur, top_k-1)):
                                # Create a new polynomial
                                poly = [top_terms[x] for x in places]
                                # Compute expected value
                                expp = term_eval.expp_poly(poly)
                                # Expected counts
                                exp_cnt = term_eval.cur_evals * expp
                                # Evaluate polynomial
                                obs_cnt = term_eval.hw(term_eval.eval_poly(poly))
                                # ZScore
                                zscore = common.zscore(obs_cnt, exp_cnt, term_eval.cur_evals)
                                comb = Combined(poly, expp, exp_cnt, obs_cnt, zscore)
                                top_res.append(comb)

                            # Combine & store results - AND
                            logger.info('Combining...')
                            Combined = collections.namedtuple('Combined', ['poly', 'expp', 'exp_cnt', 'obs_cnt', 'zscore'])
                            for idx, places in enumerate(common.term_generator(top_comb_cur, top_k - 1)):
                                # Create a new polynomial
                                poly = [reduce(lambda x, y: x + y, [top_terms[x] for x in places])]
                                # Compute expected value
                                expp = term_eval.expp_poly(poly)
                                # Expected counts
                                exp_cnt = term_eval.cur_evals * expp
                                # Evaluate polynomial
                                obs_cnt = term_eval.hw(term_eval.eval_poly(poly))
                                # ZScore
                                zscore = common.zscore(obs_cnt, exp_cnt, term_eval.cur_evals)
                                comb = Combined(poly, expp, exp_cnt, obs_cnt, zscore)
                                top_res.append(comb)

                        logger.info('Evaluating')
                        top_res.sort(key=lambda x: abs(x.zscore), reverse=True)
                        for i in range(min(len(top_res), 10)):
                            comb = top_res[i]
                            print(' - best poly zscore %9.5f, expp: %.4f, exp: %4d, obs: %s, diff: %f %%, poly: %s'
                                  % (comb.zscore, comb.expp, comb.exp_cnt, comb.obs_cnt,
                                     100.0*(comb.exp_cnt - comb.obs_cnt)/comb.exp_cnt, sorted(comb.poly)))

                    # Polynomial test here
                    for idx, poly in enumerate(poly_test):
                        poly_acc[idx] += term_eval.hw(term_eval.eval_poly(poly))

                # test polynomials
                for idx, poly in enumerate(poly_test):
                    print('Test polynomial %02d: %s' % (idx, poly))
                    expp = term_eval.expp_poly(poly)
                    exp_cnt = total_n * expp
                    obs_cnt = poly_acc[idx]
                    zscore = common.zscore(obs_cnt, exp_cnt, total_n)
                    print('  Expected probability: %s' % expp)
                    print('  Expected cnt: %10d, observed cnt: %10d, diff: %s, diff %02f %%'
                          % (exp_cnt, obs_cnt, exp_cnt-obs_cnt, 100.0*(exp_cnt-obs_cnt) / float(exp_cnt)))
                    print('  Zscore: %s' % zscore)

                total_fails_avg = float(sum(total_fails)) / len(total_fails)
                print('Total fails: %s' % total_fails)
                print('Total fails avg: %f%%' % (100.0*total_fails_avg))

                exp_count = term_eval.expp_term_deg(deg) * float(total_n)
                difs = [(abs(x - exp_count), idx) for idx, x in enumerate(total_hws)]
                difs.sort(key=lambda x: x[0], reverse=True)
                zscores = [common.zscore(x, exp_count, total_n) for x in total_hws]

                # top x diffs
                for x in difs[0:10]:
                    observed = total_hws[x[1]]
                    zscore = common.zscore(observed, exp_count, total_n)
                    fail = 'x' if abs(zscore) > zscore_thresh else ' '
                    print(' - zscore: %+05.5f, observed: %08d, expected: %08d %s idx: %6d, term: %s'
                          % (zscore, observed, exp_count, fail, x[1], term_map[deg][x[1]]))

                mean = sum(total_hws) / float(len(total_hws))
                mean_zscore = sum(zscores) / float(len(zscores))

                fails = sum([1 for x in zscores if abs(x) > zscore_thresh])
                fails_fraction = float(fails) / len(zscores)
                total_fails.append(fails_fraction)
                print('Mean value: %s' % mean)
                print('Mean zscore: %s' % mean_zscore)
                print('Num of fails: %s = %02f.5%%' % (fails, 100.0 * fails_fraction))

                # bar_data = []
                # for idx, x in enumerate(total_hws):
                #     bar_data.append((idx, x-exp_count))
                # bar_chart(res=bar_data)

                bar_data = [[x, 0] for x in range(blocklen)]
                for x in difs:
                    observed = total_hws[x[1]]
                    zscore = common.zscore(observed, exp_count, total_n)
                    if abs(zscore) < zscore_thresh:
                        break
                    term = term_map[deg][x[1]]
                    for bit in term:
                        bar_data[bit][1] += 1
                bar_chart(res=bar_data)

    def main(self):
        logger.debug('App started')

        parser = argparse.ArgumentParser(description='PolyDist')
        parser.add_argument('-t', '--threads', dest='threads', type=int, default=None,
                            help='Number of threads to use')
        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')
        parser.add_argument('--verbose', dest='verbose', action='store_const', const=True,
                            help='enables verbose mode')

        parser.add_argument('--ref', dest='reffile',
                            help='reference file with random data')

        parser.add_argument('--block', dest='blocklen',
                            help='block size in bits')
        parser.add_argument('--degree', dest='degree',
                            help='maximum degree of computation')
        parser.add_argument('--tv', dest='tvsize',
                            help='Size of one test vector')
        parser.add_argument('-r', '--rounds', dest='rounds',
                            help='Maximal number of rounds')
        parser.add_argument('--top', dest='topk', default=30, type=int,
                            help='top K number of best distinguishers to combine together')
        parser.add_argument('--combine-deg', dest='combdeg', default=2, type=int,
                            help='default degree of combination')

        parser.add_argument('--conf', dest='conf', type=float, default=1.96,
                            help='Zscore failing threshold')

        parser.add_argument('--stdin', dest='verbose', action='store_const', const=True,
                            help='read data from STDIN')

        parser.add_argument('files', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='files to process')

        self.args = parser.parse_args()
        self.work()


# Launcher
app = None
if __name__ == "__main__":
    app = App()
    app.main()

