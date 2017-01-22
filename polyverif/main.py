import argparse
import logging, coloredlogs
import common
import os
import re
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


class HWAnalysis(object):
    """
    Analysis of all deg poly
    """
    def __init__(self, *args, **kwargs):
        self.term_map = []
        self.term_eval = None

        self.blocklen = None
        self.deg = 3
        self.top_k = None
        self.top_comb = None
        self.zscore_thresh = 1.96

        self.total_hws = []
        self.total_n = 0

    def init(self):
        logger.info('Precomputing term mappings')
        self.term_map = common.build_term_map(self.deg, self.blocklen)
        self.term_eval = common.TermEval(blocklen=self.blocklen, deg=self.deg)
        self.total_hws = [[0] * common.comb(self.blocklen, x, True) for x in range(self.deg + 1)]

    def proces_chunk(self, bits):
        # Compute the basis.
        self.term_eval.load(bits)

        # Evaluate all terms of degrees 1..deg
        logger.info('Evaluating all terms')
        hws2 = self.term_eval.eval_all_terms(self.deg)
        logger.info('Done: %s' % [len(x) for x in hws2])

        # Accumulate hws to the results.
        for d in range(1, self.deg+1):
            for i in range(len(self.total_hws[d])):
                self.total_hws[d][i] += hws2[d][i]
        self.total_n += self.term_eval.cur_evals

        # Done.
        self.analyse(hws2, self.term_eval.cur_evals)

    def finished(self):
        """
        All data read - final analysis.
        :return:
        """
        self.analyse(self.total_hws, self.total_n)

    def analyse(self, hws, num_evals):
        """
        Analyse hamming weights
        :param hws:
        :return:
        """

        probab = [self.term_eval.expp_term_deg(d) for d in range(0, self.deg + 1)]
        exp_count = [num_evals * x for x in probab]
        print(probab)
        print(exp_count)

        top_terms = []
        top_poly = []

        difs = [None] * (self.deg + 1)
        zscores = [None] * (self.deg + 1)
        for d in range(1, self.deg+1):
            difs[d] = [(abs(x - exp_count[d]), idx, d) for idx, x in enumerate(hws[d])]
            difs[d].sort(key=lambda x: x[0], reverse=True)
            zscores[d] = [common.zscore(x, exp_count[d], num_evals) for x in hws[d]]

            # Selecting TOP k polynomials
            for x in difs[d][0:15]:
                observed = hws[d][x[1]]
                zscore = common.zscore(observed, exp_count[d], num_evals)
                fail = 'x' if abs(zscore) > self.zscore_thresh else ' '
                print(' - zscore: %+05.5f, observed: %08d, expected: %08d %s idx: %6d, term: %s'
                      % (zscore, observed, exp_count[d], fail, x[1], self.term_map[d][x[1]]))

            # Take top X best polynomials
            if self.top_k is None:
                continue

            top_terms += [self.term_map[d][x[1]] for x in difs[d][0: (None if self.top_k < 0 else self.top_k)]]

            mean_zscore = sum(zscores[d])/float(len(zscores[d]))
            fails = sum([1 for x in zscores[d] if abs(x) > self.zscore_thresh])
            fails_fraction = float(fails)/len(zscores[d])
            # total_fails.append(fails_fraction)
            print('Mean zscore: %s' % mean_zscore)
            print('Num of fails: %s = %02f.5%%' % (fails, 100.0*fails_fraction))

        if self.top_k is None:
            return

        # Combine & store the results - XOR
        top_res = []
        logger.info('Combining...')
        Combined = collections.namedtuple('Combined', ['poly', 'expp', 'exp_cnt', 'obs_cnt', 'zscore'])

        comb_res = self.term_eval.new_buffer()
        comb_subres = self.term_eval.new_buffer()
        for top_comb_cur in range(1, self.top_comb + 1):
            for idx, places in enumerate(common.term_generator(top_comb_cur, len(top_terms) - 1)):
                # Create a new polynomial
                poly = [top_terms[x] for x in places]
                # Compute expected value
                expp = self.term_eval.expp_poly(poly)
                # Expected counts
                exp_cnt = num_evals * expp
                # Evaluate polynomial
                obs_cnt = self.term_eval.hw(self.term_eval.eval_poly(poly, res=comb_res, subres=comb_subres))
                # ZScore
                zscore = common.zscore(obs_cnt, exp_cnt, num_evals)
                comb = Combined(poly, expp, exp_cnt, obs_cnt, zscore)
                top_res.append(comb)

            # Combine & store results - AND
            for idx, places in enumerate(common.term_generator(top_comb_cur, len(top_terms) - 1)):
                # Create a new polynomial
                poly = [reduce(lambda x, y: x + y, [top_terms[x] for x in places])]
                # Compute expected value
                expp = self.term_eval.expp_poly(poly)
                # Expected counts
                exp_cnt = self.term_eval.cur_evals * expp
                # Evaluate polynomial
                obs_cnt = self.term_eval.hw(self.term_eval.eval_poly(poly, res=comb_res, subres=comb_subres))
                # ZScore
                zscore = common.zscore(obs_cnt, exp_cnt, num_evals)
                comb = Combined(poly, expp, exp_cnt, obs_cnt, zscore)
                top_res.append(comb)

        logger.info('Evaluating')
        top_res.sort(key=lambda x: abs(x.zscore), reverse=True)
        for i in range(min(len(top_res), 30)):
            comb = top_res[i]
            print(' - best poly zscore %9.5f, expp: %.4f, exp: %4d, obs: %s, diff: %f %%, poly: %s'
                  % (comb.zscore, comb.expp, comb.exp_cnt, comb.obs_cnt,
                     100.0 * (comb.exp_cnt - comb.obs_cnt) / comb.exp_cnt, sorted(comb.poly)))


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

    def get_testing_polynomials(self):
        return [
            [[0]],
            [[0, 1]],
            [[0, 1, 2]],
            [[0, 1, 2], [0]],
            [[0, 1, 2], [0, 1]],
            [[0, 1, 2], [3]],
            [[0, 1, 2], [2, 3, 4]],
            [[0, 1, 2], [1, 2, 3]],
            [[0, 1, 2], [3, 4, 5]],
            [[5, 6, 7], [8, 9, 10]],
            [[5, 6, 7], [7, 8, 9]],
            [[1, 2], [2, 3], [1, 3]],
            [[0, 1, 2], [2, 3, 4], [5, 6, 7]],
            [[0, 1, 2], [2, 3, 4], [1, 2, 3]],
        ]

    def get_multiplier(self, char, is_ib=False):
        """
        Returns the multiplier factor of the multiplier character. if ib is enabled, powers of
        1024 are returned, otherwise powers of 1000.

        :param char:
        :param is_ib:
        :return:
        """
        if char is None or len(char) == 0:
            return 1

        char = char[:1].lower()
        if char == 'k':
            return 1024L if is_ib else 1000L
        elif char == 'm':
            return 1024L * 1024L if is_ib else 1000L * 1000L
        elif char == 'g':
            return 1024L * 1024L * 1024L if is_ib else 1000L * 1000L * 1000L
        elif char == 't':
            return 1024L * 1024L * 1024L * 1024L if is_ib else 1000L * 1000L * 1000L * 1000L
        else:
            raise ValueError('Unknown multiplier %s' % char)

    def process_size(self, size_param):
        """
        Processes size parameter and evaluates the multipliers (e.g., 3M).
        :param size_param:
        :return:
        """
        if size_param is None:
            return None

        if isinstance(size_param, (int, long)):
            return size_param

        if not isinstance(size_param, basestring):
            raise ValueError('Unknown type of the input parameter')

        if len(size_param) == 0:
            return None

        if size_param.isdigit():
            return long(size_param)

        matches = re.match('^([0-9a-fA-F]+(.[0-9]+)?)([kKmMgGtT]([iI])?)?$', size_param)
        if matches is None:
            raise ValueError('Unknown size specifier')

        is_ib = matches.group(4) is not None
        mult_char = matches.group(3)
        multiplier = self.get_multiplier(mult_char, is_ib)
        return long(float(matches.group(1)) * multiplier)

    def work(self):
        blocklen = int(self.defset(self.args.blocklen, 128))
        deg = int(self.defset(self.args.degree, 3))
        tvsize_orig = long(self.defset(self.process_size(self.args.tvsize), 1024*256))
        zscore_thresh = float(self.args.conf)
        rounds = int(self.args.rounds) if self.args.rounds is not None else None
        top_k = int(self.args.topk) if self.args.topk is not None else None
        top_comb = int(self.defset(self.args.combdeg, 2))
        reffile = self.defset(self.args.reffile)
        all_deg = self.args.alldeg

        logger.info('Basic settings, deg: %s, blocklen: %s, TV size: %s, rounds: %s'
                    % (deg, blocklen, tvsize_orig, rounds))

        # Prebuffer map 3deg terms
        # logger.info('Precomputing term mappings')
        # term_map = common.build_term_map(deg, blocklen)

        # specific polynomial testing
        logger.info('Initialising')
        poly_test = self.get_testing_polynomials()
        poly_acc = [0] * len(poly_test)

        # test polynomials
        term_eval = common.TermEval(blocklen=blocklen, deg=deg)
        for idx, poly in enumerate(poly_test):
            print('Test polynomial: %02d, %s' % (idx, poly))
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
                logger.info('File size is smaller than TV, updating TV to %d' % size)
                tvsize = size

            hwanalysis = HWAnalysis()
            hwanalysis.deg = deg
            hwanalysis.blocklen = blocklen
            hwanalysis.top_comb = top_comb
            hwanalysis.top_k = top_k
            logger.info('Initializing test')
            hwanalysis.init()

            term_eval = common.TermEval(blocklen=blocklen, deg=deg)
            total_terms = long(scipy.misc.comb(blocklen, deg, True))
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

                    hwanalysis.proces_chunk(bits)


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

        parser.add_argument('--alldeg', dest='alldeg', action='store_const', const=True, default=False,
                            help='Evaluate all degree of polynomials to the threshold, e.g., 1,2,3 for deg 3')

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

