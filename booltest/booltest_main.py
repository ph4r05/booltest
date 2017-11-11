#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from past.builtins import basestring
from past.builtins import long

import collections
import heapq
import json
import logging
import random
import re
from functools import reduce

import argparse
import coloredlogs
import scipy
import scipy.misc
import scipy.stats

from booltest import common

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


Combined = collections.namedtuple('Combined', ['poly', 'expp', 'exp_cnt', 'obs_cnt', 'zscore'])
CombinedIdx = collections.namedtuple('CombinedIdx', ['poly', 'expp', 'exp_cnt', 'obs_cnt', 'zscore', 'idx'])
ValueIdx = collections.namedtuple('ValueIdx', ['value', 'idx'])


def bar_chart(sources=None, values=None, res=None, error=None, xlabel=None, title=None):
    import numpy as np
    import matplotlib.pyplot as plt
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
        self.ref_term_eval = None

        self.blocklen = None
        self.deg = 3
        self.top_k = None
        self.comb_random = None
        self.top_comb = None
        self.zscore_thresh = 1.96
        self.combine_all_deg = False
        self.do_ref = False
        self.no_comb_xor = False
        self.no_comb_and = False
        self.prob_comb = 1.0
        self.skip_print_res = False
        self.do_only_top_comb = False
        self.do_only_top_deg = False
        self.no_term_map = False
        self.use_zscore_heap = False
        self.sort_best_zscores = -1
        self.best_x_combinations = None  # if a number is here, best combinations are done by heap

        self.total_rounds = 0
        self.total_hws = []
        self.ref_total_hws = []
        self.total_n = 0
        self.last_res = None

        self.all_deg_compute = True
        self.input_poly = []
        self.input_poly_exp = []
        self.input_poly_hws = []
        self.input_poly_ref_hws = []
        self.input_poly_vars = set()
        self.input_poly_last_res = None

        # Buffers - allocated during computation for fast copy evaluation
        self.comb_res = None
        self.comb_subres = None

    def init(self):
        """
        Initializes state, term_eval engine, input polynomials expected probability.
        :return:
        """
        logger.info('Initializing HWanalysis')

        if not self.no_term_map:
            logger.info('Precomputing term mappings')
            self.term_map = common.build_term_map(self.deg, self.blocklen)

        self.term_eval = common.TermEval(blocklen=self.blocklen, deg=self.deg)
        self.ref_term_eval = common.TermEval(blocklen=self.blocklen, deg=self.deg)
        self.total_hws = [[0] * common.comb(self.blocklen, x, True) for x in range(self.deg + 1)]
        self.ref_total_hws = [[0] * common.comb(self.blocklen, x, True) for x in range(self.deg + 1)]
        self.input_poly_exp = [0] * len(self.input_poly)
        self.input_poly_hws = [0] * len(self.input_poly)
        self.input_poly_ref_hws = [0] * len(self.input_poly)
        self.precompute_input_poly()
        if self.best_x_combinations is not None and self.best_x_combinations <= 0:
            self.best_x_combinations = None

    def reset(self):
        """
        Reset internal stats - for use with new test vector set
        :return:
        """
        self.total_n = 0
        self.total_rounds = 0
        self.total_hws = [[0] * common.comb(self.blocklen, x, True) for x in range(self.deg + 1)]
        self.ref_total_hws = [[0] * common.comb(self.blocklen, x, True) for x in range(self.deg + 1)]
        self.input_poly_hws = [0] * len(self.input_poly)
        self.input_poly_ref_hws = [0] * len(self.input_poly)
        self.last_res = None
        self.input_poly_last_res = None

    def precompute_input_poly(self):
        """
        Precompute expected values for input polynomials
        :return:
        """
        self.input_poly_exp = []
        for poly in self.input_poly:
            exp_cnt = self.term_eval.expp_poly(poly)
            self.input_poly_exp.append(exp_cnt)

            # Used variables in input polynomials
            for term in poly:
                for var in term:
                    self.input_poly_vars.add(var)

    def proces_chunk(self, bits, ref_bits=None):
        """
        Processes input chunk of bits for analysis.
        :param bits:
        :param ref_bits:
        :return:
        """

        # Compute the basis.
        # Input polynomials optimization - evaluate basis only for variables used in polynomials.
        self.term_eval.load(bits, eval_only_vars=None if len(self.input_poly_vars) == 0 else self.input_poly_vars)
        ln = len(bits)
        hws2, hws_input = None, None

        # Evaluate all terms of degrees 1..deg
        if self.all_deg_compute:
            logger.info('Evaluating all terms, bitlen: %d, bytes: %d' % (ln, ln//8))
            hws2 = self.term_eval.eval_all_terms(self.deg)
            logger.info('Done: %s' % [len(x) for x in hws2])

            # Accumulate hws to the results.
            # If the first round, use the returned array directly to reduce time & memory for copying.
            if self.total_rounds == 0:
                self.total_hws = hws2
                logger.info('HWS merged - move')

            else:
                for d in range(1, self.deg+1):
                    for i in common.range2(len(self.total_hws[d])):
                        self.total_hws[d][i] += hws2[d][i]
                logger.info('HWS merged - merge')
            self.total_rounds += 1

        # Evaluate given input polynomials
        if len(self.input_poly) > 0:
            comb_res = self.term_eval.new_buffer()
            comb_subres = self.term_eval.new_buffer()
            hws_input = [0] * len(self.input_poly)
            for idx, poly in enumerate(self.input_poly):
                obs_cnt = self.term_eval.hw(self.term_eval.eval_poly(poly, res=comb_res, subres=comb_subres))
                hws_input[idx] = obs_cnt
                self.input_poly_hws[idx] += obs_cnt

        self.total_n += self.term_eval.cur_evals

        # Reference stream
        ref_hws = self.process_ref(ref_bits, ln)

        # Done.
        self.analyse(num_evals=self.term_eval.cur_evals, hws=hws2, hws_input=hws_input, ref_hws=ref_hws)

    def process_ref(self, ref_bits, ln):
        """
        Process reference data stream
        :return:
        """
        if ref_bits is None:
            return None

        if len(ref_bits) != ln:
            raise ValueError('Reference data stream has a different size')

        logger.info('Evaluating ref data stream')
        if self.all_deg_compute:
            self.ref_term_eval.load(ref_bits)
            ref_hws = self.ref_term_eval.eval_all_terms(self.deg)
            for d in range(1, self.deg+1):
                for i in common.range2(len(self.ref_total_hws[d])):
                    self.ref_total_hws[d][i] += ref_hws[d][i]
            return ref_hws

        else:
            return None

    def finished(self):
        """
        All data read - final analysis.
        :return:
        """
        return self.analyse(self.total_hws, self.total_n)

    def tprint(self, *args, **kwargs):
        if self.skip_print_res:
            return
        print(*args, **kwargs)

    def unrank(self, deg, index):
        """
        Converts index to the polynomial of given degree.
        Uses either memoryzed table or unranking algorithm
        :param deg:
        :param index:
        :return:
        """
        if self.no_term_map:
            return common.unrank(index, self.blocklen, deg)
        else:
            return self.term_map[deg][index]

    def analyse_input(self, num_evals, hws_input=None):
        """
        Analyses input polynomials result on the data
        :param num_evals:
        :param hws_input:
        :return:
        """
        if hws_input is None:
            return

        results = [None] * len(self.input_poly)
        for idx, poly in enumerate(self.input_poly):
            expp = self.input_poly_exp[idx]
            exp_cnt = num_evals * expp
            obs_cnt = hws_input[idx]
            zscore = common.zscore(obs_cnt, exp_cnt, num_evals)
            results[idx] = CombinedIdx(None, expp, exp_cnt, obs_cnt, zscore, idx)

        # Sort by the zscore
        results.sort(key=lambda x: abs(x.zscore), reverse=True)

        for res in results:
            fail = 'x' if abs(res.zscore) > self.zscore_thresh else ' '
            self.tprint(' - zscore[idx%02d]: %+05.5f, observed: %08d, expected: %08d %s idx: %6d, poly: %s'
                        % (res.idx, res.zscore, res.obs_cnt, res.exp_cnt, fail, res.idx, self.input_poly[res.idx]))

        self.input_poly_last_res = results
        return results

    def best_zscored_base_poly(self, deg, zscores, zscores_ref, num_evals, hws=None, ref_hws=None, exp_count=None):
        """
        Computes best X zscores
        :param deg:
        :param zscores:
        :return: (zscore mean, number of zscores above threshold)
        """
        if self.use_zscore_heap and deg > 1:
            return self.best_zscored_base_poly_heap(deg, zscores, zscores_ref, num_evals, hws, ref_hws, exp_count)
        else:
            return self.best_zscored_base_poly_all(deg, zscores, zscores_ref, num_evals, hws, ref_hws, exp_count)

    def best_zscored_base_poly_heap(self, deg, zscores, zscores_ref, num_evals, hws=None, ref_hws=None, exp_count=None):
        """
        Uses heap to keep X best base distinguishers in the zscores array
        :param deg:
        :param zscores:
        :return: (zscore mean, number of zscores above threshold)
        """
        logger.info('Find best with heap start deg: %d' % deg)
        zscore_denom = common.zscore_denominator(exp_count[deg], num_evals)
        if ref_hws is not None:
            raise ValueError('Heap optimization not allowed with ref stream')

        # zscore = hwdiff * 1/num_evals * 1/zscore_denom
        # zscore mean = \sum_{i=0}^{cnt} (hwdiff) * 1/num_evals * 1/zscore_denom / cnt
        hw_diff_sum = 0

        # threshold zscore = self.zscore_thresh,
        # threshold hw_diff = self.zscore_thresh * zscore_denom * num_evals
        hw_diff_threshold = self.zscore_thresh * zscore_denom * num_evals
        hw_diff_over = 0

        # After this iteration hp will be a heap with sort_best_zscores elements
        hp = []
        hp_size = 0
        for (idx, hw) in enumerate(hws[deg]):
            hw_diff = abs(hw - exp_count[deg])
            hw_diff_sum += hw_diff
            hw_diff_over += 1 if hw_diff >= hw_diff_threshold else 0

            if self.sort_best_zscores < 0 or hp_size <= self.sort_best_zscores:
                heapq.heappush(hp, (hw_diff, hw, idx))
                hp_size += 1

            elif hw_diff > hp[0][0]:   # this difference is larger than minimum in heap
                heapq.heapreplace(hp, (hw_diff, hw, idx))
        logger.info('Heap done: %d' % len(hp))

        # zscores[deg] space allocation
        top_range = min(len(hp), self.sort_best_zscores if self.sort_best_zscores >= 0 else len(hp))
        if len(zscores[deg]) < top_range:
            zscores[deg] = [0] * top_range

        # Take n largest from the heap, zscore.
        # Size of the queue ~ number of elements to sort, using sorted on the heap array is faster.
        hp.sort(reverse=True)
        logger.info('Heap sorted, len: %s' % top_range)

        for i in common.range2(top_range):
            hw_diff, hw, idx = hp[i]
            zscores[deg][i] = common.zscore_den(hw, exp_count[deg], num_evals, zscore_denom), idx, hw

        # stats
        total_n = float(len(hws[deg]))
        zscore_mean = hw_diff_sum / zscore_denom / num_evals / total_n
        logger.info('Stats done [%d], mean zscore: %s' % (deg, zscore_mean))
        return zscore_mean, hw_diff_over

    def best_zscored_base_poly_all(self, deg, zscores, zscores_ref, num_evals, hws=None, ref_hws=None, exp_count=None):
        """
        Computes all zscores
        :param deg:
        :param zscores:
        :return: (zscore mean, number of zscores above threshold)
        """
        logger.info('Find best with allsort start deg: %d' % deg)
        zscore_denom = common.zscore_denominator(exp_count[deg], num_evals)
        if ref_hws is not None:
            zscores_ref[deg] = [common.zscore_den(x, exp_count[deg], num_evals, zscore_denom)
                                for x in ref_hws[deg]]

            zscores[deg] = [((common.zscore_den(x, exp_count[deg], num_evals, zscore_denom)), idx, x)
                            for idx, x in enumerate(hws[deg])]  # - zscores_ref[deg][idx]

            zscores_ref[deg].sort(key=lambda x: abs(x), reverse=True)

        else:
            zscores[deg] = [(common.zscore_den(x, exp_count[deg], num_evals, zscore_denom), idx, x)
                            for idx, x in enumerate(hws[deg])]

        logger.info('Sorting...')
        zscores[deg].sort(key=lambda x: abs(x[0]), reverse=True)
        logger.info('Sorted... len: %d' % len(zscores[deg]))

        mean_zscore = sum([abs(x[0]) for x in zscores[deg]]) / float(len(zscores[deg]))
        fails = sum([1 for x in zscores[deg] if abs(x[0]) > self.zscore_thresh])
        logger.info('Stats computed [%d], mean zscore: %s' % (deg, mean_zscore))
        return mean_zscore, fails

    def analyse(self, num_evals, hws=None, hws_input=None, ref_hws=None):
        """
        Analyse hamming weights
        :param num_evals:
        :param hws: hamming weights on results for all degrees
        :param hws_input: hamming weights on results for input polynomials
        :param ref_hws: reference hamming weights
        :return:
        """

        # Input polynomials
        self.analyse_input(num_evals=num_evals, hws_input=hws_input)

        # All degrees polynomials + combinations
        if not self.all_deg_compute:
            return

        probab = [self.term_eval.expp_term_deg(deg) for deg in range(0, self.deg + 1)]
        exp_count = [num_evals * x for x in probab]
        logger.info('Probabilities: %s, expected count: %s' % (probab, exp_count))

        top_terms = []
        zscores = [[0] * len(x) for x in hws]
        zscores_ref = [[0] * len(x) for x in hws]
        start_deg = self.deg if self.do_only_top_deg else 1
        for deg in range(start_deg, self.deg+1):
            # Compute (zscore, idx)
            # Memory optimizations:
            #  1. for ranking avoid z-score computation - too expensive.
            #  2. add polynomials to the heap, keep there max 1-10k elements.
            mean_zscore, fails = self.best_zscored_base_poly(deg, zscores, zscores_ref, num_evals,
                                                             hws, ref_hws, exp_count)

            # Selecting TOP k polynomials for further combinations
            for idx, x in enumerate(zscores[deg][0:15]):
                fail = 'x' if abs(x[0]) > self.zscore_thresh else ' '
                self.tprint(' - zscore[deg=%d]: %+05.5f, %+05.5f, observed: %08d, expected: %08d %s idx: %6d, term: %s'
                            % (deg, x[0], zscores_ref[deg][idx]-x[0], x[2],
                               exp_count[deg], fail, x[1], self.unrank(deg, x[1])))

            # Take top X best polynomials
            if self.top_k is None:
                continue

            logger.info('Comb...')
            if self.combine_all_deg or deg == self.deg:
                top_terms += [self.unrank(deg, x[1]) for x in zscores[deg][0: (None if self.top_k < 0 else self.top_k)]]

                if self.comb_random > 0:
                    random_subset = random.sample(zscores[deg], self.comb_random)
                    top_terms += [self.unrank(deg, x[1]) for x in random_subset]

            logger.info('Stats...')
            fails_fraction = float(fails)/len(zscores[deg])

            self.tprint('Mean zscore[deg=%d]: %s' % (deg, mean_zscore))
            self.tprint('Num of fails[deg=%d]: %s = %02f.5%%' % (deg, fails, 100.0*fails_fraction))

        if self.top_k is None:
            return

        # Combine & store the results - XOR, AND combination
        top_res = []
        logger.info('Combining %d terms in %d degree, total = %s evals, keep best limit: %s'
                    % (len(top_terms), self.top_comb, scipy.misc.comb(len(top_terms), self.top_comb, True),
                       self.best_x_combinations))

        self.comb_res = self.term_eval.new_buffer()
        self.comb_subres = self.term_eval.new_buffer()
        start_deg = self.top_comb if self.do_only_top_comb else 1
        for top_comb_cur in common.range2(start_deg, self.top_comb + 1):

            # Combine * store results - XOR
            if not self.no_comb_xor:
                self.comb_xor(top_comb_cur=top_comb_cur, top_terms=top_terms, top_res=top_res, num_evals=num_evals,
                              ref_hws=ref_hws)

            # Combine & store results - AND
            if not self.no_comb_and:
                self.comb_and(top_comb_cur=top_comb_cur, top_terms=top_terms, top_res=top_res, num_evals=num_evals,
                              ref_hws=ref_hws)

        logger.info('Evaluating')
        top_res = self.sort_top_res(top_res)

        for i in range(min(len(top_res), 30)):
            comb = top_res[i]
            self.tprint(' - best poly zscore %9.5f, expp: %.4f, exp: %4d, obs: %s, diff: %f %%, poly: %s'
                        % (comb.zscore, comb.expp, comb.exp_cnt, comb.obs_cnt,
                           100.0 * (comb.exp_cnt - comb.obs_cnt) / comb.exp_cnt, sorted(comb.poly)))

        self.last_res = top_res
        return top_res

    def sort_top_res(self, top_res):
        """
        Sorts top_res. After this call it should be top_res = list(comb1, comb2, ...)
        :param top_res:
        :return:
        """
        if self.best_x_combinations is not None:  # de-heapify, project only the comb element.
            top_res = [x[1] for x in top_res]

        top_res.sort(key=lambda x: abs(x.zscore), reverse=True)
        return top_res

    def comb_add_result(self, comb, top_res):
        """
        Adds result to the top results.
        Can use heap to optimize eval speed if caller does not require all results.
        :param comb:
        :param top_res:
        :return:
        """
        if self.best_x_combinations is None:
            top_res.append(comb)
            return

        # Using heap to store only top self.best_x_combinations distinguishers here.
        # If comb contains pvalue in the future, compare better pval.
        new_item = (abs(comb.zscore), comb)
        if len(top_res) <= self.best_x_combinations:
            heapq.heappush(top_res, new_item)

        elif abs(comb.zscore) > top_res[0][0]:  # this difference is larger than minimum in heap
            heapq.heapreplace(top_res, new_item)

    def comb_base(self, top_comb_cur, top_terms, top_res, num_evals, poly_builder, ref_hws=None):
        """
        Base skeleton for generating all combinations from top_terms up to degree top_comb_cur
        :param top_comb_cur: current degree of the combination
        :param top_terms: top terms buffer to choose terms out of
        :param top_res: top results accumulator to put
        :param num_evals: number of evaluations in this round - zscore computation
        :param poly_builder: function of (places, top_terms) returns a new polynomial
        :param ref_hws: reference results
        :return:
        """
        for idx, places in enumerate(common.term_generator(top_comb_cur, len(top_terms) - 1, self.prob_comb)):
            poly = poly_builder(places, top_terms)
            expp = self.term_eval.expp_poly(poly)
            exp_cnt = num_evals * expp
            if exp_cnt == 0:
                continue

            obs_cnt = self.term_eval.hw(self.term_eval.eval_poly(poly, res=self.comb_res, subres=self.comb_subres))
            zscore = common.zscore(obs_cnt, exp_cnt, num_evals)

            comb = None
            if ref_hws is None:
                comb = Combined(poly, expp, exp_cnt, obs_cnt, zscore)
            else:
                ref_obs_cnt = self.ref_term_eval.hw(
                    self.ref_term_eval.eval_poly(poly, res=self.comb_res, subres=self.comb_subres))
                zscore_ref = common.zscore(ref_obs_cnt, exp_cnt, num_evals)
                comb = Combined(poly, expp, exp_cnt, obs_cnt, zscore - zscore_ref)

            self.comb_add_result(comb, top_res)

    def comb_xor(self, top_comb_cur, top_terms, top_res, num_evals, ref_hws=None):
        """
        Combines top terms with XOR operation
        :param top_comb_cur: current degree of the combination
        :param top_terms: top terms buffer to choose terms out of
        :param top_res: top results accumulator to put
        :param num_evals: number of evaluations in this round - zscore computation
        :param ref_hws: reference results
        :return:
        """
        poly_builder = lambda places, top_terms: [top_terms[x] for x in places]
        return self.comb_base(top_comb_cur, top_terms, top_res, num_evals, poly_builder, ref_hws)

    def comb_and(self, top_comb_cur, top_terms, top_res, num_evals, ref_hws=None):
        """
        Combines top terms with AND operation
        :param top_comb_cur: current degree of the combination
        :param top_terms: top terms buffer to choose terms out of
        :param top_res: top results accumulator to put
        :param num_evals: number of evaluations in this round - zscore computation
        :param ref_hws: reference results
        :return:
        """
        poly_builder = lambda places, top_terms: [reduce(lambda x, y: x + y, [top_terms[x] for x in places])]
        return self.comb_base(top_comb_cur, top_terms, top_res, num_evals, poly_builder, ref_hws)

    def comb_all(self):
        """
        TODO: implement
        :return:
        """


# Main - argument parsing + processing
class Booltest(object):
    """
    Main booltest object
    """
    def __init__(self, *args, **kwargs):
        self.args = None
        self.tester = None
        self.blocklen = None
        self.input_poly = []
        self.input_objects = []

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
            return 1024 if is_ib else 1000
        elif char == 'm':
            return 1024 * 1024 if is_ib else 1000 * 1000
        elif char == 'g':
            return 1024 * 1024 * 1024 if is_ib else 1000 * 1000 * 1000
        elif char == 't':
            return 1024 * 1024 * 1024 * 1024 if is_ib else 1000 * 1000 * 1000 * 1000
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
            return int(size_param)

        matches = re.match('^([0-9a-fA-F]+(.[0-9]+)?)([kKmMgGtT]([iI])?)?$', size_param)
        if matches is None:
            raise ValueError('Unknown size specifier')

        is_ib = matches.group(4) is not None
        mult_char = matches.group(3)
        multiplier = self.get_multiplier(mult_char, is_ib)
        return int(float(matches.group(1)) * multiplier)

    # noinspection PyMethodMayBeStatic
    def _fix_poly(self, poly):
        """
        Checks if the input polynomial is a valid polynomial
        :param poly:
        :return:
        """
        if not isinstance(poly, list):
            raise ValueError('Polynomial is not valid (list expected) %s' % poly)

        if len(poly) == 0:
            raise ValueError('Empty polynomial not allowed')

        first_elem = poly[0]
        if not isinstance(first_elem, list):
            poly = [poly]

        for idxt, term in enumerate(poly):
            if not isinstance(term, list):
                raise ValueError('Term %s in the polynomial %s is not valid (list expected)' % (term, poly))
            for idxv, var in enumerate(term):
                if not isinstance(var, (int, long)):
                    raise ValueError('Variable %s not valid in the polynomial %s (number expected)' % (var, poly))
                if var >= self.blocklen:
                    if self.args.poly_ignore:
                        return None
                    elif self.args.poly_mod:
                        poly[idxt][idxv] = var % self.blocklen
                    else:
                        raise ValueError('Variable %s not valid in the polynomial %s (blocklen is %d)'
                                         % (var, poly, self.blocklen))

        return poly

    def load_input_poly(self):
        """
        Loads input polynomials.
        :return:
        """
        for poly in self.args.polynomials:
            poly_js = self._fix_poly(json.loads(poly))
            self.input_poly.append(poly_js)

        for poly_file in self.args.poly_file:
            with open(poly_file, 'r') as fh:
                for line in fh:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    if line.startswith('#'):
                        continue
                    if line.startswith('//'):
                        continue
                    poly_js = self._fix_poly(json.loads(line))
                    if poly_js is None:
                        continue
                    self.input_poly.append(poly_js)

                logger.debug('Poly file %s loaded' % poly_file)

        logger.debug('Input polynomials length: %s' % len(self.input_poly))

    def load_input_objects(self):
        """
        Loads input objects to an array
        :return:
        """
        for file in self.args.files:
            io = common.FileInputObject(fname=file, fmode='rb')
            io.check()
            self.input_objects.append(io)

        if len(self.input_objects) == 0 or self.args.stdin:
            self.input_objects.append(common.StdinInputObject(desc=self.args.stdin_desc))

    def test_polynomials_exp_values(self):
        """
        Simple testing routine to validate expected polynomial results on static polynomials
        :return:
        """
        poly_test = self.get_testing_polynomials()
        poly_acc = [0] * len(poly_test)

        # test polynomials
        term_eval = common.TermEval(blocklen=self.blocklen, deg=3)
        for idx, poly in enumerate(poly_test):
            print('Test polynomial: %02d, %s' % (idx, poly))
            expp = term_eval.expp_poly(poly)
            print('  Expected probability: %s' % expp)

    def work(self):
        """
        Main entry point - data processing
        :return:
        """
        self.blocklen = int(self.defset(self.args.blocklen, 128))
        deg = int(self.defset(self.args.degree, 3))
        tvsize_orig = int(self.defset(self.process_size(self.args.tvsize), 1024*256))
        zscore_thresh = float(self.args.conf)
        rounds = int(self.args.rounds) if self.args.rounds is not None else None
        top_k = int(self.args.topk) if self.args.topk is not None else None
        top_comb = int(self.defset(self.args.combdeg, 2))
        reffile = self.defset(self.args.reffile)
        all_deg = self.args.alldeg

        # Load input polynomials
        self.load_input_poly()
        self.load_input_objects()

        logger.info('Basic settings, deg: %s, blocklen: %s, TV size: %s, rounds: %s'
                    % (deg, self.blocklen, tvsize_orig, rounds))

        # specific polynomial testing
        logger.info('Initialising')

        # read file by file
        for iobj in self.input_objects:
            tvsize = tvsize_orig

            iobj.check()
            size = iobj.size()
            logger.info('Testing input object: %s, size: %d kB' % (iobj, size/1024.0))

            # size smaller than TV? Adapt tv then
            if size >= 0 and size < tvsize:
                logger.info('File size is smaller than TV, updating TV to %d' % size)
                tvsize = size

            if tvsize*8 % self.blocklen != 0:
                rem = tvsize*8 % self.blocklen
                logger.warning('Input data size not aligned to the block size. '
                               'Input bytes: %d, block bits: %d, rem: %d' % (tvsize, self.blocklen, rem))
                tvsize -= rem//8
                logger.info('Updating TV to %d' % tvsize)

            hwanalysis = HWAnalysis()
            hwanalysis.deg = deg
            hwanalysis.blocklen = self.blocklen
            hwanalysis.top_comb = top_comb
            hwanalysis.comb_random = self.args.comb_random
            hwanalysis.top_k = top_k
            hwanalysis.combine_all_deg = all_deg
            hwanalysis.zscore_thresh = zscore_thresh
            hwanalysis.do_ref = reffile is not None
            hwanalysis.input_poly = self.input_poly
            hwanalysis.no_comb_and = self.args.no_comb_and
            hwanalysis.no_comb_xor = self.args.no_comb_xor
            hwanalysis.prob_comb = self.args.prob_comb
            hwanalysis.do_only_top_comb = self.args.only_top_comb
            hwanalysis.do_only_top_deg = self.args.only_top_deg
            hwanalysis.no_term_map = self.args.no_term_map
            hwanalysis.use_zscore_heap = self.args.topterm_heap
            hwanalysis.sort_best_zscores = max(common.replace_none([self.args.topterm_heap_k, top_k, 100]))
            hwanalysis.best_x_combinations = self.args.best_x_combinations

            # compute classical analysis only if there are no input polynomials
            hwanalysis.all_deg_compute = len(self.input_poly) == 0
            logger.info('Initializing test')
            hwanalysis.init()

            total_terms = int(scipy.misc.comb(self.blocklen, deg, True))
            logger.info('BlockLength: %d, deg: %d, terms: %d' % (self.blocklen, deg, total_terms))

            # Reference data stream reading
            # Read the file until there is no data.
            fref = None
            if reffile is not None:
                fref = open(reffile, 'r')

            with iobj:
                data_read = 0
                cur_round = 0

                while size < 0 or data_read < size:
                    if rounds is not None and cur_round > rounds:
                        break

                    data = iobj.read(tvsize)
                    bits = common.to_bitarray(data)
                    if len(bits) == 0:
                        logger.info('File read completely')
                        break

                    ref_bits = None
                    if fref is not None:
                        ref_data = fref.read(tvsize)
                        ref_bits = common.to_bitarray(ref_data)

                    logger.info('Pre-computing with TV, deg: %d, blocklen: %04d, tvsize: %08d = %8.2f kB = %8.2f MB, '
                                'round: %d, avail: %d' %
                                (deg, self.blocklen, tvsize, tvsize/1024.0, tvsize/1024.0/1024.0, cur_round, len(bits)))

                    hwanalysis.proces_chunk(bits, ref_bits)
                    cur_round += 1
                pass

            logger.info('Finished processing %s ' % iobj)
            logger.info('Data read %s ' % iobj.data_read)
            logger.info('Read data hash %s ' % iobj.sha1.hexdigest())

            if fref is not None:
                fref.close()
        logger.info('Processing finished')

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
                            help='Size of one test vector, in this interpretation = number of bytes to read from file. '
                                 'Has to be aligned on block size')
        parser.add_argument('-r', '--rounds', dest='rounds',
                            help='Maximal number of rounds')

        parser.add_argument('--top', dest='topk', default=30, type=int,
                            help='top K number of best distinguishers to combine together')

        parser.add_argument('--comb-rand', dest='comb_random', default=0, type=int,
                            help='number of terms to add randomly to the combination set')

        parser.add_argument('--combine-deg', dest='combdeg', default=2, type=int,
                            help='Degree of combination')

        parser.add_argument('--conf', dest='conf', type=float, default=1.96,
                            help='Zscore failing threshold')

        parser.add_argument('--alldeg', dest='alldeg', action='store_const', const=True, default=False,
                            help='Add top K best terms to the combination group also for lower degree, not just top one')

        parser.add_argument('--poly', dest='polynomials', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='input polynomial to evaluate on the input data instead of generated one')

        parser.add_argument('--poly-file', dest='poly_file', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='input file with polynomials to test, one polynomial per line, in json array notation')

        parser.add_argument('--poly-ignore', dest='poly_ignore', action='store_const', const=True, default=False,
                            help='Ignore input polynomial variables out of range')

        parser.add_argument('--poly-mod', dest='poly_mod', action='store_const', const=True, default=False,
                            help='Mod input polynomial variables out of range')

        parser.add_argument('--no-comb-xor', dest='no_comb_xor', action='store_const', const=True, default=False,
                            help='Disables XOR combinations')

        parser.add_argument('--no-comb-and', dest='no_comb_and', action='store_const', const=True, default=False,
                            help='Disables AND combinations')

        parser.add_argument('--only-top-comb', dest='only_top_comb', action='store_const', const=True, default=False,
                            help='If set only the top combination is performed, otherwise all up to given combination degree')

        parser.add_argument('--only-top-deg', dest='only_top_deg', action='store_const', const=True, default=False,
                            help='If set only the top degree if base polynomials combinations are considered, otherwise '
                                 'also lower degrees are input to the topk for next state - combinations')

        parser.add_argument('--no-term-map', dest='no_term_map', action='store_const', const=True, default=False,
                            help='Disables term map precomputation, uses unranking algorithm instead')

        parser.add_argument('--topterm-heap', dest='topterm_heap', action='store_const', const=True, default=False,
                            help='Use heap to compute best X terms for stats & input to the combinations')

        parser.add_argument('--topterm-heap-k', dest='topterm_heap_k', default=None, type=int,
                            help='Number of terms to keep in the heap')

        parser.add_argument('--best-x-combs', dest='best_x_combinations', default=None, type=int,
                            help='Number of best combinations to return. If defined, heap is used')

        parser.add_argument('--prob-comb', dest='prob_comb', type=float, default=1.0,
                            help='Probability the given combination is going to be chosen.')

        parser.add_argument('files', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='files to process')

        parser.add_argument('--stdin', dest='stdin', action='store_const', const=True, default=False,
                            help='Read from the stdin')

        parser.add_argument('--stdin-desc', dest='stdin_desc', default=None,
                            help='Stdin descriptor')

        self.args = parser.parse_args()
        self.work()


# Launcher
app = None


def main():
    """
    Main booltest wrapper
    :return:
    """
    app = Booltest()
    app.main()


if __name__ == '__main__':
    main()

