#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from past.builtins import basestring
from past.builtins import long

import collections
import heapq
import json
import time
import logging
import random
import re
import os
from functools import reduce

import argparse
import coloredlogs

from booltest import common
from booltest import misc
from booltest import timer
from booltest.common import Combined, CombinedIdx, ValueIdx
from booltest.jsonenc import NoIndent

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class HWAnalysis(object):
    """
    Analysis of all deg poly
    """
    def __init__(self, *args, **kwargs):
        self.term_map = []
        self.term_eval = None  # type: common.TermEval
        self.ref_term_eval = None  # type: common.TermEval

        self.blocklen = None
        self.deg = 3
        self.top_k = None
        self.comb_random = None
        self.top_comb = None
        self.zscore_thresh = None
        self.combine_all_deg = False
        self.no_comb_xor = False
        self.no_comb_and = False
        self.prob_comb = 1.0
        self.skip_print_res = False
        self.log_prints = False
        self.do_only_top_comb = False
        self.do_only_top_deg = False
        self.no_term_map = False
        self.use_zscore_heap = False
        self.sort_best_zscores = -1
        self.best_x_combinations = None  # if a number is here, best combinations are done by heap
        self.ref_db_path = None
        self.ref_samples = None
        self.ref_minmax = None
        self.comp_indiv_pval = False

        self.total_rounds = 0
        self.total_hws = []
        self.ref_total_hws = []
        self.total_n = 0
        self.last_res = None  # type: list[Combined]

        self.all_deg_compute = True
        self.input_poly = []
        self.input_poly_exp = []
        self.input_poly_hws = []
        self.input_poly_ref_hws = []
        self.input_poly_vars = set()
        self.input_poly_last_res = None

        self.all_zscore_comp = False  # all z-scores for degs, reference test
        self.all_zscore_list = None
        self.all_zscore_means = None

        # Buffers - allocated during computation for fast copy evaluation
        self.comb_res = None
        self.comb_subres = None

    def init(self):
        """
        Initializes state, term_eval engine, input polynomials expected probability.
        :return:
        """
        logger.debug('Initializing HWanalysis')

        if not self.no_term_map:
            logger.debug('Precomputing term mappings')
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

        self.try_read_db()

    def set_input_poly(self, poly):
        # compute classical analysis only if there are no input polynomials
        self.input_poly = poly if poly else []
        self.all_deg_compute = len(self.input_poly) == 0

    def try_read_db(self):
        if not self.ref_db_path or not os.path.exists(self.ref_db_path):
            return

        try:
            refjs = json.loads(open(self.ref_db_path, 'r').read())
            recs = [x for x in refjs if x['block'] == self.blocklen and x['deg'] == self.deg and x['comb_deg'] == self.top_comb]
            if len(recs) == 0:
                return

            recs = sorted(recs, key=lambda x: -x['nsamples'])
            self.ref_samples = recs[0]['nsamples']
            self.ref_minmax = (recs[0]['minv'], recs[0]['maxv'])
            logger.info('Pval db loaded, samples: %s, min: %s, max: %s' % (self.ref_samples, self.ref_minmax[0], self.ref_minmax[1]))

        except Exception as e:
            logger.error("Could not read the db", exc_info=e)

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
        self.input_poly_vars = set()
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

    def process_chunk(self, bits, ref_bits=None):
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
                obs_cnt = self.term_eval.eval_poly_hw(common.mutable_poly(poly), res=comb_res, subres=comb_subres)
                hws_input[idx] = obs_cnt
                self.input_poly_hws[idx] += obs_cnt

        self.total_n += self.term_eval.cur_evals

        # Reference stream
        ref_hws = self.process_ref(ref_bits, ln)

        # Done.
        r = self.analyse(num_evals=self.term_eval.cur_evals, hws=hws2, hws_input=hws_input, ref_hws=ref_hws)
        return r

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
        if self.log_prints:
            logger.info(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def unrank(self, deg, index):
        """
        Converts index to the polynomial of given degree.
        Uses either memorized table or unranking algorithm
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
            results[idx] = CombinedIdx(poly, expp, exp_cnt, obs_cnt, zscore, idx)

        # Sort by the zscore
        results.sort(key=lambda x: abs(x.zscore), reverse=True)

        if self.comp_indiv_pval:
            from scipy import stats

        for res in results:
            fail = 'x' if self.zscore_thresh and abs(res.zscore) > self.zscore_thresh else ' '
            indiv_pval = ' indiv-pval: %.5e,' % (stats.binom_test(res.obs_cnt, n=num_evals, p=res.expp, alternative='two-sided'),) if self.comp_indiv_pval else ''
            self.tprint(' - zscore[idx%02d]: %+05.5f, observed: %08d, expected: %08d %s idx: %6d,%s poly: %s'
                        % (res.idx, res.zscore, res.obs_cnt, res.exp_cnt, fail, res.idx, indiv_pval, self.input_poly[res.idx]))

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
        Uses heap to keep X best base distinguishers in the zscores array.
        Computes z-score for the best distinguishers as it is not computed by default for all
        as it is unnecessary overhead (HW is enough for ranking, zscore is floating point, expensive).

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
        hw_diff_threshold = self.zscore_thresh * zscore_denom * num_evals if self.zscore_thresh else None
        hw_diff_over = 0 if self.zscore_thresh else None

        # After this iteration hp will be a heap with sort_best_zscores elements
        hp = []
        hp_size = 0
        for (idx, hw) in enumerate(hws[deg]):
            hw_diff = abs(hw - exp_count[deg])
            hw_diff_sum += hw_diff
            if hw_diff_threshold and hw_diff >= hw_diff_threshold:
                hw_diff_over += 1

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
        fails = sum([1 for x in zscores[deg] if abs(x[0]) > self.zscore_thresh]) if self.zscore_thresh else None
        logger.info('Stats computed [%d], mean zscore: %s' % (deg, mean_zscore))
        return mean_zscore, fails

    def all_zscore_base_poly(self, deg, zscores, zscores_ref, num_evals, hws=None, ref_hws=None, exp_count=None):
        """
        Computes all zscores for all base polynomials
        :param deg:
        :param zscores:
        :return: (zscore mean, number of zscores above threshold)
        """
        logger.info('All zscores: %d ref: %s' % (deg, ref_hws is not None))
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

        mean_zscore = sum([abs(x[0]) for x in zscores[deg]]) / float(len(zscores[deg]))
        fails = sum([1 for x in zscores[deg] if abs(x[0]) > self.zscore_thresh]) if self.zscore_thresh else None
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
        r = self.analyse_input(num_evals=num_evals, hws_input=hws_input)

        # All degrees polynomials + combinations
        if not self.all_deg_compute:
            self.last_res = r
            return r

        probab = [self.term_eval.expp_term_deg(deg) for deg in range(0, self.deg + 1)]
        exp_count = [num_evals * x for x in probab]
        logger.info('Probabilities: %s, expected count: %s' % (probab, exp_count))

        top_terms = []
        mean_zscores = []
        zscores = [[0] * len(x) for x in hws]
        zscores_ref = [[0] * len(x) for x in hws]
        start_deg = self.deg if self.do_only_top_deg else 1
        for deg in range(start_deg, self.deg+1):
            # Reference computation - all zscore list
            # Used only for special theory check, not during normal computation
            if self.all_zscore_comp:
                mean_zscore, fails = self.all_zscore_base_poly(deg, zscores, zscores_ref, num_evals,
                                                               hws, ref_hws, exp_count)
                mean_zscores.append(mean_zscore)
                continue

            # Compute (zscore, idx)
            # Memory optimizations:
            #  1. for ranking avoid z-score computation - too expensive.
            #  2. add polynomials to the heap, keep there max 1-10k elements.
            mean_zscore, fails = self.best_zscored_base_poly(deg, zscores, zscores_ref, num_evals,
                                                             hws, ref_hws, exp_count)

            # Selecting TOP k polynomials for further combinations
            for idx, x in enumerate(zscores[deg][0:15]):
                fail = 'x' if self.zscore_thresh and abs(x[0]) > self.zscore_thresh else ' '
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

            self.tprint('Mean zscore[deg=%d]: %s' % (deg, mean_zscore))
            if fails:
                fails_fraction = float(fails) / len(zscores[deg])
                self.tprint('Num of fails[deg=%d]: %s = %02f.5%%' % (deg, fails, 100.0*fails_fraction))

        if self.all_zscore_comp:
            self.all_zscore_list = zscores
            self.all_zscore_means = mean_zscores
            return

        if self.top_k is None:
            return

        # Combine & store the results - XOR, AND combination
        top_res = []
        logger.info('Combining %d terms in %d degree, total = %s evals, keep best limit: %s'
                    % (len(top_terms), self.top_comb, common.comb(len(top_terms), self.top_comb, True),
                       self.best_x_combinations))

        self.comb_res = self.term_eval.new_buffer()
        self.comb_subres = self.term_eval.new_buffer()
        start_deg = max(1, self.top_comb if self.do_only_top_comb else 1)
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

        if self.comp_indiv_pval:
            from scipy import stats

        for i in range(min(len(top_res), 30)):
            comb = top_res[i]
            indiv_pval = ' indiv-pval: %.5e,' % (stats.binom_test(comb.obs_cnt, n=num_evals, p=comb.expp, alternative='two-sided'),) if self.comp_indiv_pval else ''
            self.tprint(' - best poly zscore %9.5f, expp: %.4e, exp: %7d, obs: %7d, diff: %10.7f %%,%s poly: %s'
                        % (comb.zscore, comb.expp, comb.exp_cnt, comb.obs_cnt,
                           100.0 * (comb.exp_cnt - comb.obs_cnt) / comb.exp_cnt, indiv_pval, sorted(comb.poly)))

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

    def comb_should_add(self, zscore, top_res):
        if self.best_x_combinations is None:
            return True
        if len(top_res) <= self.best_x_combinations:
            return True
        return zscore > top_res[0][0]

    def comb_add_result(self, comb, top_res):
        """
        Adds result to the top results.
        Can use heap to optimize eval speed if caller does not require all results.
        Should be called only if comb_should_add() returns true

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

        elif new_item[0] > top_res[0][0]:  # this difference is larger than minimum in heap
            heapq.heapreplace(top_res, new_item)

    def comb_base(self, top_comb_cur, top_terms, top_res, num_evals, poly_builder, ref_hws=None):
        """
        Base skeleton for generating all combinations from top_terms up to degree top_comb_cur.
        Evaluates polynomial, computes expected results, computes zscores.

        :param top_comb_cur: current degree of the combination
        :param top_terms: top terms buffer to choose terms out of
        :param top_res: top results accumulator to put, Combined(poly, expp, exp_cnt, obs_cnt, zscore - zscore_ref)
        :param num_evals: number of evaluations in this round - zscore computation
        :param poly_builder: function of (places, top_terms) returns a new polynomial
        :param ref_hws: reference results
        :return:
        """
        iterator = common.term_generator(top_comb_cur, len(top_terms) - 1, self.prob_comb)
        for idx, places in enumerate(iterator):
            poly = poly_builder(places, top_terms)
            expp = self.term_eval.expp_poly(poly)
            exp_cnt = num_evals * expp
            if exp_cnt == 0:
                continue

            obs_cnt = self.term_eval.eval_poly_hw(poly, self.comb_res, self.comb_subres)
            zscore = common.zscore(obs_cnt, exp_cnt, num_evals)
            if not self.comb_should_add(abs(zscore), top_res):
                continue

            comb = None
            if ref_hws is None:
                comb = Combined(poly, expp, exp_cnt, obs_cnt, zscore)

            else:
                ref_obs_cnt = self.ref_term_eval.eval_poly_hw(poly, res=self.comb_res, subres=self.comb_subres)
                zscore_ref = common.zscore(ref_obs_cnt, exp_cnt, num_evals)
                comb = Combined(poly, expp, exp_cnt, obs_cnt, zscore - zscore_ref)

            self.comb_add_result(comb, top_res)

    def comb_base_batch(self, top_comb_cur, top_terms, top_res, num_evals, poly_builder, ref_hws=None):
        """
        Base skeleton for generating all combinations from top_terms up to degree top_comb_cur.
        Evaluates polynomial, computes expected results, computes zscores.

        :param top_comb_cur: current degree of the combination
        :param top_terms: top terms buffer to choose terms out of
        :param top_res: top results accumulator to put, Combined(poly, expp, exp_cnt, obs_cnt, zscore - zscore_ref)
        :param num_evals: number of evaluations in this round - zscore computation
        :param poly_builder: function of (places, top_terms) returns a new polynomial
        :param ref_hws: reference results
        :return:
        """
        iterator = common.term_generator(top_comb_cur, len(top_terms) - 1, self.prob_comb, clone=True)
        btch = self.term_eval.base_buff_size
        polys = [None] * btch
        hws = [0] * btch

        for batch in common.chunks(iterator, btch):
            ln = len(batch)
            for ix, places in enumerate(batch):
                polys[ix] = poly_builder(places, top_terms)

            obs_cnts = self.term_eval.eval_polys_hw(polys[:ln], res=self.comb_res, subres=self.comb_subres, hws=hws)
            for ix in range(ln):
                poly = polys[ix]
                expp = self.term_eval.expp_poly(poly)
                exp_cnt = num_evals * expp
                if exp_cnt == 0:
                    continue

                obs_cnt = obs_cnts[ix]
                zscore = common.zscore(obs_cnt, exp_cnt, num_evals)
                if not self.comb_should_add(abs(zscore), top_res):
                    continue

                comb = None
                if ref_hws is None:
                    comb = Combined(poly, expp, exp_cnt, obs_cnt, zscore)

                else:
                    ref_obs_cnt = self.ref_term_eval.eval_poly_hw(poly, res=self.comb_res, subres=self.comb_subres)
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

    def eval_combs(self, polys, data):
        """
        Evaluates polynomials on the data, computes zscores
        :param polys:
        :param data:
        :return:
        """
        eval_res = self.term_eval.eval_polys_idx_data_strategy(polys, data)
        num_evals = len(eval_res[0])
        res = []

        for idx, poly in enumerate(polys):
            expp = self.term_eval.expp_poly(poly)
            exp_cnt = num_evals * expp
            obs_cnt = self.term_eval.hw(eval_res[idx])
            zscore = common.zscore(obs_cnt, exp_cnt, num_evals) if exp_cnt > 0 else None
            comb = Combined(poly, expp, exp_cnt, obs_cnt, zscore)
            res.append(comb)
        return res

    def to_json(self):
        """
        Serializes state to the json
        :return:
        """
        return dict(self.__dict__)

    def from_json(self, js):
        """
        Loads object config to the object
        :param js:
        :type js: dict
        :return:
        """
        for ckey in js.keys():
            if hasattr(self, ckey):
                setattr(self, ckey, js[ckey])

        self.input_poly_vars = set(self.input_poly_vars) if self.input_poly_vars else set()


# Main - argument parsing + processing
class Booltest(object):
    """
    Main booltest object
    """
    def __init__(self, *args, **kwargs):
        self.args = None
        self.tester = None
        self.blocklen = None
        self.deg = None
        self.top_comb = None
        self.top_k = 128
        self.all_deg = None
        self.zscore_thresh = None
        self.rounds = 0
        self.do_halving = False
        self.halving_top = 1
        self.json_top = 30
        self.json_nice = False
        self.input_poly = []
        self.input_objects = []
        self.dump_cpu_info = False
        self.alpha = 1./40000

        self.hwanalysis = None
        self.timer_data_read = timer.Timer(start=False)
        self.timer_data_bins = timer.Timer(start=False)
        self.timer_process = timer.Timer(start=False)

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
                ones[idx] += term_eval.eval_term_idx_single(term, val)
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

    def load_input_poly(self, poly=None, poly_files=None):
        """
        Loads input polynomials.
        :param poly:
        :param poly_files:
        :return:
        """
        for poly in (self.args.polynomials if not poly else poly):
            poly_js = self._fix_poly(json.loads(poly))
            self.input_poly.append(poly_js)

        for poly_file in (self.args.poly_file if not poly_files else poly_files):
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
            if file.startswith('http://') or file.startswith('https://') or file.startswith('ftp://'):
                io = common.LinkInputObject(file)
            else:
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

    def try_find_refdb(self):
        if self.args.ref_db:
            return self.args.ref_db

        fname = 'pval_db.json'
        pths = []
        try:
            pths.append(os.path.join(os.path.dirname(__file__), 'assets'))
        except:
            pass

        pths += os.getenv('PATH', '').split(os.pathsep)
        for p in pths:
            c = os.path.join(p, fname)
            try:
                if not os.path.exists(c):
                    continue
                js = json.load(open(c, 'r'))
                if js:
                    return c
            except:
                pass

        return None

    def init_params(self):
        self.blocklen = int(self.defset(self.args.blocklen, 256))

        # Default params
        if self.args.default_params:
            self.args.topk = 128
            self.args.no_comb_and = True
            self.args.only_top_comb = True
            self.args.only_top_deg = True
            self.args.no_term_map = True
            self.args.topterm_heap = True
            self.args.topterm_heap_k = 256

        self.deg = int(self.defset(self.args.degree, 2))
        self.top_k = int(self.args.topk) if self.args.topk is not None else None
        self.top_comb = int(self.defset(self.args.combdeg, 2))
        self.alpha = self.args.alpha
        self.all_deg = self.args.alldeg
        self.rounds = int(self.args.rounds) if self.args.rounds is not None else None
        self.do_halving = self.args.halving
        self.halving_top = self.args.halving_top
        self.json_top = self.args.json_top
        self.json_nice = self.args.json_nice

    def setup_hwanalysis(self, deg, top_comb, top_k, all_deg, zscore_thresh):
        hwanalysis = HWAnalysis()
        hwanalysis.deg = deg
        hwanalysis.blocklen = self.blocklen
        hwanalysis.top_comb = top_comb
        hwanalysis.comb_random = self.args.comb_random
        hwanalysis.top_k = top_k
        hwanalysis.combine_all_deg = all_deg
        hwanalysis.zscore_thresh = zscore_thresh
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
        hwanalysis.ref_db_path = self.try_find_refdb()
        hwanalysis.comp_indiv_pval = self.args.indiv_pval
        hwanalysis.log_prints = self.args.log_prints if (self.args and 'log_prints' in self.args) else False

        # compute classical analysis only if there are no input polynomials
        hwanalysis.all_deg_compute = len(self.input_poly) == 0
        return hwanalysis

    def build_poly_sort_index(self, poly):
        impoly = [common.immutable_poly(x) for x in poly]
        sorder = collections.defaultdict(lambda: 2 ** 40, list(zip(impoly, range(len(poly)))))
        return sorder

    def sort_res_by_input_poly(self, res, hwanalysis=None):
        hwanalysis = hwanalysis if hwanalysis else self.hwanalysis
        if not hwanalysis.input_poly:
            return res
        sorder = self.build_poly_sort_index(hwanalysis.input_poly)
        return sorted(res, key=lambda x: sorder[common.immutable_poly(x.poly)])

    def noindent(self, val):
        return NoIndent(val) if val and self.json_nice and not isinstance(val, NoIndent) else val

    def proc_offset(self, offset):
        if offset is not None and '.' in offset:
            offset = float(offset)
        elif offset is not None:
            offset = int(offset)
        else:
            offset = 0
        return offset

    def adjust_tvsize(self, tvsize, size, coffset=0):
        if tvsize is None:
            tvsize = size - coffset

        # size smaller than TV? Adapt tv then
        if size >= 0 and size < tvsize:
            logger.info('File size is smaller than TV, updating TV to %d' % size)
            tvsize = size - coffset

        if tvsize < 0:
            raise ValueError('Negative TV size: %s' % tvsize)

        coef = 8 if not self.do_halving else 4
        if (tvsize * coef) % self.blocklen != 0:
            rem = (tvsize * coef) % self.blocklen
            logger.warning('Input data size not aligned to the block size. '
                           'Input bytes: %d, block bits: %d, rem: %d' % (tvsize, self.blocklen, rem))
            tvsize -= rem // coef
            logger.info('Updating TV to %d' % tvsize)
        return int(tvsize)

    def human_poly(self, poly):
        p = []
        for term in poly:
            p.append(' * '.join(['x_{%d}' % int(x) for x in term]))
        if len(p) > 1:
            p = ['(%s)' % x for x in p]
        return ' + '.join(p)

    def print_human_readable_results(self, jsres):
        if self.args and 'no_summary' in self.args and self.args.no_summary:
            return
        
        try:
            print('-' * 80)
            print('BoolTest run with the configuration (%s, %s, %s)' % (jsres['blocklen'], jsres['degree'], jsres['comb_degree']))
            print(' - the block size is %s, i.e., the number of variables in the polynomials. x_{0}, ..., x_{%s}' % (jsres['blocklen'], jsres['blocklen'] - 1))
            print(' - the maximal order of the first level terms is %s. Combined by AND.' % (jsres['degree']))
            print(' - the maximal order of the second level polynomials is %s. Combined by XOR/AND.' % (jsres['comb_degree']))
            print('')
            for ix, inp in enumerate(jsres['inputs']):
                print('-' * 80)

                dist = inp['res'][0]['dists'][0]
                aux_dist = ' in the first half of the data' if self.do_halving else ''
                first_half_rejected = False
                print('[+] Results for input: %s' % inp['iobj'])
                print('The best distinguisher found%s: %s' % (aux_dist, self.human_poly(dist['poly'])))
                print(' - z-score achieved: %s' % abs(dist['zscore']))
                print('')

                if 'rejects' in inp['res'][0]:
                    alpha = 1./inp['res'][0]['ref_samples']
                    print('BoolTest has reference statistics for used configuration, number of collected samples: %s'
                          % inp['res'][0]['ref_samples'])
                    print('Allowed z-score region in the reference data: [%s, %s]' % inp['res'][0]['ref_minmax'])
                    if inp['res'][0]['rejects']:
                        first_half_rejected = True
                        print('As the achieved z-score %s lies outside of the interval, the randomness hypothesis '
                              'is REJECTED with alpha %.e' % (abs(dist['zscore']), alpha))
                        print(' = Data is not random')
                    else:
                        print('As the achieved z-score %s lies in the allowed interval, the randomness hypothesis '
                              'cannot be rejected with the alpha %.e' % (abs(dist['zscore']), alpha))
                        print(' = BoolTest could not find statistically significant non-randomness')

                else:
                    print('Unfortunately, for selected BoolTest parameters we don\'t have reference statistics '
                          'computed, thus we cannot say if the randomness hypothesis is rejected or not.')
                    if not self.do_halving:
                        print('You may try to run BoolTest with --halving parameter which runs a different evaluation '
                              'test without a need for reference data computation')

                    print('Expert tip: If you wish to analyze the input data with these parameters, '
                          'you need to collect reference statistics. For that, run BoolTest on a large sample '
                          'of random data (e.g., AES with random key and input), collect z-score statistics '
                          'and find maximum and minimum. If measured z-score lies outside of this interval, '
                          'you can reject the randomness hypothesis with alpha=1/num_of_reference_samples')

                if not self.do_halving:
                    continue

                hlv = inp['res'][1]['halvings'][0]
                print('\nHalving results (test on the another half of the input):')
                print('The selected distinguisher from the first BoolTest run: %s' % self.human_poly(hlv['poly']))
                print('The computed p-value on the input data: %.e' % hlv['pval'])
                if hlv['pval'] < self.alpha:
                    print('As the p-value is lower than alpha=%.e, the randomness hypothesis is REJECTED' % self.alpha)
                    print(' = Data is not random')

                else:
                    print('As the p-value >= alpha %.e, the randomness hypothesis cannot be rejected' % self.alpha)
                    print(' = BoolTest could not find statistically significant non-randomness')
                    if first_half_rejected:
                        print('However, the first phase of the BoolTest run on the first half of the input data '
                              'found a significant statistical deviation so this can be considered as a weak fail. '
                              'You may want to run the BoolTest without --halving to assess the whole data length.')

        except Exception as e:
            logger.error('Exception in processing the results: %s' % e, exc_info=e)

    def work(self):
        """
        Main entry point - data processing
        :return:
        """
        if self.args.debug:
            coloredlogs.install(level=logging.DEBUG)

        self.init_params()

        tvsize_orig = self.defset(self.process_size(self.args.tvsize), None)
        zscore_thresh = self.args.conf
        offset = self.proc_offset(self.args.offset)

        self.timer_data_read.reset()
        self.timer_data_bins.reset()
        self.timer_process.reset()
        cpu_pcnt_load_before = misc.try_get_cpu_percent()
        cpu_load_before = misc.try_get_cpu_load()

        # Load input polynomials
        self.load_input_poly()
        self.load_input_objects()

        logger.info('Basic settings, deg: %s, blocklen: %s, comb deg: %s, TV size: %s, rounds: %s'
                    % (self.deg, self.blocklen, self.top_comb, tvsize_orig, self.rounds))

        # specific polynomial testing
        logger.debug('Initialising')
        time_test_start = time.time()

        jscres = collections.OrderedDict()
        jsres_acc = [jscres]
        jsout = collections.OrderedDict([
            ('blocklen', self.blocklen),
            ('degree', self.deg),
            ('comb_degree', self.top_comb),
            ('top_k', self.top_k),
            ('input_poly', self.input_poly),
            ('offset', offset),
            ('halving', self.do_halving),
            ('inputs', jsres_acc)
        ])

        # read file by file
        for iobj in self.input_objects:
            tvsize = tvsize_orig
            coffset = offset

            iobj.check()
            size = iobj.size()
            logger.info('Testing input object: %s, size: %d kB' % (iobj, size/1024.0))
            jscres['iobj'] = str(iobj)
            jscres['size'] = size

            if isinstance(coffset, float):
                coffset = int(coffset * size)
            jscres['offset'] = coffset

            tvsize = self.adjust_tvsize(tvsize, size, coffset)

            self.hwanalysis = self.setup_hwanalysis(self.deg, self.top_comb, self.top_k, self.all_deg, zscore_thresh)
            if self.hwanalysis.ref_db_path:
                logger.info('Using reference data file %s' % self.hwanalysis.ref_db_path)

            logger.debug('Initializing test')
            self.hwanalysis.init()

            total_terms = int(common.comb(self.blocklen, self.deg, True))
            logger.info('BlockLength: %d, deg: %d, terms: %d' % (self.blocklen, self.deg, total_terms))

            jscres['sha1'] = ''
            jscres['sha256'] = ''
            jscres['tvsize'] = tvsize
            jscres['blocks'] = int((tvsize * 8) // self.blocklen)
            jscres['res'] = []

            # Read the file until there is no data.
            with iobj:
                self.analyze_iobj(iobj, coffset, tvsize, jscres['res'])

            for rr in jscres['res']:
                if 'dists' in rr:
                    rr['dists'] = [NoIndent(common.jsunwrap(x)) for x in rr['dists']]
                if 'halvings' in rr:
                    rr['halvings'] = [NoIndent(common.jsunwrap(x)) for x in rr['halvings']]

            logger.info('Finished processing: %s ' % iobj)
            logger.info('Data read: %s B' % iobj.data_read)
            logger.info('Read data SHA1:   %s ' % iobj.sha1.hexdigest())
            logger.info('Read data SHA256: %s ' % iobj.sha256.hexdigest())

            jscres['sha1'] = iobj.sha1.hexdigest()
            jscres['sha256'] = iobj.sha256.hexdigest()
            jscres = collections.OrderedDict()
            jsres_acc.append(jscres)

        jsres_acc.pop()  # remove the last empty record
        logger.info('Processing finished in %s sec' % (time.time() - time_test_start,))

        jsout['time_elapsed'] = time.time() - time_test_start
        jsout['time_data_read'] = self.timer_data_read.total()
        jsout['time_data_bins'] = self.timer_data_bins.total()
        jsout['time_process'] = self.timer_process.total()

        if self.dump_cpu_info:
            jsout['hostname'] = misc.try_get_hostname()
            jsout['cpu_pcnt_load_before'] = cpu_pcnt_load_before
            jsout['cpu_load_before'] = self.noindent(cpu_load_before)
            jsout['cpu_pcnt_load_after'] = misc.try_get_cpu_percent()
            jsout['cpu_load_after'] = self.noindent(misc.try_get_cpu_load())
            jsout['cpu'] = misc.try_get_cpu_info()
            if 'flags' in jsout['cpu']:
                jsout['cpu']['flags'] = self.noindent(jsout['cpu']['flags'])

        kwargs = {'indent': 2} if self.json_nice else {}
        if self.args.json_out:
            print(common.json_dumps(jsout, **kwargs))

        if self.args.json_out_file:
            with open(self.args.json_out_file, 'w+') as fh:
                common.json_dump(jsout, fh, **kwargs)

        jsout = common.jsunwrap(jsout)

        # Human-understandable results.
        self.print_human_readable_results(jsout)
        return jsout

    def analyze_iobj(self, iobj, coffset=0, tvsize=None, jscres=None):
        data_read = 0
        cur_round = 0

        prev_hwanalysis = None
        size = iobj.size()
        tvsize = tvsize if tvsize is not None else size
        jscres = jscres if jscres is not None else []

        if coffset > 0:
            with self.timer_data_read:
                iobj.read(coffset)
            size -= coffset

        if self.do_halving:
            tvsize = tvsize // 2

        while size < 0 or data_read < size:
            if self.rounds is not None and self.rounds > 0 and cur_round > self.rounds:
                break

            with self.timer_data_read:
                data = iobj.read(tvsize)

            data_read += len(data)
            if len(data) == 0:
                logger.info('Input <%s> read completely' % iobj)
                break

            if (len(data) * 8 % self.hwanalysis.blocklen) != 0:
                logger.info('Not aligned block read, terminating. Data left: %s bits, block size: %s bits'
                            % (len(data) * 8, self.hwanalysis.blocklen))
                break

            with self.timer_data_bins:
                bits = common.to_bitarray(data)

            logger.info('Analyzing input, deg: %d, blocklen: %4d, tvsize: %8d = %8.2f kB = %8.2f MB, '
                        'num-blocks: %d, round: %d, process: %d bits' %
                        (self.deg, self.blocklen, tvsize, tvsize / 1024.0, tvsize / 1024.0 / 1024.0,
                         (tvsize * 8) // self.blocklen, cur_round, len(bits)))

            prev_hwanalysis = None
            with self.timer_process:
                r = self.hwanalysis.process_chunk(bits)

            rshort = r[:min(len(r), self.json_top)]
            jsres = collections.OrderedDict([('round', cur_round)])
            jsres_dists = [common.comb2dict(x, self.json_nice) for x in rshort]
            jsres['dists'] = jsres_dists

            if self.hwanalysis.ref_samples and jsres_dists and (not self.do_halving or cur_round & 1 == 0):
                best_zsc = abs(jsres_dists[0]['zscore'])
                jsres['ref_samples'] = self.hwanalysis.ref_samples
                jsres['ref_alpha'] = 1. / self.hwanalysis.ref_samples
                jsres['ref_minmax'] = self.hwanalysis.ref_minmax
                jsres['rejects'] = best_zsc < self.hwanalysis.ref_minmax[0] or best_zsc > self.hwanalysis.ref_minmax[1]
                logger.info(
                    'Ref samples: %s, min-zscrore: %s, max-zscore: %s, best observed: %s, rejected: %s, alpha: %.5e'
                    % (self.hwanalysis.ref_samples, self.hwanalysis.ref_minmax[0], self.hwanalysis.ref_minmax[1],
                       best_zsc, jsres['rejects'], 1. / self.hwanalysis.ref_samples))

            if self.do_halving and cur_round & 1:
                from scipy import stats

                jsres['halvings'] = []
                for ix, cr in enumerate(self.sort_res_by_input_poly(rshort)):
                    ntrials = (tvsize * 8) // self.blocklen
                    pval = stats.binom_test(cr.obs_cnt, n=ntrials, p=cr.expp, alternative='two-sided')

                    jsresc = collections.OrderedDict()
                    jsresc['nsamples'] = ntrials
                    jsresc['nsucc'] = cr.obs_cnt
                    jsresc['pval'] = pval
                    jsresc['poly'] = self.noindent(common.immutable_poly(cr.poly))
                    jsres['halvings'].append(jsresc)

                    logger.info(
                        'Binomial dist [%d], two-sided pval: %.5e, poly pst: %.4e, ntrials: %s, succ: %s, poly: %s'
                        % (ix, pval, cr.expp, ntrials, cr.obs_cnt, cr.poly))

            jscres.append(jsres)
            cur_round += 1

            if self.do_halving:
                prev_hwanalysis = self.hwanalysis
                self.hwanalysis = self.setup_hwanalysis(self.deg, self.top_comb, self.top_k, self.all_deg, self.hwanalysis.zscore_thresh)
                if cur_round & 1:  # custom poly = best dist
                    selected_poly = [common.jsunwrap(jsres_dists[ix]['poly']) for ix in
                                     range(min(self.halving_top, len(jsres_dists)))]
                    logger.info('Halving, setting the best poly: %s' % selected_poly)
                    self.hwanalysis.set_input_poly(selected_poly)
                self.hwanalysis.init()

        # Revert last used object - one more init done in the halving which is not needed
        if self.do_halving and prev_hwanalysis:
            self.hwanalysis = prev_hwanalysis

        return jscres

    def main(self):
        logger.debug('App started')

        parser = self.argparser()
        self.args = parser.parse_args()
        self.work()

    def argparser(self):
        parser = argparse.ArgumentParser(description='BoolTest')

        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')

        parser.add_argument('--ref-db', dest='ref_db',
                            help='Reference JSON database file')

        parser.add_argument('--block', dest='blocklen', type=int, default=256,
                            help='block size in bits, number of bit variables to construct terms from')

        parser.add_argument('--degree', dest='degree', type=int, default=2,
                            help='maximum degree of terms to construct in the first phase (AND)')

        parser.add_argument('--tv', dest='tvsize',
                            help='Size of input data to process in bytes.'
                                 'Has to be aligned on block size')

        parser.add_argument('-r', '--rounds', dest='rounds', type=int, default=0,
                            help='Maximal number of test rounds, independent tests. '
                                 'One test round = one test vector length processing')

        parser.add_argument('--top', dest='topk', default=128, type=int,
                            help='top K number of the best distinguishers to select to the combination phase '
                                 '(second phase), combining terms with XOR')

        parser.add_argument('--comb-rand', dest='comb_random', default=0, type=int,
                            help='number of terms to add randomly to the combination set')

        parser.add_argument('--combine-deg', dest='combdeg', default=2, type=int,
                            help='Degree of combinations in the second phase (Combining terms by XOR). '
                                 'Number of terms to combine to one distinguisher.')

        parser.add_argument('--conf', dest='conf', type=float, default=None,
                            help='Zscore failing threshold. If set, BoolTest will compute failing stats. '
                                 'Takes only zscores higher than this value. Low limit is not supported. '
                                 'Setup if you know what are you doing. Computing failing zscore is hard due to'
                                 'dependencies and unknown z-score distribution. ')

        parser.add_argument('--alldeg', dest='alldeg', action='store_const', const=True, default=False,
                            help='Add top K best terms to the combination phase also for lower degree, '
                                 'not just the top one')

        parser.add_argument('--poly', dest='polynomials', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='input polynomial to evaluate on the input data instead of generated one')

        parser.add_argument('--poly-file', dest='poly_file', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='input file with polynomials to test, one polynomial per line, in json array notation')

        parser.add_argument('--poly-ignore', dest='poly_ignore', action='store_const', const=True, default=False,
                            help='Ignore input polynomial variables out of range')

        parser.add_argument('--poly-mod', dest='poly_mod', action='store_const', const=True, default=False,
                            help='Mod input polynomial variables out of range')

        parser.add_argument('--no-comb-xor', dest='no_comb_xor', action='store_const', const=True, default=False,
                            help='Disables XOR in the combination phase')

        parser.add_argument('--no-comb-and', dest='no_comb_and', action='store_const', const=True, default=False,
                            help='Disables AND in the combination phase')

        parser.add_argument('--only-top-comb', dest='only_top_comb', action='store_const', const=True, default=False,
                            help='If set, only the comb-degree combination is performed, '
                                 'otherwise all combinations up to given comb-degree')

        parser.add_argument('--only-top-deg', dest='only_top_deg', action='store_const', const=True, default=False,
                            help='If set, only the top degree of 1st stage polynomials are evaluated'
                                 '(zscore is computed), otherwise also lower degrees are input to the '
                                 'topk for next state - combinations')

        parser.add_argument('--no-term-map', dest='no_term_map', action='store_const', const=True, default=False,
                            help='Disables term map precomputation, uses unranking algorithm instead')

        parser.add_argument('--topterm-heap', dest='topterm_heap', action='store_const', const=True, default=False,
                            help='Use heap to compute best K terms for stats & input to the combinations')

        parser.add_argument('--topterm-heap-k', dest='topterm_heap_k', default=256, type=int,
                            help='Number of terms to keep in the heap, should be at least top_k')

        parser.add_argument('--best-x-combs', dest='best_x_combinations', default=None, type=int,
                            help='Number of best combinations to return. If defined, heap is used')

        parser.add_argument('--prob-comb', dest='prob_comb', type=float, default=1.0,
                            help='Probability the given combination is going to be chosen. '
                                 'Enables stochastic test, useful for large degrees.')

        parser.add_argument('--default-params', dest='default_params', action='store_const', const=True, default=False,
                            help='Default parameter settings for testing, used in the paper')

        parser.add_argument('files', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='files to process')

        parser.add_argument('--stdin', dest='stdin', action='store_const', const=True, default=False,
                            help='Read from the stdin')

        parser.add_argument('--stdin-desc', dest='stdin_desc', default=None,
                            help='Stdin descriptor')

        parser.add_argument('--json-out', dest='json_out', action='store_const', const=True, default=False,
                            help='Produce json result')

        parser.add_argument('--json-out-file', dest='json_out_file', default=None,
                            help='Produce json result to a file')

        parser.add_argument('--json-nice', dest='json_nice', action='store_const', const=True, default=False,
                            help='Nicely formatted json output')

        parser.add_argument('--json-top', dest='json_top', type=int, default=30,
                            help='Number of the best results to store to the output json')

        parser.add_argument('--offset', dest='offset',
                            help='Offset to start file reading')

        parser.add_argument('--halving', dest='halving', action='store_const', const=True, default=False,
                            help='Pick the best distinguisher on the first half, evaluate on the second half')

        parser.add_argument('--indiv-pval', dest='indiv_pval', action='store_const', const=True, default=False,
                            help='Compute individual pvalue with binomial test for each distinguisher. '
                                 'Warning! The computed pvals cannot be directly used as there are strong '
                                 'dependencies. Use only when you know what you are doing')

        parser.add_argument('--halving-top', dest='halving_top', type=int, default=1,
                            help='Number of top distinguishers to select to the halving phase')

        parser.add_argument('--alpha', dest='alpha', type=float, default=1e-4,
                            help='Alpha for pvalue hypothesis rejection, for halving method only')

        parser.add_argument('--no-summary', dest='no_summary', action='store_const', const=True, default=False,
                            help='Turns off the human readable summary')

        parser.add_argument('--log-prints', dest='log_prints', action='store_const', const=True, default=False,
                            help='Do not write info logs to stdout, log to stderr instead')

        return parser


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

