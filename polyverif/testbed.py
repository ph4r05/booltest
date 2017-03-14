#!/usr/bin/env python
# -*- coding: utf-8 -*-

from past.builtins import basestring
from functools import reduce
import argparse
import logging
import coloredlogs
import common
import os
import re
import six
import sys
import math
import time
import random
import json
import types
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.misc
import scipy.stats
import subprocess
import signal
import psutil
import shutil
from main import *
import egenerator


logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


# Main - argument parsing + processing
class TestbedBenchmark(App):
    def __init__(self, *args, **kwargs):
        super(TestbedBenchmark, self).__init__(*args, **kwargs)
        self.args = None
        self.tester = None
        self.input_poly = []

        self.results_dir = None
        self.generator_path = None
        self.test_stride = None
        self.test_manuals = None
        self.top_k = 128
        self.zscore_thresh = None
        self.all_deg = None

        self.config_js = None

    def init_params(self):
        """
        Parameter processing
        :return:
        """
        # Results dir
        self.results_dir = self.args.results_dir
        if self.results_dir is None:
            logger.warning('Results dir is not defined, using current directory')
            self.results_dir = os.getcwd()

        elif not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # Generator path
        self.generator_path = self.args.generator_path
        if self.generator_path is None:
            logger.warning('Generator path is not given, using current directory')
            self.generator_path = os.path.join(os.getcwd(), 'generator')

        if not os.path.exists(self.generator_path):
            raise ValueError('Generator not found: %s' % self.generator_path)

        # Stride
        self.test_stride = self.args.tests_stride
        self.test_manuals = self.args.tests_manuals

        # Other
        self.zscore_thresh = self.args.conf

        deg = int(self.defset(self.args.degree, 3))
        tvsize_orig = int(self.defset(self.process_size(self.args.tvsize), 1024 * 256))
        top_k = int(self.args.topk) if self.args.topk is not None else None
        top_comb = int(self.defset(self.args.combdeg, 2))
        self.all_deg = self.args.alldeg
        tvsize = tvsize_orig

    def gen_randomdir(self, function, round):
        """
        Generates random directory name
        :return:
        """
        dirname = 'testbed-%s-r%s-%d-%d' % (function, round, int(time.time()), random.randint(0, 2**32-1))
        return os.path.join('/tmp', dirname)

    def test_case_generator(self, test_sizes_mb, test_block_sizes, test_degree, test_comb_k):
        """
        Generator for test cases
        :param functions:
        :param test_sizes_mb:
        :param test_block_sizes:
        :param test_degree:
        :param test_comb_k:
        :return: test_idx, data_size, block_size, degree, comb_k
        """
        test_idx = 0

        # Several data sizes
        for data_size in test_sizes_mb:

            # parameters iteration, block, deg, k.
            for block_size in test_block_sizes:

                for degree in test_degree:

                    for comb_k in test_comb_k:

                        yield test_idx, data_size, block_size, degree, comb_k
                        test_idx += 1

    # noinspection PyBroadException
    def work(self):
        """
        Main entry point - data processing
        :return:
        """
        self.init_params()

        # Init logic, analysis.
        # Define test set.
        test_sizes_mb = [1, 10, 100]
        test_block_sizes = [128, 256, 384, 512]
        test_degree = [1, 2]
        test_comb_k = [1, 2]
        results_acc = []

        # Test all functions
        functions = sorted(list(egenerator.ROUNDS.keys()))
        for function in functions:
            rounds = egenerator.ROUNDS[function]

            # Generate random tmpdir, generate data, test it there...
            for cur_round in rounds:
                tmpdir = self.gen_randomdir(function, cur_round)
                new_gen_path = os.path.join(tmpdir, 'generator')
                data_to_gen = max(test_sizes_mb) * 1024 * 1024

                # Copy generator executable here, generate data.
                os.makedirs(tmpdir)
                shutil.copy(self.generator_path, new_gen_path)

                self.config_js = egenerator.get_config(function_name=function, rounds=cur_round, data=data_to_gen)
                config_str = json.dumps(self.config_js, indent=2)
                with open(os.path.join(tmpdir, 'generator.json'), 'w') as fh:
                    fh.write(config_str)

                # Generate some data here
                logger.info('Generating data for %s, round %s to %s' % (function, cur_round, tmpdir))
                p = subprocess.Popen(new_gen_path, shell=True, cwd=tmpdir)
                p.communicate()
                if p.returncode != 0:
                    logger.error('Could not generate data, code: %s' % p.returncode)
                    continue

                # Generated file:
                data_files = [f for f in os.listdir(tmpdir) if os.path.isfile(os.path.join(tmpdir, f))
                                                                              and f.endswith('bin')]
                if len(data_files) != 1:
                    logger.error('Error in generating data to process. Files found: %s' % data_files)
                    continue

                data_file = os.path.join(tmpdir, data_files[0])
                logger.info('Data file generated to: %s' % data_file)

                # Generate test cases, run the analysis.
                for test_case in self.test_case_generator(test_sizes_mb, test_block_sizes, test_degree, test_comb_k):
                    test_idx, data_size, block_size, degree, comb_deg = test_case
                    test_desc = 'idx: %04d, data: %04d, block: %d, deg: %d, comb-deg: %d' \
                                % (test_idx, data_size, block_size, degree, comb_deg)

                    if self.test_manuals > 1 and (test_idx % self.test_manuals) != self.test_stride:
                        logger.info('Skipping test %s' % test_desc)
                        continue

                    logger.info('Working on test: %s' % test_desc)
                    jsres = self.testcase(function, cur_round, data_size, block_size, degree, comb_deg,
                                          data_file, tmpdir)

                    res_file = '%s-r%s-seed%s-%sMB-%sbl-%sdeg-%sk.json' \
                               % (function, cur_round, self.config_js['seed'], data_size, block_size, degree,
                                  comb_deg)

                    res_file_path = os.path.join(self.results_dir, res_file)
                    with open(res_file_path, 'w') as fh:
                        fh.write(json.dumps(jsres, indent=2))

                # Remove test dir
                shutil.rmtree(tmpdir)

    def testcase(self, function, cur_round, size_mb, blocklen, degree, comb_deg, data_file, tmpdir):
        """
        Test case executor
        :param function:
        :param cur_round:
        :param size_mb:
        :param blocklen:
        :param degree:
        :param comb_deg:
        :param data_file:
        :return:
        """
        rounds = 0
        tvsize = 1024 * 1024 * size_mb

        # Load input polynomials
        self.load_input_poly()
        script_path = common.get_script_path()

        logger.info('Basic settings, deg: %s, blocklen: %s, TV size: %s' % (degree, blocklen, tvsize))

        total_terms = int(scipy.misc.comb(blocklen, degree, True))
        hwanalysis = HWAnalysis()
        hwanalysis.deg = degree
        hwanalysis.blocklen = blocklen
        hwanalysis.top_comb = comb_deg

        hwanalysis.comb_random = self.args.comb_random
        hwanalysis.top_k = self.top_k
        hwanalysis.combine_all_deg = self.all_deg
        hwanalysis.zscore_thresh = self.zscore_thresh
        hwanalysis.do_ref = None
        hwanalysis.skip_print_res = True
        hwanalysis.input_poly = self.input_poly
        hwanalysis.no_comb_and = self.args.no_comb_and
        hwanalysis.no_comb_xor = self.args.no_comb_xor
        hwanalysis.prob_comb = self.args.prob_comb
        hwanalysis.all_deg_compute = len(self.input_poly) == 0
        hwanalysis.do_only_top_comb = self.args.only_top_comb
        hwanalysis.do_only_top_deg = self.args.only_top_deg
        hwanalysis.no_term_map = self.args.no_term_map
        hwanalysis.use_zscore_heap = self.args.topterm_heap
        hwanalysis.sort_best_zscores = max(self.args.topterm_heap_k, self.top_k, 100)
        logger.info('Initializing test')
        hwanalysis.init()

        # Process input object
        iobj = common.FileInputObject(data_file)
        size = iobj.size()
        logger.info('Testing input object: %s, size: %d kB' % (iobj, size/1024.0))

        # size smaller than TV? Adapt tv then
        if size >= 0 and size < tvsize:
            logger.info('File size is smaller than TV, updating TV to %d' % size)
            tvsize = size

        if tvsize*8 % blocklen != 0:
            rem = tvsize*8 % blocklen
            logger.warning('Input data size not aligned to the block size. '
                           'Input bytes: %d, block bits: %d, rem: %d' % (tvsize, blocklen, rem))
            tvsize -= rem//8
            logger.info('Updating TV to %d' % tvsize)

        hwanalysis.reset()
        logger.info('BlockLength: %d, deg: %d, terms: %d' % (blocklen, degree, total_terms))
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

                logger.info('Pre-computing with TV, deg: %d, blocklen: %04d, tvsize: %08d = %8.2f kB = %8.2f MB, '
                            'round: %d, avail: %d' %
                            (degree, blocklen, tvsize, tvsize/1024.0, tvsize/1024.0/1024.0, cur_round, len(bits)))

                hwanalysis.proces_chunk(bits, None)
                cur_round += 1
            pass

        # RESULT process...
        total_results = len(hwanalysis.last_res)
        best_dists = hwanalysis.last_res[0 : min(128, total_results)]
        data_hash = iobj.sha1.hexdigest()

        jsres = collections.OrderedDict()
        jsres['best_zscore'] = best_dists[0].zscore
        jsres['best_poly'] = best_dists[0].poly

        jsres['blocklen'] = blocklen
        jsres['degree'] = degree
        jsres['comb_degree'] = comb_deg
        jsres['top_k'] = self.top_k
        jsres['all_deg'] = self.all_deg

        jsres['data_hash'] = data_hash
        jsres['data_read'] = iobj.data_read
        jsres['generator'] = self.config_js
        jsres['best_dists'] = best_dists

        logger.info('Finished processing %s ' % iobj)
        logger.info('Data read %s ' % iobj.data_read)
        logger.info('Read data hash %s ' % data_hash)
        return jsres

    def main(self):
        logger.debug('App started')

        parser = argparse.ArgumentParser(description='PolyDist - Testbed testing')
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

        parser.add_argument('--prob-comb', dest='prob_comb', type=float, default=1.0,
                            help='Probability the given combination is going to be chosen.')

        parser.add_argument('--topterm-heap', dest='topterm_heap', action='store_const', const=True, default=False,
                            help='Use heap to compute best X terms for stats & input to the combinations')

        parser.add_argument('--topterm-heap-k', dest='topterm_heap_k', default=None, type=int,
                            help='Number of terms to keep in the heap')

        parser.add_argument('--csv-zscore', dest='csv_zscore', action='store_const', const=True, default=False,
                            help='CSV output with zscores')

        #
        # Testbed related options
        #

        parser.add_argument('--generator-path', dest='generator_path', default=None,
                            help='Path to the generator executable')

        parser.add_argument('--result-dir', dest='results_dir', default=None,
                            help='Directory to put results to')

        parser.add_argument('--tests-manuals', dest='tests_manuals', default=1, type=int,
                            help='Number of manually started workers for this computation')

        parser.add_argument('--tests-stride', dest='tests_stride', default=1, type=int,
                            help='Tests stride, skipping tests')

        self.args = parser.parse_args()
        self.work()


# Launcher
app = None
if __name__ == "__main__":
    app = TestbedBenchmark()
    app.main()

