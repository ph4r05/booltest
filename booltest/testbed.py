#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import os
import shutil
import subprocess
import time
import traceback

import scipy.misc
import scipy.stats

from . import egenerator
from . import common
from .booltest_main import *


logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


# Main - argument parsing + processing
class TestbedBenchmark(Booltest):
    """
    Testbed run matrix of tests on multiple possible functions.

    Used in conjunction with EACirc generator. A function \in generator is used with
    different number of rounds to benchmark success rate of the polynomial approach.

    Tests scenarios are generated in a way parallelization is possible.
    """
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

        self.randomize_tests = True
        self.test_random = random.Random()
        self.test_random.seed(0)

        self.seed = '1fe40505e131963c'
        self.data_to_gen = 0
        self.config_js = None
        self.cur_data_file = None  # (tmpdir, config, file)

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

        self.all_deg = self.args.alldeg
        self.test_random.seed(self.args.tests_random_select_eed)

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
        :return: data_size, block_size, degree, comb_k
        """
        cases = [list(test_sizes_mb), list(test_block_sizes), list(test_degree), list(test_comb_k)]
        iterator = itertools.product(*cases)

        data = [x for x in iterator]
        self.test_random.shuffle(data)
        return data

    def check_res_file(self, path):
        """
        Returns True if the given result path seems valid
        :param path:
        :return:
        """
        if not os.path.exists(path):
            return False

        # noinspection PyBroadException
        try:
            with open(path, 'r') as fh:
                js = json.load(fh)
                return 'best_zscore' in js and 'best_dists' in js
        except:
            return False

    def find_data_file(self, function, round):
        """
        Tries to find a data file
        :param function:
        :param round:
        :return:
        """
        data_dir = self.args.data_dir

        if data_dir is None or not os.path.exists(data_dir):
            logger.info('Data dir empty %s' % data_dir)
            return None

        candidates = [
            '%s_r%s_seed%s_%sMB.bin' % (function, round, self.seed, self.data_to_gen//1024//1024),
            '%s_r%s_seed%s.bin' % (function, round, self.seed),
            '%s_r%s_b8.bin' % (function, round),
            '%s_r%s_b16.bin' % (function, round),
            '%s_r%s_b32.bin' % (function, round),
            '%s_r%s.bin' % (function, round),
            '%s.bin' % function
        ]

        for cand in candidates:
            fpath = os.path.join(data_dir, cand)

            if not os.path.exists(fpath):
                continue

            if os.path.getsize(fpath) < self.data_to_gen:
                logger.info('File %s exists but is to small' % fpath)
                continue

            return fpath
        return None

    def data_generator(self, tmpdir, function, cur_round):
        """
        Used to call generator to generate data to test. Prepares data to test.
        If the file has already been generated, just returns the generated file.
        :return:
        """
        if self.cur_data_file is not None \
                and self.cur_data_file[0] == tmpdir \
                and self.cur_data_file[1] == self.config_js:
            logger.info('Using already generated data file: %s' % self.cur_data_file[2])
            return self.cur_data_file[2]

        data_file = self.find_data_file(function=function, round=cur_round)
        if data_file is not None:
            return data_file

        # Egenerator procedure: new temp folder, generate config, generate data.
        logger.info('Generating data for %s, round %s to %s' % (function, cur_round, tmpdir))
        data_file = self.eacirc_generator(tmpdir=tmpdir, generator_path=self.generator_path, config_js=self.config_js)

        if data_file is not None:
            logger.info('Data file generated to: %s' % data_file)
            self.cur_data_file = tmpdir, self.config_js, data_file

        return data_file

    def eacirc_generator(self, tmpdir, generator_path, config_js):
        """
        Uses Egenerator to produce the file
        :param tmpdir:
        :param generator_path:
        :param config_js:
        :return:
        """
        os.makedirs(tmpdir)

        new_generator_path = os.path.join(tmpdir, 'generator')
        shutil.copy(generator_path, new_generator_path)

        config_str = json.dumps(config_js, indent=2)
        with open(os.path.join(tmpdir, 'generator.json'), 'w') as fh:
            fh.write(config_str)

        # Generate some data here

        p = subprocess.Popen(new_generator_path, shell=True, cwd=tmpdir)
        p.communicate()
        if p.returncode != 0:
            logger.error('Could not generate data, code: %s' % p.returncode)
            return None

        # Generated file:
        data_files = [f for f in os.listdir(tmpdir) if os.path.isfile(os.path.join(tmpdir, f))
                      and f.endswith('bin')]
        if len(data_files) != 1:
            logger.error('Error in generating data to process. Files found: %s' % data_files)
            return None

        data_file = os.path.join(tmpdir, data_files[0])
        return data_file

    def clean_temp_dir(self, tmpdir):
        """
        Cleans artifacts
        :param tmpdir:
        :return:
        """
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)

    def is_function_egen(self, function):
        """
        Returns true if function is generated by EAcirc generator
        :param function:
        :return:
        """
        return function in egenerator.ROUNDS or function in egenerator.SHA3 or function in egenerator.ESTREAM

    def get_test_battery(self):
        """
        Returns function -> [r1, r2, r3, ...] to test on given number of rounds.
        :return:
        """
        battery = dict(egenerator.ROUNDS)

        # Another tested functions, not (yet) included in egen.
        battery['MD5'] = [15, 16, 17]
        battery['SHA256'] = [3, 4]
        battery['RC4'] = [1]
        battery['RC4_Col'] = [1]

        battery['crand_aisa'] = [1]
        battery['javarand'] = [1]

        return battery

    def egen_benchmark(self):
        """
        Benchmarks egenerator speed.
        :return:
        """
        fnc_estream = sorted(list(egenerator.ESTREAM.keys()))
        fnc_sha3 = sorted(list(egenerator.SHA3.keys()))
        fnc_block = sorted(list(egenerator.BLOCK.keys()))

        logger.info('Egenerator benchmark. Total functions:')
        logger.info('  Estream: %s' % fnc_estream)
        logger.info('  Sha3:    %s' % fnc_sha3)
        logger.info('  Block:   %s' % fnc_block)

        fnc_rounds = sorted(list(egenerator.ROUNDS.keys()))
        logger.info('  Round reduced targets: %s' % fnc_rounds)

        data_to_gen = 1024*1024*10
        results = {}
        for fnc in fnc_rounds:
            rounds = egenerator.ROUNDS[fnc]
            cur_round = max(rounds)

            logger.info('Testing %s, round %s' % (fnc, cur_round))
            tmpdir = self.gen_randomdir(fnc, cur_round)
            try:
                config_js = egenerator.get_config(function_name=fnc, rounds=cur_round, data=data_to_gen)

                time_start = time.time()
                data_file = self.eacirc_generator(tmpdir=tmpdir, generator_path=self.generator_path,
                                                  config_js=config_js)

                time_elapsed = time.time() - time_start
                results[fnc] = time_elapsed

                logger.info('Finished, time: %s' % time_elapsed)

            except Exception as e:
                logger.error('Exception in generating %s: %s' % (fnc, e))
                logger.debug(traceback.format_exc())

            finally:
                self.clean_temp_dir(tmpdir)

        # JSON + CSV output
        print(json.dumps(results, indent=2))
        print('-'*80)
        print('function,time')
        for fnc in results:
            print('%s,%s' % (fnc, results[fnc]))

    # noinspection PyBroadException
    def work(self):
        """
        Main entry point - data processing
        :return:
        """
        self.init_params()

        # Special code path: benchmarking egenerator
        if self.args.egen_benchmark:
            self.egen_benchmark()
            return

        # Init logic, analysis.
        # Define test set.
        test_sizes_mb = self.args.matrix_size
        test_block_sizes = self.args.matrix_block
        test_degree = self.args.matrix_deg
        test_comb_k = self.args.matrix_comb_deg
        logger.info('Computing test matrix for sizes: %s, blocks: %s, degree: %s, comb degree: %s'
                    % (test_sizes_mb, test_block_sizes, test_degree, test_comb_k))

        # Test all functions
        battery = self.get_test_battery()
        functions = sorted(list(battery.keys()))
        self.data_to_gen = max(test_sizes_mb) * 1024 * 1024
        logger.info('Battery of functions to test: %s' % battery)

        total_test_idx = 0
        for function in functions:
            rounds = battery[function]

            # Generate random tmpdir, generate data, test it there...
            for cur_round in rounds:
                tmpdir = self.gen_randomdir(function, cur_round)
                if self.is_function_egen(function):
                    self.config_js = egenerator.get_config(function_name=function, rounds=cur_round, data=self.data_to_gen)
                else:
                    self.config_js = {'algorithm': function, 'round': cur_round, 'seed': self.seed}

                # Reseed testcase scenario random generator
                test_rand_seed = self.test_random.randint(0, 2**64-1)
                self.test_random.seed(test_rand_seed)

                # Generate test cases, run the analysis.
                for test_case in self.test_case_generator(test_sizes_mb, test_block_sizes, test_degree, test_comb_k):
                    data_size, block_size, degree, comb_deg = test_case
                    total_test_idx += 1

                    test_desc = 'idx: %04d, data: %04d, block: %d, deg: %d, comb-deg: %d, fun: %s, round: %s' \
                                % (total_test_idx, data_size, block_size, degree, comb_deg, function, cur_round)

                    if self.test_manuals > 1 and (total_test_idx % self.test_manuals) != self.test_stride:
                        logger.info('Skipping test %s' % test_desc)
                        continue

                    res_file = '%s-r%02d-seed%s-%04dMB-%sbl-%sdeg-%sk.json' \
                               % (function, cur_round, self.config_js['seed'], data_size, block_size, degree, comb_deg)
                    res_file_path = os.path.join(self.results_dir, res_file)
                    if self.check_res_file(res_file_path):
                        logger.info('Already computed test %s' % test_desc)
                        continue

                    data_file = self.data_generator(tmpdir=tmpdir, function=function, cur_round=cur_round)
                    if data_file is None:
                        logger.error('Data file is invalid')
                        continue

                    logger.info('Working on test: %s' % test_desc)
                    jsres = self.testcase(function, cur_round, data_size, block_size, degree, comb_deg,
                                          data_file, tmpdir)

                    with open(res_file_path, 'w') as fh:
                        fh.write(json.dumps(jsres, indent=2))

                # Remove test dir
                self.clean_temp_dir(tmpdir)

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
        hwanalysis.best_x_combinations = self.args.best_x_combinations

        logger.info('Initializing test')
        time_test_start = time.time()
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
        jsres['time_elapsed'] = time.time() - time_test_start

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

        parser = argparse.ArgumentParser(description='Testbed run matrix of tests on multiple possible functions.')
        parser.add_argument('-t', '--threads', dest='threads', type=int, default=None,
                            help='Number of threads to use')

        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')

        parser.add_argument('--verbose', dest='verbose', action='store_const', const=True,
                            help='enables verbose mode')

        parser.add_argument('--top', dest='topk', default=30, type=int,
                            help='top K number of best distinguishers to combine together')

        parser.add_argument('--comb-rand', dest='comb_random', default=0, type=int,
                            help='number of terms to add randomly to the combination set')

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

        parser.add_argument('--best-x-combs', dest='best_x_combinations', default=None, type=int,
                            help='Number of best combinations to return. If defined, heap is used')

        #
        # Testbed related options
        #

        parser.add_argument('--generator-path', dest='generator_path', default=None,
                            help='Path to the EAcirc generator executable')

        parser.add_argument('--result-dir', dest='results_dir', default=None,
                            help='Directory to put results files to')

        parser.add_argument('--data-dir', dest='data_dir', default=None,
                            help='Directory to load data from (precomputed samples to test)')

        parser.add_argument('--tests-manuals', dest='tests_manuals', default=1, type=int,
                            help='Total number of manually started workers for this computation')

        parser.add_argument('--tests-stride', dest='tests_stride', default=1, type=int,
                            help='Tests stride, skipping tests. Index of this particular worker in the batch.')

        parser.add_argument('--tests-random-select-seed', dest='tests_random_select_eed', default=0, type=int,
                            help='Seed for test ordering randomization, defined allocation on workers')

        parser.add_argument('--egen-benchmark', dest='egen_benchmark', action='store_const', const=True, default=False,
                            help='Benchmarks speed of the egenerator')

        #
        # Testing matrix definition
        #

        parser.add_argument('--matrix-block', dest='matrix_block', nargs=argparse.ZERO_OR_MORE,
                            default=[128, 256, 384, 512], type=int,
                            help='List of block sizes to test')

        parser.add_argument('--matrix-size', dest='matrix_size', nargs=argparse.ZERO_OR_MORE,
                            default=[1, 10, 100], type=int,
                            help='List of data sizes to test in MB')

        parser.add_argument('--matrix-deg', dest='matrix_deg', nargs=argparse.ZERO_OR_MORE,
                            default=[1, 2], type=int,
                            help='List of degree to test')

        parser.add_argument('--matrix-comb-deg', dest='matrix_comb_deg', nargs=argparse.ZERO_OR_MORE,
                            default=[1, 2, 3], type=int,
                            help='List of degree of combinations to test')

        self.args = parser.parse_args()
        self.work()


# Launcher
app = None
if __name__ == "__main__":
    app = TestbedBenchmark()
    app.main()

