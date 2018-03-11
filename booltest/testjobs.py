#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import os
import shutil
import subprocess
import time
import math
import traceback
import copy

import scipy.misc
import scipy.stats

from booltest import egenerator
from booltest import common
from booltest.booltest_main import *


logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)

job_tpl_hdr = '''#!/bin/bash

export BOOLDIR="/storage/brno3-cerit/home/${LOGNAME}/booltest/assets"
export RESDIR="/storage/brno3-cerit/home/${LOGNAME}/bool-res"
export LOGDIR="/storage/brno3-cerit/home/${LOGNAME}/bool-log"

cd "${BOOLDIR}"

'''

job_tpl = '''
./generator-metacentrum.sh -c=%s | ./booltest-json-metacentrum.sh \\
    %s > "${LOGDIR}/%s.out" 2> "${LOGDIR}/%s.err"
'''


class TestCaseEntry(object):
    def __init__(self, fnc, params=None, stream_type=None, data_file=None):
        self.fnc = fnc
        self.params = params
        self.stream_type = stream_type
        self.data_file = data_file
        self.rounds = None
        self.c_round = None
        self.data_size = None
        self.is_egen = False
        self.gen_cfg = None
        self.strategy = 'static'

    def to_json(self):
        """
        To json serialize
        :return:
        """
        return dict(self.__dict__)


class TestRun(object):
    def __init__(self, spec, block_size=None, degree=None, comb_deg=None, total_test_idx=None, test_desc=None,
                 res_file=None):
        self.spec = spec  # type: TestCaseEntry
        self.block_size = block_size
        self.degree = degree
        self.comb_deg = comb_deg
        self.total_test_idx = total_test_idx
        self.test_desc = test_desc
        self.res_file = res_file

    def to_json(self):
        """
        To json serialize
        :return:
        """
        return dict(self.__dict__)


# Main - argument parsing + processing
class Testjobs(Booltest):
    """
    Testbed run matrix of tests on multiple possible functions.

    Used in conjunction with EACirc generator. A function \in generator is used with
    different number of rounds to benchmark success rate of the polynomial approach.

    Generates PBSpro jobs
    """
    def __init__(self, *args, **kwargs):
        super(Testjobs, self).__init__(*args, **kwargs)
        self.args = None
        self.tester = None
        self.input_poly = []

        self.results_dir = None
        self.job_dir = None
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
        self.job_dir = self.args.job_dir
        if not self.job_dir:
            raise ValueError('job dir empty')

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

    def find_data_file(self, function, round, size=None):
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

        size = size if size else self.data_to_gen
        candidates = [
            '%s_r%s_seed%s_%sMB.bin' % (function, round, self.seed, size//1024//1024),
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

            if os.path.getsize(fpath) < size:
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

    def get_test_battery(self, include_all=False):
        """
        Returns function -> [r1, r2, r3, ...] to test on given number of rounds.
        :return:
        """
        battery = dict(egenerator.ROUNDS)

        # Another tested functions, not (yet) included in egen.
        # battery['MD5'] = [15, 16, 17]
        # battery['SHA256'] = [3, 4]
        # battery['RC4'] = [1]
        # battery['RC4_Col'] = [1]
        #
        # battery['crand_aisa'] = [1]
        # battery['javarand'] = [1]

        # for non-stated try all
        if not include_all:
            return battery

        all_fnc = common.merge_dicts([egenerator.SHA3, egenerator.ESTREAM, egenerator.BLOCK])
        for ckey in all_fnc.keys():
            fnc = all_fnc[ckey]  # type: egenerator.FunctionParams
            max_rounds = fnc.rounds if fnc else None
            if max_rounds is None:
                battery['ckey'] = [1]
                continue

            try_rounds = max_rounds if max_rounds < 10 else min(int(math.ceil(max_rounds * 0.4)), 15)
            battery[ckey] = list(range(1, try_rounds))

        return battery

    # noinspection PyBroadException
    def work(self):
        """
        Main entry point - data processing
        :return:
        """
        self.init_params()

        # Init logic, analysis.
        # Define test set.
        test_sizes_mb = self.args.matrix_size
        test_block_sizes = self.args.matrix_block
        test_degree = self.args.matrix_deg
        test_comb_k = self.args.matrix_comb_deg
        logger.info('Computing test matrix for sizes: %s, blocks: %s, degree: %s, comb degree: %s'
                    % (test_sizes_mb, test_block_sizes, test_degree, test_comb_k))

        # Test all functions
        all_functions = common.merge_dicts([egenerator.SHA3, egenerator.ESTREAM, egenerator.BLOCK])
        battery = self.get_test_battery()
        functions = sorted(list(battery.keys()))
        logger.info('Battery of functions to test: %s' % battery)

        # (function, round, processing_type)
        test_array = []

        total_test_idx = 0
        for fnc in functions:
            rounds = battery[fnc]
            params = all_functions[fnc] if fnc in all_functions else None  # type: egenerator.FunctionParams
            tce = TestCaseEntry(fnc, params)
            tce.rounds = rounds

            if egenerator.is_function_egen(fnc):
                fnc = egenerator.normalize_function_name(fnc)
                tce.fnc = fnc
                tce.stream_type = egenerator.function_to_stream_type(fnc)
                tce.is_egen = True

            iterator = itertools.product(rounds, test_sizes_mb)
            for cur_round, size_mb in iterator:
                tce_c = copy.deepcopy(tce)
                tce_c.c_round = cur_round
                tce_c.data_size = size_mb*1024*1024

                if not tce.is_egen:
                    tce_c.data_file = self.find_data_file(function=tce.fnc, round=cur_round, size=tce_c.data_size)
                    test_array.append(tce_c)
                    continue

                is_sha3 = tce.stream_type == egenerator.FUNCTION_SHA3
                is_stream = tce.stream_type == egenerator.FUNCTION_ESTREAM
                is_block = tce.stream_type == egenerator.FUNCTION_BLOCK

                fgc = egenerator.FunctionGenConfig(fnc, rounds=cur_round, data=tce_c.data_size)
                fun_configs = []

                # Random key, enc zeros
                if is_stream:
                    fun_configs += [
                        egenerator.get_function_config(fgc,
                                                       src_input=egenerator.get_zero_stream(),
                                                       src_key=egenerator.get_random_stream())
                    ]

                # zero input, reinit keys, different types (col style)
                # if is_stream or is_block:
                #     fun_configs += [
                #          egenerator.zero_inp_reinit_key(fgc, egenerator.get_hw_stream(4)),
                #          egenerator.zero_inp_reinit_key(fgc, egenerator.get_counter_stream()),
                #          egenerator.zero_inp_reinit_key(fgc, egenerator.get_sac_step_stream()),
                #          egenerator.zero_inp_reinit_key(fgc, egenerator.get_random_stream()),
                #     ]

                # 1. (rpsc, rpsc-xor, sac, sac-xor, hw, counter) input, random key. 2 different random keys
                for i in range(1 if is_sha3 else 2):
                    if is_stream:
                        continue
                    fun_key = egenerator.get_zero_stream() if i == 0 else egenerator.get_random_stream(i-1)
                    fun_configs += [
                        egenerator.rpcs_inp(fgc, fun_key),
                        egenerator.rpcs_inp_xor(fgc, fun_key),
                        egenerator.sac_inp(fgc, fun_key),
                        egenerator.sac_xor_inp(fgc, fun_key),
                        egenerator.get_function_config(fgc, src_input=egenerator.get_hw_stream(4), src_key=fun_key),
                        egenerator.get_function_config(fgc, src_input=egenerator.get_counter_stream(), src_key=fun_key),
                    ]

                for fun_cfg in fun_configs:
                    tce_c.gen_cfg = egenerator.get_config_header(fgc, stdout=True, stream=fun_cfg)
                    tce_c.strategy = egenerator.get_scode(fun_cfg)
                    test_array.append(copy.deepcopy(tce_c))

        # test multiple booltest params
        test_runs = []  # type: list[TestRun]
        logger.info('Test array size: %s' % len(test_array))

        # For each test specification
        for test_spec in test_array:  # type: TestCaseEntry

            # Generate test cases, run the analysis.
            for test_case in itertools.product(test_block_sizes, test_degree, test_comb_k):
                block_size, degree, comb_deg = test_case
                data_size = int(test_spec.data_size / 1024 / 1024)
                total_test_idx += 1

                test_desc = 'idx: %04d, data: %04d, block: %s, deg: %s, comb-deg: %s, fun: %s, round: %s scode: %s' \
                            % (total_test_idx, data_size, block_size, degree, comb_deg, test_spec.fnc,
                               test_spec.c_round, test_spec.strategy)

                res_file = '%s-%04dMB-%sbl-%sdeg-%sk.json' \
                           % (test_spec.strategy, data_size, block_size, degree, comb_deg)
                res_file = res_file.replace(' ', '')

                trun = TestRun(test_spec, block_size, degree, comb_deg, total_test_idx, test_desc, res_file)
                test_runs.append(trun)

        # Sort by estimated complexity
        test_runs.sort(key=lambda x: (x.spec.data_size, x.degree, x.block_size, x.comb_deg))
        logger.info('Total num of tests: %s' % len(test_runs))

        # Load input polynomials
        self.load_input_poly()

        # Generate job files
        job_files = []
        job_batch = []
        job_batch_max_size = 50
        cur_batch_def = None  # type: TestRun

        memory_threshold = 50
        for fidx, trun in enumerate(test_runs):  # type: TestRun
            hwanalysis = self.testcase(trun.block_size, trun.degree, trun.comb_deg)
            json_config = collections.OrderedDict()
            size_mb = trun.spec.data_size / 1024 / 1024
            json_config['config'] = trun
            json_config['hwanalysis'] = hwanalysis
            json_config['fidx'] = fidx

            res_file_path = os.path.join(self.results_dir, trun.res_file)
            gen_file_path = os.path.join(self.job_dir, 'gen-' + trun.res_file)
            job_file_path = os.path.join(self.job_dir, 'job-' + trun.res_file + '.sh')
            cfg_file_path = os.path.join(self.job_dir, 'cfg-' + trun.res_file)
            if os.path.exists(cfg_file_path):
                logger.warning('Conflicting config: %s' % common.json_dumps(json_config, indent=2))
                raise ValueError('File name conflict: %s, test idx: %s' % (cfg_file_path, fidx))

            json_config['res_file'] = res_file_path
            json_config['gen_file'] = gen_file_path

            with open(gen_file_path, 'w+') as fh:
                json.dump(trun.spec.gen_cfg, fh, indent=2)
            with open(cfg_file_path, 'w+') as fh:
                fh.write(common.json_dumps(json_config, indent=2))

            args = ' --config-file %s' % cfg_file_path
            job_data = job_tpl % (gen_file_path, args, trun.res_file, trun.res_file)
            job_batch.append(job_data)

            flush_batch = False
            if cur_batch_def is None:
                cur_batch_def = trun
                job_batch_max_size = 10
                if size_mb < 11:
                    job_batch_max_size = 25
                if size_mb < 2:
                    job_batch_max_size = 50

            elif cur_batch_def.spec.data_size != trun.spec.data_size \
                    or len(job_batch) >= job_batch_max_size:
                flush_batch = True

            if flush_batch:
                job_data = job_tpl_hdr + '\n'.join(job_batch)
                job_batch = []
                cur_batch_def = None

                with open(job_file_path, 'w+') as fh:
                    fh.write(job_data)

                ram = '6gb' if size_mb > memory_threshold else '2gb'
                job_time = '24:00:00'
                if size_mb < 11:
                    job_time = '4:00:00'
                if size_mb < 2:
                    job_time = '2:00:00'
                job_files.append((job_file_path, ram, job_time))

            if fidx % 1000 == 0:
                logger.debug('Generated %s files, jobs: %s' % (fidx, len(job_files)))

        logger.info('Generated job files: %s' % len(job_files))

        # Enqueue
        with open(os.path.join(self.job_dir, 'enqueue-meta-%s.sh' % int(time.time())), 'w') as fh:
            fh.write('#!/bin/bash\n\n')
            for fn in job_files:
                fh.write('qsub -l select=1:ncpus=1:mem=%s -l walltime=%s %s \n' % (fn[1], fn[2], fn[0]))

    def testcase(self, blocklen, degree, comb_deg):
        """
        Test case constructor
        :param function:
        :param cur_round:
        :param size_mb:
        :param blocklen:
        :param degree:
        :param comb_deg:
        :param data_file:
        :return:
        """
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
        hwanalysis.sort_best_zscores = max(common.replace_none([self.args.topterm_heap_k, self.top_k, 100]))
        hwanalysis.best_x_combinations = self.args.best_x_combinations
        return hwanalysis

    def main(self):
        logger.debug('App started')

        parser = argparse.ArgumentParser(description='Generates a job matrix of tests on multiple possible functions.')

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

        parser.add_argument('--job-dir', dest='job_dir', default=None,
                            help='Directory to put job files to')

        parser.add_argument('--result-dir', dest='results_dir', default=None,
                            help='Directory to put results files to')

        parser.add_argument('--data-dir', dest='data_dir', default=None,
                            help='Directory to load data from (precomputed samples to test)')

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

        parser.add_argument('--matrix-method', dest='matrix_method', nargs=argparse.ZERO_OR_MORE,
                            default=None,
                            help='List of methods for data generation')

        self.args = parser.parse_args()
        self.work()


# Launcher
app = None
if __name__ == "__main__":
    app = Testjobs()
    app.main()

