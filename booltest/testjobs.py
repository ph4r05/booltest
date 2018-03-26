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
import subprocess
import sys

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
                 res_file=None, gen_file=None):
        self.spec = spec  # type: TestCaseEntry
        self.block_size = block_size
        self.degree = degree
        self.comb_deg = comb_deg
        self.total_test_idx = total_test_idx
        self.test_desc = test_desc
        self.res_file = res_file
        self.gen_file = gen_file
        self.iteration = 0

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
        self.time_experiment = int(time.time())
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

        if self.args.only_rounds:
            self.args.only_rounds = [int(x) for x in self.args.only_rounds]

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
                return 'best_dists' in js and 'data_read' in js and js['data_read'] > 0
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

    def random_seed(self):
        """
        Generates random seed
        :return:
        """
        r = self.test_random.getrandbits(8*8)
        return '%016x' % r

    def try_chmod_grx(self, path):
        """
        Attempts to add +x flag
        :param path:
        :return:
        """
        try:
            os.chmod(path, 0o750)
        except Exception as e:
            logger.warning('Exception chmoddin %s: %s' % (path, e))

    def try_chmod_gr(self, path):
        """
        Update permissions
        :param path:
        :return:
        """
        try:
            os.chmod(path, 0o640)
        except Exception as e:
            logger.warning('Error chmodding %s : %s' % (path, e))

    def get_test_battery(self):
        """
        Returns function -> [r1, r2, r3, ...] to test on given number of rounds.
        :return:
        """
        if self.args.ref_only or self.args.all_zscores:
            return {'AES': [10]}

        battery = dict(egenerator.ROUNDS)
        if self.args.only_crypto:
            battery = egenerator.filter_functions(battery, set(self.args.only_crypto))
        elif self.args.narrow:
            battery = egenerator.filter_functions(battery, egenerator.NARROW_SELECTION)

        if self.args.only_rounds:
            for fnc in battery:
                battery[fnc] = self.args.only_rounds
        skip = {}

        # Another tested functions, not (yet) included in egen.
        # battery['MD5'] = [15, 16, 17]
        # battery['SHA256'] = [3, 4]
        # battery['RC4'] = [1]
        # battery['RC4_Col'] = [1]
        #
        # battery['crand_aisa'] = [1]
        # battery['javarand'] = [1]

        # for non-stated try all
        if not self.args.include_all:
            return battery

        all_fnc = egenerator.all_functions()
        for ckey in all_fnc.keys():
            fnc = all_fnc[ckey]  # type: egenerator.FunctionParams
            max_rounds = fnc.rounds if fnc else None
            if max_rounds is None or ckey in battery or ckey in egenerator.ROUNDS or ckey in skip:
                continue

            try_rounds = min(5, max_rounds)
            battery[ckey] = list(range(1, try_rounds))

        return battery

    def is_job_expired(self, config_path):
        """
        Determines if the job expired
        :param config_path:
        :return:
        """
        mtime = os.path.getmtime(config_path)
        return self.time_experiment >= mtime + 60*60*24*1.2

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
        all_functions = egenerator.all_functions()
        battery = self.get_test_battery()
        functions = sorted(list(battery.keys()))
        logger.info('Battery of functions to test: %s' % battery)

        # Combinations
        combinations = {}
        for bl in list(set(test_block_sizes + [self.args.topk])):
            combinations[bl] = {}
            for ii in list(set(test_degree + test_comb_k)):
                combinations[bl][ii] = int(scipy.misc.comb(bl, ii, True))
        full_combination = combinations[max(test_block_sizes)][max(max(test_degree), max(test_comb_k))]

        # (function, round, processing_type)
        test_array = []

        total_test_idx = 0
        for fnc in functions:
            rounds = battery[fnc]

            # Validation round 1
            if self.args.add_round1 and 1 not in rounds:
                rounds.insert(0, 1)

            params = all_functions[fnc] if fnc in all_functions else None  # type: egenerator.FunctionParams
            tce = TestCaseEntry(fnc, params)
            tce.rounds = rounds
            is_3des = fnc.lower() == 'triple-des'  # special key handling, cannot use ctr/hw

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

                is_sha3 = tce.stream_type in [egenerator.FUNCTION_SHA3, egenerator.FUNCTION_HASH]
                is_stream = tce.stream_type == egenerator.FUNCTION_ESTREAM
                is_block = tce.stream_type == egenerator.FUNCTION_BLOCK

                blk_lower_than_16b = params and params.block_size and params.block_size < 16
                key_lower_than_16b = params and params.key_size and params.key_size < 16
                hw_val = 4 if not blk_lower_than_16b else 6
                hw_key_val = 4 if not key_lower_than_16b else 6

                fgc = egenerator.FunctionGenConfig(fnc, rounds=cur_round, data=tce_c.data_size, params=tce_c.params)
                fun_configs = []

                # Random key, enc zeros - stream ciphers
                if is_stream:
                    fun_configs += [
                        egenerator.get_function_config(fgc,
                                                       src_input=egenerator.get_zero_stream(),
                                                       src_key=egenerator.get_random_stream())
                    ]

                # zero input, reinit keys, different types (col style)
                if not self.args.no_reinit and (is_stream or is_block):
                    fun_configs += [
                         egenerator.zero_inp_reinit_key(fgc, egenerator.get_random_stream()),
                    ]

                    if not is_3des:
                        fun_configs += [
                            egenerator.zero_inp_reinit_key(fgc, egenerator.get_hw_stream(hw_key_val)),
                            egenerator.zero_inp_reinit_key(fgc, egenerator.get_counter_stream())
                        ]

                    if self.args.ref_only:
                        fun_configs += [
                            egenerator.zero_inp_reinit_key(fgc, egenerator.get_hw_stream(6)),
                        ]

                    if not self.args.no_sac:
                        fun_configs += [
                            egenerator.zero_inp_reinit_key(fgc, egenerator.get_sac_step_stream()),
                        ]

                # 1. (rpsc, rpsc-xor, sac, sac-xor, hw, counter) input, random key. 2 different random keys
                target_iterations = 1 if is_sha3 else 2
                if self.args.ref_only or self.args.all_zscores:
                    target_iterations = 1

                for i in range(target_iterations):
                    if is_stream:
                        continue

                    fun_key = egenerator.get_zero_stream() if i == 0 and not self.args.ref_only else\
                        egenerator.get_random_stream(i-1)

                    if not self.args.no_counters:
                        fun_configs += [
                            egenerator.get_function_config(fgc, src_input=egenerator.get_hw_stream(hw_val), src_key=fun_key),
                            egenerator.get_function_config(fgc, src_input=egenerator.get_counter_stream(), src_key=fun_key),
                        ]

                    if self.args.ref_only:
                        fun_configs += [
                            egenerator.get_function_config(fgc, src_input=egenerator.get_hw_stream(6), src_key=fun_key)
                        ]

                    if not self.args.counters_only and not self.args.no_rpcs:
                        fun_configs += [
                            egenerator.rpcs_inp(fgc, fun_key),
                        ]

                    if not self.args.counters_only and not self.args.no_sac:
                        fun_configs += [
                            egenerator.sac_inp(fgc, fun_key),
                        ]

                    if not self.args.counters_only and not self.args.no_xor_strategy:
                        fun_configs += [
                            egenerator.rpcs_inp_xor(fgc, fun_key),
                            egenerator.sac_xor_inp(fgc, fun_key),
                        ]

                    if self.args.inhwr1:
                        fun_configs += [
                            egenerator.get_function_config(fgc, src_input=egenerator.get_hw_stream(1, randomize_overflow=True), src_key=fun_key),
                        ]

                    if self.args.inhwr2:
                        fun_configs += [
                            egenerator.get_function_config(fgc, src_input=egenerator.get_hw_stream(2, randomize_overflow=True), src_key=fun_key),
                        ]

                    if self.args.inhwr4:
                        fun_configs += [
                            egenerator.get_function_config(fgc, src_input=egenerator.get_hw_stream(4, randomize_overflow=True), src_key=fun_key),
                        ]

                for fun_cfg in fun_configs:
                    seed = self.random_seed()
                    tce_c.gen_cfg = egenerator.get_config_header(fgc, stdout=True, stream=fun_cfg, seed=seed)
                    tce_c.gen_cfg['exp_time'] = self.time_experiment
                    tce_c.strategy = egenerator.get_scode(fun_cfg)
                    test_array.append(copy.deepcopy(tce_c))

        # test multiple booltest params
        test_runs = []  # type: list[TestRun]
        logger.info('Test array size: %s' % len(test_array))

        # For each test specification
        generator_files = set()
        for test_spec in test_array:  # type: TestCaseEntry

            # Generate test cases, run the analysis.
            test_runs_times = [0]
            if self.args.ref_only:
                test_runs_times = range(100)

            for test_case in itertools.product(test_block_sizes, test_degree, test_comb_k, test_runs_times):
                block_size, degree, comb_deg, trt = test_case
                data_size = int(test_spec.data_size / 1024 / 1024)
                total_test_idx += 1

                if trt > 0:
                    test_spec = copy.deepcopy(test_spec)
                    seed = self.random_seed()
                    test_spec.gen_cfg['seed'] = seed

                test_desc = 'idx: %04d, data: %04d, block: %s, deg: %s, comb-deg: %s, fun: %s, round: %s scode: %s, %s' \
                            % (total_test_idx, data_size, block_size, degree, comb_deg, test_spec.fnc,
                               test_spec.c_round, test_spec.strategy, trt)

                suffix = 'json' if not self.args.all_zscores else 'csv'
                test_type = '' if not self.args.all_zscores else '-zscores'
                res_file = '%s-%04dMB-%sbl-%sdeg-%sk%s%s.%s' \
                           % (test_spec.strategy, data_size, block_size, degree, comb_deg,
                              ('-%s' % trt) if trt > 0 else '', test_type, suffix)
                res_file = res_file.replace(' ', '')

                gen_file = 'gen-%s-%04dMB-%s.json' % (test_spec.strategy, data_size, trt)
                gen_file = gen_file.replace(' ', '')

                if data_size == test_sizes_mb[0]:
                    generator_files.add(os.path.join(self.job_dir, gen_file))

                trun = TestRun(test_spec, block_size, degree, comb_deg, total_test_idx, test_desc, res_file, gen_file)
                trun.iteration = trt
                test_runs.append(trun)

        # Sort by estimated complexity
        test_runs.sort(key=lambda x: (x.spec.data_size, x.degree, x.block_size, x.comb_deg))
        logger.info('Total num of tests: %s' % len(test_runs))

        # Load input polynomials
        self.load_input_poly()

        # Generate job files
        job_files = []
        job_batch = []
        batch_max_bl = 0
        batch_max_deg = 0
        batch_max_comb_deg = 0
        job_batch_max_size = 50
        cur_batch_def = None  # type: TestRun

        memory_threshold = 50
        num_skipped = 0
        num_skipped_existing = 0
        for fidx, trun in enumerate(test_runs):  # type: TestRun
            hwanalysis = self.testcase(trun.block_size, trun.degree, trun.comb_deg)
            json_config = collections.OrderedDict()
            size_mb = trun.spec.data_size / 1024 / 1024
            json_config['exp_time'] = self.time_experiment
            json_config['config'] = trun
            json_config['hwanalysis'] = hwanalysis
            json_config['fidx'] = fidx

            res_file_path = os.path.join(self.results_dir, trun.res_file)
            gen_file_path = os.path.join(self.job_dir, trun.gen_file)
            job_file_path = os.path.join(self.job_dir, 'job-' + trun.res_file + '.sh')
            cfg_file_path = os.path.join(self.job_dir, 'cfg-' + trun.res_file)

            json_config['res_file'] = res_file_path
            json_config['gen_file'] = gen_file_path
            json_config['backup_dir'] = self.args.backup_dir
            json_config['skip_finished'] = self.args.skip_finished
            json_config['all_zscores'] = self.args.all_zscores

            if fidx % 1000 == 0:
                logger.debug('Processing file %s, jobs: %s' % (fidx, len(job_files)))

            if self.args.skip_finished and self.check_res_file(res_file_path):
                num_skipped += 1
                continue

            if os.path.exists(cfg_file_path):
                if self.args.skip_existing:
                    if self.args.expiring and self.is_job_expired(cfg_file_path):
                        logger.debug('Job expired: %s' % cfg_file_path)
                    else:
                        num_skipped_existing += 1
                        continue

                if self.args.overwrite_existing:
                    logger.debug('Overwriting %s' % cfg_file_path)

                else:
                    logger.warning('Conflicting config: %s' % common.json_dumps(json_config, indent=2))
                    raise ValueError('File name conflict: %s, test idx: %s' % (cfg_file_path, fidx))

            if not os.path.exists(gen_file_path):
                with open(gen_file_path, 'w+') as fh:
                    fh.write(common.json_dumps(trun.spec.gen_cfg, indent=2))
                self.try_chmod_gr(gen_file_path)

            with open(cfg_file_path, 'w+') as fh:
                fh.write(common.json_dumps(json_config, indent=2))

            args = ' --config-file %s' % cfg_file_path
            job_data = job_tpl % (gen_file_path, args, trun.res_file, trun.res_file)
            job_batch.append(job_data)
            generator_files.add(gen_file_path)

            batch_max_bl = max(batch_max_bl, trun.block_size)
            batch_max_deg = max(batch_max_deg, trun.degree)
            batch_max_comb_deg = max(batch_max_comb_deg, trun.comb_deg)

            flush_batch = False
            if cur_batch_def is None:
                cur_batch_def = trun
                job_batch_max_size = 15

                if size_mb < 11:
                    job_batch_max_size = 25
                    if batch_max_deg <= 2 and batch_max_comb_deg <= 2:
                        job_batch_max_size = 50
                    if batch_max_deg <= 1 and batch_max_comb_deg <= 2:
                        job_batch_max_size = 100

                if size_mb < 2:
                    job_batch_max_size = 100
                    if batch_max_deg <= 2 and batch_max_comb_deg <= 2:
                        job_batch_max_size = 200
                    if batch_max_deg <= 1 and batch_max_comb_deg <= 2:
                        job_batch_max_size = 300

            elif cur_batch_def.spec.data_size != trun.spec.data_size \
                    or len(job_batch) >= job_batch_max_size:
                flush_batch = True

            if flush_batch:
                job_data = job_tpl_hdr + '\n'.join(job_batch)
                job_batch = []
                batch_max_bl = 0
                batch_max_deg = 0
                batch_max_comb_deg = 0
                cur_batch_def = None

                with open(job_file_path, 'w+') as fh:
                    fh.write(job_data)

                ram = '12gb' if size_mb > memory_threshold else '6gb'
                job_time = '24:00:00'
                if size_mb < 11:
                    job_time = '4:00:00'
                job_files.append((job_file_path, ram, job_time))

        logger.info('Generated job files: %s, skipped: %s, skipped existing: %s'
                    % (len(job_files), num_skipped, num_skipped_existing))

        # Enqueue
        enqueue_path = os.path.join(self.job_dir, 'enqueue-meta-%s.sh' % int(time.time()))
        with open(enqueue_path, 'w') as fh:
            fh.write('#!/bin/bash\n\n')
            for fn in job_files:
                fh.write('qsub -l select=1:ncpus=1:mem=%s -l walltime=%s %s \n' % (fn[1], fn[2], fn[0]))

        # Generator tester file
        testgen_path = os.path.join(self.job_dir, 'test-generators-%s.sh' % int(time.time()))
        with open(testgen_path, 'w') as fh:
            fh.write(job_tpl_hdr)
            for fn in sorted(list(generator_files)):
                if not os.path.exists(fn):
                    continue
                fh.write('./generator-metacentrum.sh -c=%s 2>/dev/null >/dev/null\n' % fn)
                fh.write('if [ $? -ne 0 ]; then echo "Generator failed: %s"; else echo -n "."; fi\n' % fn)
            fh.write('\n')

        # chmod
        self.try_chmod_grx(enqueue_path)
        self.try_chmod_grx(testgen_path)
        logger.info('Gentest: %s' % testgen_path)
        logger.info('Enqueue: %s' % enqueue_path)

        if self.args.enqueue:
            logger.info('Enqueueing...')
            p = subprocess.Popen(enqueue_path, stdout=sys.stdout, stderr=sys.stderr, shell=True)
            p.wait()

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

        parser.add_argument('--backup-dir', dest='backup_dir', default=None,
                            help='Backup directory for overwritten results')

        parser.add_argument('--data-dir', dest='data_dir', default=None,
                            help='Directory to load data from (precomputed samples to test)')

        parser.add_argument('--tests-random-select-seed', dest='tests_random_select_eed', default=0, type=int,
                            help='Seed for test ordering randomization, defined allocation on workers')

        parser.add_argument('--egen-benchmark', dest='egen_benchmark', action='store_const', const=True, default=False,
                            help='Benchmarks speed of the egenerator')

        parser.add_argument('--skip-existing', dest='skip_existing', action='store_const', const=True, default=False,
                            help='Skip existing jobs')

        parser.add_argument('--expiring', dest='expiring', action='store_const', const=True, default=False,
                            help='Skip existing jobs - but check for expiration on result, did job finished?')

        parser.add_argument('--overwrite-existing', dest='overwrite_existing', action='store_const', const=True, default=False,
                            help='Overwrites existing jobs')

        parser.add_argument('--skip-finished', dest='skip_finished', action='store_const', const=True, default=False,
                            help='Skip tests with generated valid results')

        parser.add_argument('--add-round1', dest='add_round1', action='store_const', const=True, default=False,
                            help='Adds first round to the testing - validation round')

        parser.add_argument('--include-all', dest='include_all', action='store_const', const=True, default=False,
                            help='Include all known')

        parser.add_argument('--ref-only', dest='ref_only', action='store_const', const=True, default=False,
                            help='Computes reference statistics')

        parser.add_argument('--only-rounds', dest='only_rounds', nargs=argparse.ZERO_OR_MORE, default=None,
                            help='Only given number of rounds')

        parser.add_argument('--only-crypto', dest='only_crypto', nargs=argparse.ZERO_OR_MORE, default=None,
                            help='Only give crypto')

        parser.add_argument('--narrow', dest='narrow', action='store_const', const=True, default=False,
                            help='Computes only narrow set of functions')

        parser.add_argument('--all-zscores', dest='all_zscores', action='store_const', const=True, default=False,
                            help='All zscore list')

        parser.add_argument('--counters-only', dest='counters_only', action='store_const', const=True, default=False,
                            help='Counter only jobs')

        parser.add_argument('--no-xor-strategy', dest='no_xor_strategy', action='store_const', const=True, default=False,
                            help='Disable XOR strategy')

        parser.add_argument('--no-reinit', dest='no_reinit', action='store_const', const=True, default=False,
                            help='No reinit')

        parser.add_argument('--no-sac', dest='no_sac', action='store_const', const=True, default=False,
                            help='No sac')

        parser.add_argument('--no-rpcs', dest='no_rpcs', action='store_const', const=True, default=False,
                            help='No Random plaintext ciphertext')

        parser.add_argument('--no-counters', dest='no_counters', action='store_const', const=True, default=False,
                            help='No counters')

        parser.add_argument('--inhwr1', dest='inhwr1', action='store_const', const=True, default=False,
                            help='Input HW1 with randomize overflow')
        parser.add_argument('--inhwr2', dest='inhwr2', action='store_const', const=True, default=False,
                            help='Input HW2 with randomize overflow')
        parser.add_argument('--inhwr4', dest='inhwr4', action='store_const', const=True, default=False,
                            help='Input HW4 with randomize overflow')

        parser.add_argument('--enqueue', dest='enqueue', action='store_const', const=True, default=False,
                            help='Enqueues the generated batch after job finishes')

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

