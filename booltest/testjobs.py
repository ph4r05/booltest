#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import os
import functools
import shutil
import subprocess
import time
import math
import traceback
import copy
import subprocess
import sys
import hashlib
import fnmatch

import scipy.misc
import scipy.stats

if sys.version_info >= (3, 2):
    from functools import lru_cache
else:
    from repoze.lru import lru_cache

from booltest import egenerator
from booltest import common, misc
from booltest.booltest_main import *
from booltest import testjobsbase
from booltest import timer

logger = logging.getLogger(__name__)
coloredlogs.CHROOT_FILES = []
coloredlogs.install(level=logging.DEBUG, use_chroot=False)


job_tpl_hdr = '''#!/bin/bash

export BOOLDIR="/storage/brno3-cerit/home/${LOGNAME}/booltest/assets"
export RESDIR="/storage/brno3-cerit/home/${LOGNAME}/bool-res"
export LOGDIR="/storage/brno3-cerit/home/${LOGNAME}/bool-log"
export SIGDIR="/storage/brno3-cerit/home/${LOGNAME}/bool-sig"

cd "${BOOLDIR}"

'''


class TestCaseEntry(object):
    def __init__(self, fnc, params=None, stream_type=None, data_file=None):
        self.fnc = fnc
        self.params = params
        self.stream_type = stream_type
        self.data_file = data_file
        self.data_rounds = 1
        self.rounds = None
        self.c_round = None
        self.data_size = None
        self.is_egen = False
        self.gen_cfg = None
        self.seed_code = 0
        self.is_randomized = False
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


class TestedObject(object):
    def __init__(self, fnc=None, is_fnc=True, rounds=None, data_file=None, gen_file=None, gen_data=None,
                 res_name=None, gen_size=None):
        self.fnc = fnc
        self.is_fnc = is_fnc
        self.rounds = rounds
        self.data_file = data_file
        self.gen_file = gen_file
        self.gen_data = gen_data
        self.res_name = res_name
        self.gen_size = gen_size


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
        self.mbsep = 1000

        self.time_start = 0
        self.time_file_check = timer.Timer(start=False)
        self.time_json_check = timer.Timer(start=False)
        self.time_gen_total = timer.Timer(start=False)

    def init_params(self):
        """
        Parameter processing
        :return:
        """
        self.mbsep = 1024 if self.args.mibs else 1000

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
        if not self.args.no_functions and self.generator_path is None:
            logger.warning('Generator path is not given, using current directory')
            self.generator_path = os.path.join(os.getcwd(), 'generator')

        if not self.args.no_functions and not os.path.exists(self.generator_path):
            raise ValueError('Generator not found: %s' % self.generator_path)

        # Other
        self.zscore_thresh = self.args.conf

        self.all_deg = self.args.alldeg
        self.test_random.seed(self.args.tests_random_select_eed)

        if self.args.only_rounds:
            self.args.only_rounds = [int(x) for x in self.args.only_rounds]

        if self.args.reseed is not None:
            self.seed = self.args.reseed

        # Default params
        if self.args.default_params:
            self.args.topk = 128
            self.args.no_comb_and = True
            self.args.only_top_comb = True
            self.args.only_top_deg = True
            self.args.no_term_map = True
            self.args.topterm_heap = True
            self.args.topterm_heap_k = 256

    def to_mbs(self, x, full_div=False, ceiling=False):
        if ceiling:
            full_div = False
        rs = (float(x) / (self.mbsep * self.mbsep)) if full_div else (x // (self.mbsep * self.mbsep))
        return int(math.ceil(rs)) if ceiling else int(rs)

    def from_mbs(self, x):
        return x * self.mbsep * self.mbsep

    def check_res_file(self, path):
        """
        Returns True if the given result path seems valid
        :param path:
        :return:
        """
        with self.time_file_check:
            if not os.path.exists(path):
                return False

        if not self.args.check_json:
            return True

        # noinspection PyBroadException
        with self.time_json_check:
            try:
                with open(path, 'r') as fh:
                    js = json.load(fh)
                    json_valid = 'best_dists' in js and 'data_read' in js and js['data_read'] > 0

                    if not json_valid and self.args.remove_broken_result:
                        logger.info('Removing broking result: %s' % path)
                        if not self.args.dry_run:
                            os.remove(path)

                    return json_valid

            except Exception as e:
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
            '%s_r%s_seed%s_%sMB.bin' % (function, round, self.seed, self.to_mbs(size, ceiling=True)),
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

    def get_test_battery(self):
        """
        Returns function -> [r1, r2, r3, ...] to test on given number of rounds.
        :return:
        """
        if self.args.no_functions:
            return {}

        if self.args.ref_only or self.args.all_zscores:
            return {'AES': [10]}

        battery = dict(egenerator.ROUNDS)
        if self.args.only_crypto:
            battery = egenerator.filter_functions(battery, set(self.args.only_crypto))
        elif self.args.narrow:
            battery = egenerator.filter_functions(battery, egenerator.NARROW_SELECTION)
        elif self.args.narrow2:
            battery = egenerator.filter_functions(battery, egenerator.NARROW_SELECTION_EXTPAPER)
        elif self.args.benchmark:
            battery = egenerator.filter_functions(battery, egenerator.BENCHMARK_SELECTION)

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

    def generate_strategies(self, fgc, is_stream, is_sha3, is_block, is_3des, hw_val, hw_key_val):
        """
        Generate strategy to test

        :param fgc:
        :param is_stream:
        :param is_sha3:
        :param is_block:
        :param is_3des:
        :return:
        """
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
        target_iterations = 1 if is_sha3 else self.args.rand_runs
        if self.args.ref_only or self.args.all_zscores:
            target_iterations = 1

        for i in range(target_iterations):
            if is_stream:
                continue

            fun_key = egenerator.get_zero_stream() if i == 0 and not self.args.ref_only else \
                egenerator.get_random_stream()

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
                    egenerator.get_function_config(fgc, src_input=egenerator.get_hw_stream(1, randomize_overflow=True),
                                                   src_key=fun_key),
                ]

            if self.args.inhwr2:
                fun_configs += [
                    egenerator.get_function_config(fgc, src_input=egenerator.get_hw_stream(2, randomize_overflow=True),
                                                   src_key=fun_key),
                ]

            if self.args.inhwr4:
                fun_configs += [
                    egenerator.get_function_config(fgc, src_input=egenerator.get_hw_stream(4, randomize_overflow=True),
                                                   src_key=fun_key),
                ]

        return fun_configs

    def is_narrow(self, fname, narrow_type=0):
        """
        Returns true if function is in the narrow set
        :param fname:
        :param narrow_type:
        :return:
        """
        return egenerator.is_narrow(fname, narrow_type)

    def create_batcher(self):
        batcher = testjobsbase.BatchGenerator()
        batcher.job_dir = self.job_dir
        batcher.aggregation_factor = self.args.aggregation_factor
        batcher.max_hour_job = self.args.max_hr_job
        return batcher

    def rescan_job_files(self):
        """
        Rescans existing jobs, expires them and resubmits for processing if result file is not found.
        :return:
        """
        logger.info('Scanning existing job files dir')
        batcher = self.create_batcher()

        num_gen_file_missing = 0
        num_res_ok = 0
        num_job_running = 0

        for idx, cur_file in enumerate(os.listdir(self.job_dir)):
            cfg_file_path = os.path.join(self.job_dir, cur_file)

            if idx % 1000 == 0:
                logger.debug('Scanning %s %s' % (idx, cfg_file_path))

            if not os.path.isfile(cfg_file_path):
                continue

            # Check if its a configuration file
            if not cur_file.startswith('cfg-'):
                continue

            if self.args.narrow and not self.is_narrow(cur_file):
                continue

            if self.args.narrow2 and not self.is_narrow(cur_file, 1):
                continue

            if self.args.benchmark and not self.is_narrow(cur_file, 2):
                continue

            js_txt = open(cfg_file_path, 'r').read()
            js = json.loads(js_txt)
            res_file = common.defvalkey(js, 'res_file')
            gen_file = common.defvalkey(js, 'gen_file')
            skip_finished = common.defvalkey(js, 'skip_finished')
            fnc = js['config']['spec']['fnc']

            if not os.path.exists(gen_file):
                num_gen_file_missing += 1
                continue

            if self.check_res_file(res_file):
                num_res_ok += 1
                continue

            if not self.args.overwrite_existing and self.args.expiring and not self.is_job_expired(cfg_file_path):
                num_job_running += 1
                continue

            # Transform not to overwrite existing result.
            if not skip_finished:
                js['skip_finished'] = True
                with open(cfg_file_path, 'w') as fh:
                    fh.write(common.json_dumps(js, indent=2))

            unit = self.create_batch_unit_js(cfg_file_path, js)
            batcher.add_unit(unit)
        batcher.flush()

        logger.info('Generated job files: %s, tests: %s, num missing gen: %s, num ok: %s, num running: %s'
                    % (len(batcher.job_files), batcher.num_units, num_gen_file_missing, num_res_ok, num_job_running))

        self.finalize_batch(batcher)

    def create_batch_unit_js(self, config_file, spec):
        """
        Batching unit from generated config - reruning expired jobs
        :param config_file:
        :param spec:
        :return:
        """
        unit = testjobsbase.TestBatchUnit()
        unit.cfg_file_path = config_file
        unit.res_file_path = spec['res_file']
        unit.gen_file_path = spec['gen_file']
        unit.res_file = os.path.basename(unit.res_file_path)
        unit.block_size = int(spec['hwanalysis']['blocklen'])
        unit.degree = int(spec['hwanalysis']['deg'])
        unit.comb_deg = int(spec['hwanalysis']['blocklen'])
        unit.data_size = int(spec['config']['spec']['data_size'])
        unit.size_mb = self.to_mbs(unit.data_size, ceiling=True)
        return unit

    def create_batch_unit_trun(self, trun):
        """
        Batch unit from test run
        :param trun:
        :return:
        """
        # TODO:

    def compute_batch(self, jobs):
        """
        Computes job batch for GRID / Metacentrum from batch units of work
        :param jobs: can be generator / iterator
        :return:
        """
        batcher = self.create_batcher()
        for unit in jobs:
            batcher.add_unit(unit)
        batcher.flush()

        logger.info('Generated job files: %s, tests: %s' % (len(batcher.job_files), batcher.num_units))
        self.finalize_batch(batcher)

    def finalize_batch(self, batcher):
        """
        Creates final enqueueing scripts for the batch
        :param batcher:
        :type batcher: testjobsbase.BatchGenerator
        :return:
        """
        # Enqueue
        enqueue_path = os.path.join(self.job_dir, 'enqueue-meta-%s.sh' % int(time.time()))
        with open(enqueue_path, 'w') as fh:
            fh.write('#!/bin/bash\n\n')
            qsub_args = []
            if self.args.brno:
                qsub_args.append('brno=True')
            if self.args.cluster:
                qsub_args.append('cl_%s=True' % self.args.cluster)

            nprocs = self.args.qsub_ncpu
            qsub_args = ':'.join(qsub_args)
            qsub_args = (':%s' % qsub_args) if qsub_args != '' else ''
            for fn in batcher.job_files:
                fh.write('qsub -l select=1:ncpus=%s:mem=%s%s -l walltime=%s %s \n' % (nprocs, fn[1], qsub_args, fn[2], fn[0]))

        # Generator tester file
        testgen_path = os.path.join(self.job_dir, 'test-generators-%s.sh' % int(time.time()))
        with open(testgen_path, 'w') as fh:
            fh.write(job_tpl_hdr)
            for fn in sorted(list(batcher.generator_files)):
                if not os.path.exists(fn):
                    continue
                fh.write('./generator-metacentrum.sh -c=%s 2>/dev/null >/dev/null\n' % fn)
                fh.write('if [ $? -ne 0 ]; then echo "Generator failed: %s"; else echo -n "."; fi\n' % fn)
            fh.write('\n')

        # chmod
        misc.try_chmod_grx(enqueue_path)
        misc.try_chmod_grx(testgen_path)
        logger.info('Gentest: %s' % testgen_path)
        logger.info('Enqueue: %s' % enqueue_path)

        if self.args.enqueue:
            logger.info('Enqueueing...')
            p = subprocess.Popen(enqueue_path, stdout=sys.stdout, stderr=sys.stderr, shell=True)
            p.wait()

    def unpack_prng(self, data_dir, out_dir):
        # misc.unpack_keys()
        pass

    @lru_cache(maxsize=4096)
    def file_data_size(self, fname):
        st = os.stat(fname)
        return st.st_size

    def find_generator_files(self):
        """
        Finds all generator files specified by args
        :return:
        """
        for cur_fold in self.args.generator_folder:
            for root, dirnames, filenames in os.walk(cur_fold):
                for filename in fnmatch.filter(filenames, '*.json'):
                    yield (os.path.join(root, filename), cur_fold)

    def recursive_algorithm_search(self, js):
        """
        Finds first algorithm in the JS hierarchy
        :param js:
        :return:
        """
        if isinstance(js, list):
            for xr in js:
                res = self.recursive_algorithm_search(xr)
                if res is not None:
                    return res

        elif isinstance(js, dict):
            if 'algorithm' in js:
                return js

            for key in js:
                res = self.recursive_algorithm_search(js[key])
                if res is not None:
                    return res
        return None

    def change_seed(self, js):
        """
        Changes seed in the generator file if required
        :param js:
        :return:
        """
        if self.args.reseed is None:
            return

        js['seed'] = self.args.reseed
        return js

    def read_generator_files(self, acc):
        """
        Recursivelly finds all generator files in the folders and run them
        :param acc:
        :return:
        """
        for cfile, root in self.find_generator_files():
            dirname = os.path.abspath(os.path.dirname(cfile))
            absroot = os.path.abspath(root)
            config_dir = dirname.replace(absroot, '')
            exp_name = config_dir.replace('/', '-')
            exp_name = exp_name.replace('.', '')

            bname = os.path.splitext(os.path.basename(cfile))[0]
            with open(cfile) as fh:
                js = json.load(fh)

            js['stdout'] = True
            self.change_seed(js)

            js_stream = self.recursive_algorithm_search(js)
            if js_stream is None:
                raise ValueError('Could not find generating algorithm in %s' % cfile)

            fnc = common.defvalkey(js_stream, 'algorithm')
            if fnc is None:
                raise ValueError('Generator %s has no algorithm' % cfile)

            round = common.defvalkey(js_stream, 'round', 1)
            cfg_hash = hashlib.sha1(common.json_dumps(js).encode('utf8')).hexdigest()[12:]
            gen_size = js['tv_size'] * js['tv_count']
            gen_size_mb = self.to_mbs(gen_size, ceiling=True)
            res_name = '%s-r%s-e%s-cfg%s' % (fnc, round, exp_name, cfg_hash)
            logger.debug('Generator found, fnc %s, round %s, cfile %s, res_name %s, gen_size %s MB (%s B), exp_name %s'
                         % (fnc, round, cfile, res_name, gen_size_mb, gen_size, exp_name))
            acc.append(TestedObject(fnc=fnc, rounds=[round], is_fnc=False, gen_file=cfile, res_name=res_name,
                                    gen_data=js, gen_size=gen_size))
        pass

    # noinspection PyBroadException
    def work(self):
        """
        Main entry point - data processing
        :return:
        """
        if self.args.unpack:
            self.unpack_prng(self.args.data_dir, out_dir=self.args.res_dir)
            return

        self.init_params()
        self.time_start = time.time()

        # Init logic, analysis.
        # Define test set.
        test_sizes_mb = self.args.matrix_size
        test_sizes_bytes = [self.from_mbs(float(size_mb)) for size_mb in test_sizes_mb]
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
                combinations[bl][ii] = int(common.comb(bl, ii, True))
        full_combination = combinations[max(test_block_sizes)][max(max(test_degree), max(test_comb_k))]

        if self.args.rescan_jobs:
            self.rescan_job_files()
            return

        # (function, round, processing_type)
        test_array = []
        total_test_idx = 0

        # Process arguments to object for testing
        tested_objects = []
        for fnc in functions:
            tested_objects.append(TestedObject(fnc=fnc, rounds=battery[fnc], is_fnc=True))

        # Generator files
        self.read_generator_files(tested_objects)

        # Raw files generated for testing
        for fl in self.args.test_files:
            if not os.path.exists(fl):
                raise ValueError('File does not exist: %s' % fl)
            bname = misc.normalize_card_name(os.path.basename(fl))
            tested_objects.append(TestedObject(fnc=bname, rounds=[1], is_fnc=False, data_file=fl))

        # Process objects for testing
        for tested_obj in tested_objects:
            fnc = tested_obj.fnc
            rounds = tested_obj.rounds

            # Validation round 1
            if tested_obj.is_fnc and self.args.add_round1 and 1 not in rounds:
                rounds.insert(0, 1)

            params = all_functions[fnc] if fnc in all_functions else None  # type: egenerator.FunctionParams
            tce = TestCaseEntry(fnc, params)
            tce.rounds = rounds
            is_3des = egenerator.is_3des(fnc)  # special key handling, cannot use ctr/hw

            if egenerator.is_function_egen(fnc):
                fnc = egenerator.normalize_function_name(fnc)
                tce.fnc = fnc
                tce.stream_type = egenerator.function_to_stream_type(fnc)
                tce.is_egen = True

            test_sizes_cur = test_sizes_bytes
            if tested_obj.gen_size:  # fixed generator size
                test_sizes_cur = [tested_obj.gen_size]

            iterator = itertools.product(rounds, test_sizes_cur)
            for cur_round, size_bytes in iterator:
                tce_c = copy.deepcopy(tce)
                tce_c.c_round = cur_round
                tce_c.data_size = size_bytes

                if tested_obj.gen_file:
                    tce_c.gen_cfg = tested_obj.gen_data
                    tce_c.strategy = tested_obj.res_name

                elif not tested_obj.is_fnc:
                    tce_c.data_file = os.path.abspath(tested_obj.data_file)
                    tce_c.strategy = '%s-static' % fnc

                elif not tce.is_egen:
                    tce_c.data_file = self.find_data_file(function=tce.fnc, round=cur_round, size=tce_c.data_size)
                    tce_c.strategy = '%s-static' % misc.normalize_card_name(os.path.basename(tce_c.data_file))

                if tce_c.data_file:
                    if self.file_data_size(tce_c.data_file) < tce_c.data_size:
                        continue

                    test_array.append(tce_c)
                    continue

                if tested_obj.gen_file:
                    test_array.append(tce_c)
                    continue

                is_sha3 = tce.stream_type in [egenerator.FUNCTION_SHA3, egenerator.FUNCTION_HASH]
                is_stream = tce.stream_type == egenerator.FUNCTION_ESTREAM
                is_block = tce.stream_type == egenerator.FUNCTION_BLOCK
                use_only_strategy = self.args.only_strategy

                blk_lower_than_16b = params and params.block_size and params.block_size < 16
                key_lower_than_16b = params and params.key_size and params.key_size < 16
                hw_val = 4 if not blk_lower_than_16b else 6
                hw_key_val = 4 if not key_lower_than_16b else 6

                fgc = egenerator.FunctionGenConfig(fnc, rounds=cur_round, data=tce_c.data_size, params=tce_c.params)

                fun_configs = []
                if not use_only_strategy:
                    fun_configs += self.generate_strategies(fgc, is_stream, is_sha3, is_block, is_3des, hw_val, hw_key_val)

                else:
                    for c_strategy in self.args.only_strategy:
                        c_fun_config = egenerator.determine_strategy(c_strategy, fgc=fgc)
                        fun_configs.append(c_fun_config)

                for fun_cfg in fun_configs:
                    seed = common.generate_seed()
                    tce_c.gen_cfg = egenerator.get_config_header(fgc, stdout=True, stream=fun_cfg, seed=seed)
                    tce_c.gen_cfg['exp_time'] = self.time_experiment
                    tce_c.is_randomized = egenerator.is_randomized(tce_c.gen_cfg['stream'])
                    tce_c.strategy = egenerator.get_scode(fun_cfg)
                    test_array.append(copy.deepcopy(tce_c))

        # test multiple booltest params
        test_runs = []  # type: list[TestRun]
        logger.info('Test array size: %s' % len(test_array))

        # Generate test cases, run the analysis.
        test_runs_times = range(self.args.test_rand_runs)

        # For each test specification
        generator_files = set()
        for test_spec in test_array:  # type: TestCaseEntry

            for test_case in itertools.product(test_block_sizes, test_degree, test_comb_k, test_runs_times):
                block_size, degree, comb_deg, trt = test_case
                data_size = self.to_mbs(test_spec.data_size, ceiling=True)
                total_test_idx += 1

                if trt > 0:
                    if not test_spec.is_randomized:
                        continue

                    test_spec = copy.deepcopy(test_spec)
                    test_spec.seed_code = trt
                    seed = common.generate_seed(trt)
                    test_spec.gen_cfg['seed'] = seed
                    test_spec.gen_cfg['seed_code'] = test_spec.seed_code

                test_desc = 'idx: %04d, data: %04d, block: %s, deg: %s, comb-deg: %s, fun: %s, round: %s scode: %s, %s' \
                            % (total_test_idx, data_size, block_size, degree, comb_deg, test_spec.fnc,
                               test_spec.c_round, test_spec.strategy, trt)

                suffix = 'json' if not self.args.all_zscores else 'csv'
                test_type = '' if not self.args.all_zscores else '-zscores'
                opmode = '' if not self.args.halving else 'hlv-'
                res_file = '%s%s-%04dMB-%sbl-%sdeg-%sk%s%s.%s' \
                           % (opmode, test_spec.strategy, data_size, block_size, degree, comb_deg,
                              ('-%s' % trt) if trt > 0 else '', test_type, suffix)
                res_file = res_file.replace(' ', '')

                gen_file = None
                if test_spec.gen_cfg:
                    gen_file = 'gen-%s-%04dMB-%s.json' % (test_spec.strategy, data_size, trt)
                    gen_file = gen_file.replace(' ', '')

                    if test_spec.data_size == test_sizes_bytes[0]:
                        generator_files.add(os.path.join(self.job_dir, gen_file))

                trun = TestRun(test_spec, block_size, degree, comb_deg, total_test_idx, test_desc, res_file, gen_file)
                trun.iteration = trt
                test_runs.append(trun)

        # Sort by estimated complexity
        test_runs.sort(key=lambda x: (x.spec.data_size, x.degree, x.block_size, x.comb_deg))
        logger.info('Total num of tests: %s' % len(test_runs))

        # Load input polynomials
        self.load_input_poly()

        # Load jobs if scheduling matters
        logger.info('Getting running and scheduled jobs...')
        cjobs, jobs_scheduled = misc.get_jobs_in_progress() if self.args.skip_scheduled else {}, {}
        if self.args.skip_scheduled:
            logger.info('Jobs scheduled: %s' % len(jobs_scheduled))

        # Generate job files
        batcher = self.create_batcher()
        logger.info('Batching info: agg fact: %s, max hour job: %s, job dir: %s'
                    % (batcher.aggregation_factor, batcher.max_hour_job, batcher.job_dir))
        
        num_skipped = 0
        num_skipped_existing = 0
        num_skipped_scheduled = 0
        self.time_gen_total.start()

        for fidx, trun in enumerate(test_runs):  # type: tuple(int, TestRun)
            hwanalysis = self.testcase(trun.block_size, trun.degree, trun.comb_deg)
            json_config = collections.OrderedDict()
            json_config['exp_time'] = self.time_experiment
            json_config['config'] = trun
            json_config['hwanalysis'] = hwanalysis
            json_config['fidx'] = fidx

            job_fname = 'job-%s.sh' % trun.res_file
            res_file_path = os.path.join(self.results_dir, trun.res_file)
            gen_file_path = os.path.join(self.job_dir, trun.gen_file) if trun.gen_file else None
            cfg_file_path = os.path.join(self.job_dir, 'cfg-' + trun.res_file)

            json_config['res_file'] = res_file_path
            json_config['gen_file'] = gen_file_path
            json_config['backup_dir'] = self.args.backup_dir
            json_config['skip_finished'] = self.args.skip_finished
            json_config['all_zscores'] = self.args.all_zscores
            json_config['halving'] = self.args.halving
            json_config['halving_top'] = self.args.halving_top

            if fidx % 1000 == 0:
                logger.debug('Processing file %s, jobs: %s, time: %s, timef: %s, timej: %s'
                             % (fidx, len(batcher.job_files),
                                self.time_gen_total.cur(),
                                self.time_file_check.cur(),
                                self.time_json_check.cur()
                                ))

            if self.args.skip_finished and self.check_res_file(res_file_path):
                num_skipped += 1
                continue

            if self.args.skip_scheduled and job_fname in jobs_scheduled:
                num_skipped_scheduled += 1
                continue

            job_expired = False
            if os.path.exists(cfg_file_path):
                if self.args.skip_existing:
                    if self.args.expiring and self.is_job_expired(cfg_file_path):
                        job_expired = True
                        logger.debug('Job expired: %s' % cfg_file_path)

                    else:
                        num_skipped_existing += 1
                        continue

                if self.args.overwrite_existing:
                    logger.debug('Overwriting %s' % cfg_file_path)

                elif not job_expired and not self.args.ignore_existing:
                    logger.warning('Conflicting config: %s' % common.json_dumps(json_config, indent=2))
                    logger.debug('Conflicting file: %s ' % common.try_json_dumps(
                        common.try_json_load(open(cfg_file_path, 'r').read())))
                    raise ValueError('File name conflict: %s, test idx: %s' % (cfg_file_path, fidx))

            if gen_file_path and (not os.path.exists(gen_file_path) or self.args.overwrite_existing):
                with open(gen_file_path, 'w+') as fh:
                    fh.write(common.json_dumps(trun.spec.gen_cfg, indent=2))
                misc.try_chmod_gr(gen_file_path)

            with open(cfg_file_path, 'w+') as fh:
                fh.write(common.json_dumps(json_config, indent=2))

            # Job batch creation
            unit = testjobsbase.TestBatchUnit()
            unit.cfg_file_path = cfg_file_path
            unit.gen_file_path = gen_file_path
            unit.res_file_path = res_file_path
            unit.res_file = trun.res_file
            unit.block_size = trun.block_size
            unit.degree = trun.degree
            unit.comb_deg = trun.comb_deg
            unit.data_size = trun.spec.data_size
            unit.size_mb = self.to_mbs(trun.spec.data_size, ceiling=True)
            batcher.add_unit(unit)
        batcher.flush()

        logger.info('Generated job files: %s, tests: %s, skipped: %s, skipped existing: %s, skipped scheduled: %s'
                    % (len(batcher.job_files), batcher.num_units, num_skipped, num_skipped_existing, num_skipped_scheduled))
        logger.info('Time elapsed: %s s' % (time.time() - self.time_start))

        self.finalize_batch(batcher)

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

        parser = argparse.ArgumentParser(description='Generates a job matrix of tests on multiple possible functions, used with PBSPro')

        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')

        parser.add_argument('--verbose', dest='verbose', action='store_const', const=True,
                            help='enables verbose mode')

        parser.add_argument('--top', dest='topk', default=128, type=int,
                            help='top K number of the best distinguishers to select to the combination phase')

        parser.add_argument('--comb-rand', dest='comb_random', default=0, type=int,
                            help='number of terms to add randomly to the combination set')

        parser.add_argument('--conf', dest='conf', type=float, default=1.96,
                            help='Zscore failing threshold')

        parser.add_argument('--alldeg', dest='alldeg', action='store_const', const=True, default=False,
                            help='Add top K best terms to the combination phase also for lower degree, not just the top one')

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
                            help='If set, only the comb-degree combination is performed, otherwise all combinations up to given comb-degree')

        parser.add_argument('--only-top-deg', dest='only_top_deg', action='store_const', const=True, default=False,
                            help='If set, only the top degree of 1st stage polynomials are evaluated (zscore is computed), otherwise '
                                 'also lower degrees are input to the topk for next state - combinations')

        parser.add_argument('--no-term-map', dest='no_term_map', action='store_const', const=True, default=False,
                            help='Disables term map precomputation (memory-heavy mapping idx->term), uses unranking algorithm instead')

        parser.add_argument('--prob-comb', dest='prob_comb', type=float, default=1.0,
                            help='Probability the given combination is going to be chosen. Enables stochastic test, useful for large degrees')

        parser.add_argument('--topterm-heap', dest='topterm_heap', action='store_const', const=True, default=False,
                            help='Use heap to compute best K terms for stats & input to the combinations')

        parser.add_argument('--topterm-heap-k', dest='topterm_heap_k', default=None, type=int,
                            help='Number of terms to keep in the heap, should be at least top_k')

        parser.add_argument('--best-x-combs', dest='best_x_combinations', default=None, type=int,
                            help='Number of best combinations to return. If defined, heap is used')

        parser.add_argument('--default-params', dest='default_params', action='store_const', const=True, default=False,
                            help='Default parameter settings for testing, used in the paper')

        parser.add_argument('--halving', dest='halving', action='store_const', const=True, default=False,
                            help='Pick the best distinguisher on the first half, evaluate on the second half')

        parser.add_argument('--halving-top', dest='halving_top', type=int, default=30,
                            help='Number of top distinguishers to select to the halving phase')

        #
        # Testbed related options
        #

        parser.add_argument('--generator-path', dest='generator_path', default=None,
                            help='Path to the CryptoStreams generator executable')

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

        parser.add_argument('--rand-runs', dest='rand_runs', default=2, type=int,
                            help='Number of random runs')

        parser.add_argument('--test-rand-runs', dest='test_rand_runs', default=1, type=int,
                            help='Number of random runs for the whole tests')

        parser.add_argument('--egen-benchmark', dest='egen_benchmark', action='store_const', const=True, default=False,
                            help='Benchmarks speed of the egenerator')

        parser.add_argument('--skip-existing', dest='skip_existing', action='store_const', const=True, default=False,
                            help='Skip existing jobs')

        parser.add_argument('--expiring', dest='expiring', action='store_const', const=True, default=False,
                            help='Skip existing jobs - but check for expiration on result, did job finished?')

        parser.add_argument('--overwrite-existing', dest='overwrite_existing', action='store_const', const=True, default=False,
                            help='Overwrites existing jobs')

        parser.add_argument('--ignore-existing', dest='ignore_existing', action='store_const', const=True, default=False,
                            help='Ignores existing jobs')

        parser.add_argument('--skip-finished', dest='skip_finished', action='store_const', const=True, default=False,
                            help='Skip tests with generated valid results')

        parser.add_argument('--skip-scheduled', dest='skip_scheduled', action='store_const', const=True, default=False,
                            help='Skip tests scheduled or running')

        parser.add_argument('--add-round1', dest='add_round1', action='store_const', const=True, default=False,
                            help='Adds first round to the testing - validation round')

        parser.add_argument('--include-all', dest='include_all', action='store_const', const=True, default=False,
                            help='Include all known')

        parser.add_argument('--ref-only', dest='ref_only', action='store_const', const=True, default=False,
                            help='Computes reference statistics')

        parser.add_argument('--only-rounds', dest='only_rounds', nargs=argparse.ZERO_OR_MORE, default=None,
                            help='Only given number of rounds')

        parser.add_argument('--only-crypto', dest='only_crypto', nargs=argparse.ZERO_OR_MORE, default=None,
                            help='Only given crypto')

        parser.add_argument('--only-strategy', dest='only_strategy', nargs=argparse.ZERO_OR_MORE, default=None,
                            help='Only given strategy')

        parser.add_argument('--no-functions', dest='no_functions', action='store_const', const=True, default=False,
                            help='Do not test any functions (e.g., only data files)')

        parser.add_argument('--narrow', dest='narrow', action='store_const', const=True, default=False,
                            help='Computes only narrow set of functions')

        parser.add_argument('--narrow2', dest='narrow2', action='store_const', const=True, default=False,
                            help='Computes only narrow2 set of functions')

        parser.add_argument('--benchmark', dest='benchmark', action='store_const', const=True, default=False,
                            help='Computes only benchmark set of functions')

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

        parser.add_argument('--brno', dest='brno', action='store_const', const=True, default=False,
                            help='qsub: Enqueue on Brno clusters')

        parser.add_argument('--cluster', dest='cluster', default=None,
                            help='qsub: Enqueue on specific cluster name, e.g., brno, elixir')

        parser.add_argument('--qsub-ncpu', dest='qsub_ncpu', default=1, type=int,
                            help='qsub:  Number of processors to allocate for a job')

        parser.add_argument('--aggregation-factor', dest='aggregation_factor', default=1.0, type=float,
                            help='Job aggregation factor, changes number of tests in one job file')

        parser.add_argument('--max-hr-job', dest='max_hr_job', default=24, type=int,
                            help='The biggest job to allocate on the cluster in hours')

        parser.add_argument('--enqueue', dest='enqueue', action='store_const', const=True, default=False,
                            help='Enqueues the generated batch via qsub after job finishes')

        parser.add_argument('--rescan-jobs', dest='rescan_jobs', action='store_const', const=True, default=False,
                            help='Rescans job dir for configured but expired jobs')

        parser.add_argument('--unpack', dest='unpack', action='store',
                            help='Unpack card keys from archive')

        parser.add_argument('--test-files', dest='test_files', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='Files with input data to test')

        parser.add_argument('--mibs', dest='mibs', action='store_const', const=True, default=False,
                            help='Use mibs - 2^x basis')

        parser.add_argument('--generator-folder', dest='generator_folder', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='Folders with CryptoStreams config files to run')

        parser.add_argument('--reseed', dest='reseed', default=None,
                            help='Sets seed to the defined value')

        parser.add_argument('--dry-run', dest='dry_run', action='store_const', const=True, default=False,
                            help='Dry run')

        parser.add_argument('--check-json', dest='check_json', default=1, type=int,
                            help='Check json file validity. Reads json file, parses and checks for validity')

        parser.add_argument('--remove-broken-json', dest='remove_broken_result', default=0, type=int,
                            help='Removes invalid json results')

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


app = None


def main():
    global app
    app = Testjobs()
    app.main()


if __name__ == "__main__":
    main()
