#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

from queue import Queue, Empty as QEmpty

from booltest import egenerator
from booltest import common
from booltest.booltest_main import *

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


# Main - argument parsing + processing
class TestBatteries(Booltest):
    """
    TestBatteries submits standard crypto functions to standard testing batteries for analysis.
    """
    def __init__(self, *args, **kwargs):
        super(TestBatteries, self).__init__(*args, **kwargs)
        self.args = None
        self.tester = None
        self.input_poly = []

        self.results_dir = None
        self.generator_path = None

        self.seed = '1fe40505e131963c'
        self.data_to_gen = 0
        self.config_js = None
        self.cur_data_file = None  # (tmpdir, config, file)
        self.joq_queue = Queue()
        self.res_map = {}
        self.res_lock = threading.Lock()


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

    def gen_randomdir(self, function, round):
        """
        Generates random directory name
        :return:
        """
        dirname = 'testbed-%s-r%s-%d-%d' % (function, round, int(time.time()), random.randint(0, 2**32-1))
        return os.path.join('/tmp', dirname)

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
                logger.info('File %s exists but is too small' % fpath)
                continue

            return fpath
        return None

    def data_generator(self, tmpdir, function, cur_round, config_js):
        """
        Used to call generator to generate data to test. Prepares data to test.
        If the file has already been generated, just returns the generated file.
        :return:
        """
        data_file = self.find_data_file(function=function, round=cur_round)
        if data_file is not None:
            logger.info('Data file found cached: %s' % data_file)
            return data_file

        # Egenerator procedure: new temp folder, generate config, generate data.
        logger.info('Generating data for %s, round %s to %s' % (function, cur_round, tmpdir))
        data_file = self.eacirc_generator(tmpdir=tmpdir, generator_path=self.generator_path, config_js=config_js)
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
        time.sleep(1)
        p = subprocess.Popen(new_generator_path, shell=True, cwd=tmpdir)
        p.communicate()
        if p.returncode != 0:
            logger.error('Could not generate data, genpath: %s, cwd: %s, code: %s'
                         % (new_generator_path, tmpdir, p.returncode))
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
        battery = collections.OrderedDict()
        battery['AES'] = [3]
        battery['ARIRANG'] = [3]
        battery['AURORA'] = [2]
        battery['BLAKE'] = [1]
        battery['Cheetah'] = [4]
        battery['CubeHash'] = [0]
        battery['DCH'] = [1]
        battery['DECIM'] = [5]
        battery['DynamicSHA2'] = [14]
        battery['ECHO'] = [1]
        battery['Grain'] = [2]
        battery['Grostl'] = [2]
        battery['Hamsi'] = [0]
        battery['JH'] = [6]
        battery['Keccak'] = [3]
        battery['LEX'] = [3]
        battery['Lesamnta'] = [2]
        battery['Luffa'] = [7]
        battery['MD6'] = [9]
        battery['SIMD'] = [0]
        battery['Salsa20'] = [4]
        battery['TEA'] = [4]
        battery['TSC-4'] = [14]
        battery['Tangle'] = [25]
        battery['Twister'] = [6]

        # Another tested functions, not (yet) included in egen.
        battery['MD5'] = [15, 16, 17]
        battery['SHA256'] = [3, 4]
        battery['RC4'] = [1]
        battery['RC4_Col'] = [1]

        # PRNGs
        battery['crand_aisa'] = [1]
        battery['javarand'] = [1]

        return battery

    def worker_main(self, idx):
        """
        Data gen worker method
        :return:
        """
        logger.info('Starting worker %d' % idx)
        while True:
            job = None
            function, cur_round = None, None
            try:
                job = self.joq_queue.get_nowait()
                function, cur_round = job
            except QEmpty:
                break

            try:
                tmpdir = self.gen_randomdir(function, cur_round)
                if self.is_function_egen(function):
                    config_js = egenerator.get_config(function_name=function, rounds=cur_round, data=self.data_to_gen)
                else:
                    config_js = {'algorithm': function, 'round': cur_round, 'seed': self.seed}

                logger.info('Generating %s:%s' % (function, cur_round))
                data_file = self.data_generator(tmpdir=tmpdir, function=function, cur_round=cur_round,
                                                config_js=config_js)

                if data_file is None:
                    logger.error('Data file is invalid')
                    continue

                new_data_file = os.path.join(self.args.results_dir, os.path.basename(data_file))
                if not os.path.exists(new_data_file) or not os.path.samefile(data_file, new_data_file):
                    logger.info("Copying to %s" % new_data_file)
                    shutil.copy(data_file, new_data_file)

                cfgname = 'config_%s_r%d_%04dMB' % (function, cur_round, self.data_to_gen//1024//1024)
                with open(os.path.join(self.args.results_dir, cfgname), 'w') as fh:
                    fh.write(json.dumps(config_js, indent=2))

                with self.res_lock:
                    self.res_map[(function, cur_round)] = (data_file, cfgname, config_js)

                logger.info('Generated %s:%s' % (function, cur_round))

                # Remove test dir
                self.clean_temp_dir(tmpdir)

            except Exception as e:
                logger.error('Exception when computing %s:%s : %s' % (function, cur_round, e))
                logger.debug(traceback.format_exc())
                sys.exit(9)

            finally:
                # Job finished
                self.joq_queue.task_done()

        logger.info('Terminating worker %d' % idx)

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
        battery = self.get_test_battery()
        functions = sorted(list(battery.keys()))
        self.data_to_gen = max(test_sizes_mb) * 1024 * 1024

        logger.info('Battery of functions to test: %s' % battery)
        logger.info('Sizes to test: %s' % test_sizes_mb)

        # Pre-allocate job queue
        for function in functions:
            for cur_round in battery[function]:
                self.joq_queue.put((function, cur_round))

        workers = []
        for wrk in range(self.args.threads):
            logger.info('manager: starting worker %d' % wrk)
            t = threading.Thread(target=self.worker_main, args=(wrk, ))
            t.setDaemon(True)
            t.start()
            workers.append(t)

        # Wait until all datasets are generated
        logger.info('The whole dataset generated')
        self.joq_queue.join()

        # Generate bash script to submit experiments
        bash = '#!/bin/bash\n'

        for function in functions:
            for cur_round in battery[function]:
                for cur_size in self.args.matrix_size:
                    data_file, cfgname, config_js = self.res_map[(function, cur_round)]
                    test_name = '%s_r%d_%04dMB' % (function, cur_round, cur_size)
                    test_file = os.path.join(self.args.script_data, os.path.basename(data_file))
                    line = 'submit_experiment -e %s -n "%s" -c "/home/sample-configs/%dMB.json" -f "%s" -a\n' \
                           % (self.args.email, test_name, cur_size, test_file)
                    bash += line

        bash_path = os.path.join(self.args.results_dir, 'submit.sh')
        with open(bash_path, 'w') as fh:
            fh.write(bash)

        logger.info('Finished')

    def main(self):
        logger.debug('App started')

        parser = argparse.ArgumentParser(description='Test with batteries')
        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')

        parser.add_argument('--verbose', dest='verbose', action='store_const', const=True,
                            help='enables verbose mode')

        #
        # Testbed related options
        #

        parser.add_argument('--threads', dest='threads', default=1, type=int,
                            help='Number of threads to gen data')

        parser.add_argument('--email', dest='email', default=None,
                            help='Email to sends results to')

        parser.add_argument('--generator-path', dest='generator_path', default=None,
                            help='Path to the eacirc generator executable')

        parser.add_argument('--result-dir', dest='results_dir', default=None,
                            help='Directory to put results to')

        parser.add_argument('--data-dir', dest='data_dir', default=None,
                            help='Directory to load data from (precomputed samples to test)')

        parser.add_argument('--script-data', dest='script_data', default=None,
                            help='Directory to load data from - in the benchmark script')

        #
        # Testing matrix definition
        #

        parser.add_argument('--matrix-size', dest='matrix_size', nargs=argparse.ZERO_OR_MORE,
                            default=[1, 10, 100, 1000], type=int,
                            help='List of data sizes to test in MB')

        self.args = parser.parse_args()
        if self.args.debug:
            coloredlogs.install(level=logging.DEBUG)
        self.work()


# Launcher
app = None
if __name__ == "__main__":
    app = TestBatteries()
    app.main()

