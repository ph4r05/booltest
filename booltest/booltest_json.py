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


# Main - argument parsing + processing
class BooltestJson(Booltest):
    """
    Runs PBSpro jobs
    """
    def __init__(self, *args, **kwargs):
        super(BooltestJson, self).__init__(*args, **kwargs)
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

    def work(self):
        """
        Main entry point - data processing
        :return:
        """
        config = None
        with open(self.args.config_file) as fh:
            config = json.load(fh)

        hw_cfg = config['hwanalysis']
        test_run = config['config']
        data_file = test_run['spec']['data_file']
        hwanalysis = HWAnalysis()
        hwanalysis.from_json(hw_cfg)

        rounds = None
        size_mb = test_run['spec']['data_size'] / 1024 / 1024
        tvsize = test_run['spec']['data_size']

        # Load input polynomials
        # self.load_input_poly()

        logger.info('Basic settings, deg: %s, blocklen: %s, TV size: %s' % (hwanalysis.deg, hwanalysis.blocklen, tvsize))
        total_terms = int(scipy.misc.comb(hwanalysis.blocklen, hwanalysis.deg, True))

        logger.info('Initializing test')
        time_test_start = time.time()
        hwanalysis.init()

        # Process input object
        iobj = common.FileInputObject(data_file) if data_file else common.StdinInputObject('stdin')
        size = iobj.size()
        logger.info('Testing input object: %s, size: %d kB' % (iobj, size/1024.0))

        # size smaller than TV? Adapt tv then
        if size >= 0 and size < tvsize:
            logger.info('File size is smaller than TV, updating TV to %d' % size)
            tvsize = size

        if tvsize*8 % hwanalysis.blocklen != 0:
            rem = tvsize*8 % hwanalysis.blocklen
            logger.warning('Input data size not aligned to the block size. '
                           'Input bytes: %d, block bits: %d, rem: %d' % (tvsize, hwanalysis.blocklen, rem))
            tvsize -= rem//8
            logger.info('Updating TV to %d' % tvsize)

        hwanalysis.reset()
        logger.info('BlockLength: %d, deg: %d, terms: %d' % (hwanalysis.blocklen, hwanalysis.deg, total_terms))
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
                            (hwanalysis.deg, hwanalysis.blocklen, tvsize, tvsize/1024.0, tvsize/1024.0/1024.0,
                             cur_round, len(bits)))

                hwanalysis.proces_chunk(bits, None)
                cur_round += 1
            pass

        # RESULT process...
        total_results = len(hwanalysis.last_res) if hwanalysis.last_res else 0
        best_dists = hwanalysis.last_res[0:min(128, total_results)] if hwanalysis.last_res else None
        data_hash = iobj.sha1.hexdigest()

        jsres = collections.OrderedDict()
        if best_dists:
            jsres['best_zscore'] = best_dists[0].zscore
            jsres['best_poly'] = best_dists[0].poly

        jsres['blocklen'] = hwanalysis.blocklen
        jsres['degree'] = hwanalysis.deg
        jsres['comb_degree'] = hwanalysis.top_comb
        jsres['top_k'] = self.top_k
        jsres['all_deg'] = self.all_deg
        jsres['time_elapsed'] = time.time() - time_test_start

        jsres['data_hash'] = data_hash
        jsres['data_read'] = iobj.data_read
        jsres['generator'] = self.config_js
        jsres['best_dists'] = best_dists
        jsres['config'] = config

        res_file_path = config['res_file']
        with open(res_file_path, 'w+') as fh:
            fh.write(common.json_dumps(jsres, indent=2))

        logger.info('Finished processing %s ' % iobj)
        logger.info('Data read %s ' % iobj.data_read)
        logger.info('Read data hash %s ' % data_hash)
        return jsres

    def main(self):
        logger.debug('App started')

        parser = argparse.ArgumentParser(description='Booltest with json in/out')

        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')

        parser.add_argument('--verbose', dest='verbose', action='store_const', const=True,
                            help='enables verbose mode')

        #
        # Testbed related options
        #

        parser.add_argument('--config-file', dest='config_file', default=None,
                            help='JSON config file with assignment')

        parser.add_argument('--data-dir', dest='data_dir', default=None,
                            help='Directory to load data from (precomputed samples to test)')

        self.args = parser.parse_args()
        self.work()


# Launcher
app = None
if __name__ == "__main__":
    app = BooltestJson()
    app.main()

