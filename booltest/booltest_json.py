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
from booltest import misc
from booltest import timer
from booltest.booltest_main import *


logger = logging.getLogger(__name__)
coloredlogs.CHROOT_FILES = []
coloredlogs.install(level=logging.DEBUG, use_chroot=False)


# Main - argument parsing + processing
class BooltestJson(Booltest):
    """
    Runs PBSpro jobs
    """
    def __init__(self, *args, **kwargs):
        super(BooltestJson, self).__init__(*args, **kwargs)
        self.tester = None
        self.hw_cfg = None

        self.results_dir = None
        self.job_dir = None
        self.generator_path = None
        self.test_stride = None
        self.test_manuals = None

        self.randomize_tests = True
        self.test_random = random.Random()
        self.test_random.seed(0)

        self.seed = '1fe40505e131963c'
        self.data_to_gen = 0
        self.config_js = None  # generator config
        self.config_data = None  # configuration dict
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

    def all_zscore_process(self, res_file, hwanalysis):
        """
        All-zscore processing, dump to csv
        :param res_file:
        :param hwanalysis:
        :return:
        """
        with open(res_file, 'w+') as fh:
            mean_zscores = [str(x) for x in hwanalysis.all_zscore_means]
            fh.write('means;%s\n' % (';'.join(mean_zscores)))

            for deg in range(1, len(hwanalysis.all_zscore_list)):
                fh.write('zscores-%s;' % deg)
                for idx2 in range(len(hwanalysis.all_zscore_list[deg])):
                    zscore = hwanalysis.all_zscore_list[deg][idx2]
                    if zscore == 0:
                        continue
                    fh.write('%s;' % str(zscore[0]))
                fh.write('\n')
        logger.info('All zscore computed')

    def setup_hwanalysis(self, deg, top_comb, top_k, all_deg, zscore_thresh):
        hwanalysis = HWAnalysis()
        hwanalysis.from_json(self.hw_cfg)
        hwanalysis.reset()
        return hwanalysis

    def work(self, bin_data=None):
        """
        Main entry point - data processing
        :return:
        """
        config = self.config_data

        if self.args.config_file:
            with open(self.args.config_file) as fh:
                config = json.load(fh)

        NRES_TO_DUMP = 128
        self.timer_data_read.reset()
        self.timer_data_bins.reset()
        self.timer_process.reset()
        cpu_pcnt_load_before = misc.try_get_cpu_percent()
        cpu_load_before = misc.try_get_cpu_load()

        hw_cfg = config['hwanalysis']
        test_run = config['config']
        data_file = common.defvalkeys(config, 'spec.data_file')
        skip_finished = common.defvalkey(config, 'skip_finished', False)
        self.do_halving = common.defvalkey(config, 'halving', False)
        self.halving_top = common.defvalkey(config, 'halving_top', NRES_TO_DUMP)
        res_file = common.defvalkey(config, 'res_file')
        backup_dir = common.defvalkey(config, 'backup_dir')
        all_zscores = common.defvalkey(config, 'all_zscores')

        if res_file and self.check_res_file(res_file):
            if skip_finished:
                logger.info('Already computed in %s' % res_file)
                return
            elif backup_dir:
                misc.file_backup(res_file, backup_dir=backup_dir)

        self.hw_cfg = hw_cfg
        self.hwanalysis = HWAnalysis()
        self.hwanalysis.from_json(hw_cfg)
        self.blocklen = self.hwanalysis.blocklen
        self.deg = self.hwanalysis.deg
        self.top_comb = self.hwanalysis.top_comb
        self.top_k = self.hwanalysis.top_k
        self.all_deg = self.hwanalysis.all_deg_compute
        self.zscore_thresh = self.hwanalysis.zscore_thresh
        self.json_nice = True
        self.json_top = min(NRES_TO_DUMP, self.halving_top)

        if all_zscores:
            self.hwanalysis.all_zscore_comp = True

        self.rounds = common.defvalkey(test_run['spec'], 'data_rounds')
        tvsize = common.defvalkey(test_run['spec'], 'data_size')

        # Load input polynomials
        # self.load_input_poly()

        logger.info('Basic settings, deg: %s, blocklen: %s, TV size: %s'
                    % (self.hwanalysis.deg, self.hwanalysis.blocklen, tvsize))
        total_terms = int(common.comb(self.hwanalysis.blocklen, self.hwanalysis.deg, True))

        logger.info('Initializing test')
        time_test_start = time.time()
        self.hwanalysis.init()

        # Process input object
        iobj = None
        if data_file:
            iobj = common.FileInputObject(data_file)
        elif bin_data:
            iobj = common.BinaryInputObject(bin_data)
        else:
            iobj = common.StdinInputObject('stdin')

        size = iobj.size()
        logger.info('Testing input object: %s, size: %d kB' % (iobj, size/1024.0))

        # size smaller than TV? Adapt tv then
        tvsize = self.adjust_tvsize(tvsize, size)

        self.hwanalysis.reset()
        logger.info('BlockLength: %d, deg: %d, terms: %d'
                    % (self.hwanalysis.blocklen, self.hwanalysis.deg, total_terms))

        jscres = []
        with iobj:
            self.analyze_iobj(iobj, 0, tvsize, jscres)

        data_hash = iobj.sha1.hexdigest()
        logger.info('Finished processing %s ' % iobj)
        logger.info('Data read %s ' % iobj.data_read)
        logger.info('Read data hash %s ' % data_hash)

        # All zscore list for statistical processing / theory check
        if all_zscores and res_file:
            self.all_zscore_process(res_file, self.hwanalysis)
            return

        # RESULT process...
        total_results = len(self.hwanalysis.last_res) if self.hwanalysis.last_res else 0
        best_dists = self.hwanalysis.last_res[0:min(NRES_TO_DUMP, total_results)] if self.hwanalysis.last_res else None
        halving_pvals_ok = False

        if self.do_halving and len(jscres) > 1 and 'halvings' in jscres[1] and jscres[1]['halvings']:
            halving_pvals_ok = True

            # Re-sort best distinguishers by the halving ordering
            sorder = self.build_poly_sort_index([common.jsunwrap(x['poly']) for x in jscres[1]['halvings']])
            best_dists.sort(key=lambda x: sorder[common.immutable_poly(common.jsunwrap(x.poly))])

            # Add pvalue from the halving to the best distingushers
            mrange = min(len(jscres[1]['halvings']), len(best_dists))
            best_dists = [(list(best_dists[ix]) + [jscres[1]['halvings'][ix]['pval']]) for ix in range(mrange)]

        best_dists_json = [NoIndent(x) for x in best_dists] if best_dists is not None else None

        jsres = collections.OrderedDict()
        if best_dists:
            jsres['best_zscore'] = best_dists[0][4]  # .zscore
            jsres['best_poly'] = NoIndent(best_dists[0][0])  # .poly
            if halving_pvals_ok:
                jsres['best_pval'] = jscres[1]['halvings'][0]['pval']

        for ix, rr in enumerate(jscres):
            if 'dists' in rr:
                rr['dists'] = [NoIndent(common.jsunwrap(x)) for x in rr['dists']]
            if 'halvings' in rr:
                rr['halvings'] = [NoIndent(common.jsunwrap(x)) for x in rr['halvings']]

        jsres['blocklen'] = self.hwanalysis.blocklen
        jsres['degree'] = self.hwanalysis.deg
        jsres['comb_degree'] = self.hwanalysis.top_comb
        jsres['top_k'] = self.top_k
        jsres['all_deg'] = self.all_deg
        jsres['time_elapsed'] = time.time() - time_test_start
        jsres['time_data_read'] = self.timer_data_read.total()
        jsres['time_data_bins'] = self.timer_data_bins.total()
        jsres['time_process'] = self.timer_process.total()

        jsres['data_hash'] = data_hash
        jsres['data_read'] = iobj.data_read
        jsres['generator'] = self.config_js
        jsres['best_dists'] = best_dists_json
        jsres['config'] = config
        jsres['booltest_res'] = jscres

        if self.dump_cpu_info:
            jsres['hostname'] = misc.try_get_hostname()
            jsres['cpu_pcnt_load_before'] = cpu_pcnt_load_before
            jsres['cpu_load_before'] = cpu_load_before
            jsres['cpu_pcnt_load_after'] = misc.try_get_cpu_percent()
            jsres['cpu_load_after'] = misc.try_get_cpu_load()
            jsres['cpu'] = misc.try_get_cpu_info()

        if res_file:
            with open(res_file, 'w+') as fh:
                fh.write(common.json_dumps(jsres, indent=2))
            misc.try_chmod_gr(res_file)

        return common.jsunwrap(jsres)

    def arg_parser(self):
        parser = argparse.ArgumentParser(description='BoolTest with json in/out')

        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')

        #
        # Testbed related options
        #

        parser.add_argument('--config-file', dest='config_file', default=None,
                            help='JSON config file with assignment')

        parser.add_argument('--data-dir', dest='data_dir', default=None,
                            help='Directory to load data from (precomputed samples to test)')
        return parser

    def parse_args(self, args=None):
        parser = self.arg_parser()
        self.args = parser.parse_args(args)

    def main(self):
        logger.debug('App started')

        self.parse_args()
        jsres = self.work()
        return 0 if jsres is None or jsres['data_read'] > 0 else 3


# Launcher
app = None
if __name__ == "__main__":
    import sys

    app = BooltestJson()
    code = app.main()
    sys.exit(code)

