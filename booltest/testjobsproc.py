#!/usr/bin/env python
# -*- coding: utf-8 -*-
# graphing dependencies: matplotlib, seaborn, pandas

__author__ = 'dusanklinec'

from past.builtins import cmp
import argparse
import fileinput
import json
import time
import re
import os
import copy
import shutil
import hashlib
import sys
import collections
import itertools
import traceback
import logging
import math
import coloredlogs

from booltest import common, egenerator, timer

logger = logging.getLogger(__name__)
coloredlogs.CHROOT_FILES = []
coloredlogs.install(level=logging.DEBUG, use_chroot=False)


# Method used for generating reference looking data stream
REFERENCE_METHOD = 'inctr-krnd-ri0'


class Checkpoint(object):
    def __init__(self, **kwargs):
        self.all_records = {}  # primary caching DB
        self.args = None
        self.time = time.time()

        # Secondary data for re-cache, not checkpointed as it is recreated from all_records
        self.test_records = []
        self.total_functions = []
        self.ref_bins = collections.defaultdict(lambda: [])
        self.timing_bins = collections.defaultdict(lambda: [])
        self.skipped = 0
        self.invalid_results = []
        self.invalid_results_num = 0

        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    def get_cached(self, bname):
        return self.all_records[bname] if bname in self.all_records else None

    def get_cached_keys(self):
        return set(self.all_records.keys())

    def add_record(self, tr):
        if tr is None or tr.bname in self.all_records:
            return
        self.all_records[tr.bname] = tr

    def recache(self):
        for tr in self.test_records:
            self.add_record(tr)

    def to_json(self):
        dct = dict(self.__dict__)
        dct['args'] = args_to_dict(self.args) if not isinstance(self.args, dict) else self.args
        dct['ref_bins'] = {}  # {json.dumps(x): self.ref_bins[x] for x in self.ref_bins}
        dct['timing_bins'] = {}  # {json.dumps(x): self.timing_bins[x] for x in self.timing_bins}
        dct['total_functions'] = list(self.total_functions)
        dct['test_records'] = []
        return dct

    def from_json(self, data):
        for k in data:
            setattr(self, k, data[k])

        # Migration
        if len(self.all_records) == 0:
            self.all_records = {x['bname']: x for x in self.test_records}
            logger.info('Checkpoint migrated')

        self.all_records = {x: TestRecord.new_from_json(self.all_records[x]) for x in self.all_records}

        self.total_functions = set(self.total_functions)
        self.test_records = [TestRecord.new_from_json(x) for x in self.test_records]

        ref_data = {tuple(json.loads(x)): [TestRecord.new_from_json(y) for y in self.ref_bins[x]] for x in self.ref_bins}
        self.ref_bins = collections.defaultdict(lambda: [], ref_data)

        timing_data = {tuple(json.loads(x)): self.timing_bins[x] for x in self.timing_bins}
        self.timing_bins = collections.defaultdict(lambda: [], timing_data)


class TestRecord(object):
    """
    Represents one performed test and its result.
    """
    def __init__(self, **kwargs):
        self.function = None
        self.round = None
        self.block = None
        self.deg = None
        self.comb_deg = None
        self.data = None
        self.data_bytes = None
        self.elapsed = None
        self.iteration = 0
        self.strategy = None
        self.method = None
        self.ref = False
        self.time_process = None
        self.bname = None
        self.fhash = None
        self.mtime = None
        self.cfg_file_name = None

        self.zscore = None
        self.best_poly = None

        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    def __cmp__(self, other):
        """
        Compare: function, round, data, block, deg, k.
        :param other:
        :return:
        """
        a = (self.function, self.round, self.data, self.block, self.deg, self.comb_deg)
        b = (other.function, other.round, other.data, other.block, other.deg, other.comb_deg)
        return cmp(a, b)

    def method_unhw(self):
        return re.sub(r'hw[0-9]+[rsi]{0,3}', 'hw', self.method)

    def method_generic(self):
        return REFERENCE_METHOD

    def __repr__(self):
        return '%s-r%d-d%s_bl%d-deg%d-k%d' % (self.function, self.round, self.data, self.block, self.deg, self.comb_deg)

    def ref_category(self):
        return self.method, self.block, self.deg, self.comb_deg, self.data

    def bool_category(self):
        return self.block, self.deg, self.comb_deg, self.data

    def ref_category_unhw(self):
        return self.method_unhw(), self.block, self.deg, self.comb_deg, self.data

    def ref_category_generic(self):
        return self.method_generic(), self.block, self.deg, self.comb_deg, self.data

    def to_json(self):
        return self.__dict__

    def from_json(self, data):
        for k in data:
            setattr(self, k, data[k])

    @classmethod
    def new_from_json(cls, data):
        r = cls()
        r.from_json(data)
        return r


class PvalDb(object):
    def __init__(self, fname=None):
        self.fname = fname
        self.data = None
        self.map = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: None
                )
            ))

    def load(self):
        if self.fname is None:
            return
        self.data = json.load(open(self.fname))
        for rec in self.data:
            # pvalrec = rec['pvals'][0]
            self.map[rec['block']][rec['deg']][rec['comb_deg']] = rec['extremes']

    def eval(self, block, deg, cdeg, zscore):
        if self.map[block][deg][cdeg] is None:
            return None

        minv, maxv = self.map[block][deg][cdeg][0][0], self.map[block][deg][cdeg][1][0]
        return abs(zscore) < minv or abs(zscore) > maxv


def args_to_dict(args):
    return {x: getattr(args, x, None) for x in args.__dict__} if args else None


def get_method(strategy):
    """
    Parses method from the strategy
    :param strategy:
    :return:
    """
    # strip booltest params
    method = re.sub(r'[\d]{1,4}MB-[\d]{3}bl-[\d]deg-[\d]k(-\d+)?', '', strategy)
    # strip function dependent info
    method = re.sub(r'tp[\w]+-[\w-]+?-r\d+-tv\d+', '', method)
    method = re.sub(r'^[-]+', '', method)
    method = re.sub(r'[-]+$', '', method)
    # strip krnd iteration
    method = method.replace('krnd0', 'krnd')
    method = method.replace('krnd1', 'krnd')
    method = method.replace('krnd-1', 'krnd')
    method = re.sub(r'-krnd[0-9]+-$', 'krnd', method)
    method = method.replace('--', '-')
    if method.endswith('-static'):
        return 'static'
    return method


def process_file(js, fname, args=None):
    """
    Process file json
    :param js:
    :param fname:
    :param args:
    :return:
    """
    tr = TestRecord()
    tr.zscore = common.defvalkey(js, 'best_zscore')
    if tr.zscore:
        tr.zscore = round(tr.zscore, 6)
        if args.zscore_shape:
            tr.zscore = int(abs(round(tr.zscore)))

    tr.best_poly = common.defvalkey(js, 'best_poly')
    tr.function = common.defvalkeys(js, 'config.config.spec.fnc')
    tr.round = common.defvalkeys(js, 'config.config.spec.c_round')
    tr.data = common.defvalkeys(js, 'config.config.spec.data_size')
    tr.strategy = common.defvalkeys(js, 'config.config.spec.strategy')
    tr.method = get_method(tr.strategy)
    tr.time_process = common.defvalkeys(js, 'time_process')
    tr.cfg_file_name = common.defvalkeys(js, 'config.config.spec.gen_cfg.file_name')
    tr.data_bytes = common.defvalkeys(js, 'data_read')

    if tr.data:
        tr.data = int(math.ceil(math.ceil(tr.data/1024.0)/1024.0))

    tr.bname = fname
    mtch = re.search(r'-(\d+)\.json$', fname)
    if mtch:
        tr.iteration = int(mtch.group(1))

    # if 'stream' in js['generator']:
    #     tr.function = js['generator']['stream']['algorithm']
    #     tr.round = js['generator']['stream']['round']
    #
    # else:
    #     tr.function = js['generator']['algorithm']
    #     tr.round = js['generator']['round']

    tr.block = js['blocklen']
    tr.deg = js['degree']
    tr.comb_deg = js['comb_degree']

    # if 'elapsed' in js:
    #     tr.elapsed = js['elapsed']

    return tr


def fls(x):
    """
    Converts float to string, replacing . with , - excel separator
    :param x:
    :return:
    """
    return str(x).replace('.', ',')


def get_ref_value(ref_avg, tr):
    """
    Returns reference value closest to the test.
    Fallbacks to the generic
    :param ref_avg:
    :param tr:
    :return:
    """
    ctg = tr.ref_category()
    if ctg in ref_avg:
        return ref_avg[ctg]
    ctg_unhw = tr.ref_category_unhw()
    if ctg_unhw in ref_avg:
        return ref_avg[ctg_unhw]
    ctg_gen = tr.ref_category_generic()
    if ctg_gen in ref_avg:
        return ref_avg[ctg_gen]
    return None


def is_over_threshold(ref_avg, tr):
    """
    Returns true of tr is over the reference threshold
    :param ref_bins:
    :param tr:
    :type tr: TestRecord
    :return:
    """
    ref = get_ref_value(ref_avg, tr)
    if ref is None:
        return False

    return abs(tr.zscore) >= ref + 1.0


def get_ref_val_def(ref_avg, block, deg, comb_deg, data):
    cat = (REFERENCE_METHOD, block, deg, comb_deg, data)
    return ref_avg[cat] if cat in ref_avg else None


def is_narrow(fname, narrow_type=0):
    """
    Returns true if function is in the narrow set
    :param fname:
    :param narrow_type:
    :return:
    """
    return egenerator.is_narrow(fname, narrow_type)


def average(it):
    return sum(it)/float(len(it))


class Processor(object):
    CHECKPOINT_NAME = 'booltest_proc_checkpoint.json'
    REF_NAME = '-aAES-r10-'

    def __init__(self):
        self.args = None
        self.checkpoint = Checkpoint()
        self.time_checkpoint = timer.Timer(start=False)
        self.checkpointed_files = set()
        self.last_checkpoint = time.time()  # do not re-create checkpoint right from the start
        self.last_checkpoint_new_rec = 0
        self.last_checkpoint_cached_rec = 0
        self.tf = None  # tarfile

        # ref bins: method, bl, deg, comb, data
        self.skipped = None
        self.total_functions = None
        self.ref_bins = None
        self.timing_bins = None
        self.test_records = None
        self.invalid_results = None
        self.invalid_results_num = None
        self.reset_state()

    def reset_state(self):
        self.skipped = 0
        self.total_functions = set()
        self.ref_bins = collections.defaultdict(lambda: [])
        self.timing_bins = collections.defaultdict(lambda: [])
        self.test_records = []
        self.invalid_results = []
        self.invalid_results_num = 0

    def get_parser(self):
        parser = argparse.ArgumentParser(description='Process battery of tests')

        parser.add_argument('--json', dest='json', default=False, action='store_const', const=True,
                            help='JSON output')

        parser.add_argument('--zscore-shape', dest='zscore_shape', default=False, action='store_const', const=True,
                            help='abs(round(zscore))')

        parser.add_argument('--out-dir', dest='out_dir', default='.',
                            help='dir for results')

        parser.add_argument('--delim', dest='delim', default=';',
                            help='CSV delimiter')

        parser.add_argument('--tar', dest='tar', default=False, action='store_const', const=True,
                            help='Rad tar archive instead of the folder')

        parser.add_argument('--narrow', dest='narrow', default=False, action='store_const', const=True,
                            help='Process only smaller set of functions')

        parser.add_argument('--narrow2', dest='narrow2', default=False, action='store_const', const=True,
                            help='Process only smaller set of functions2')

        parser.add_argument('--benchmark', dest='benchmark', default=False, action='store_const', const=True,
                            help='Process only smaller set of fnc: benchmark')

        parser.add_argument('--static', dest='static', default=False, action='store_const', const=True,
                            help='Process only static test files')

        parser.add_argument('--aes-ref', dest='aes_ref', default=False, action='store_const', const=True,
                            help='Process only AES reference')

        parser.add_argument('--pval-data', dest='pval_data', default=None,
                            help='file with pval tables')

        parser.add_argument('--num-inp', dest='num_inp', default=None, type=int,
                            help='Max number of inputs, for testing')

        parser.add_argument('--checkpoint', dest='checkpoint', default=False, action='store_const', const=True,
                            help='Dump checkpoints')

        parser.add_argument('--checkpoint-period', dest='checkpoint_period', default=50000, type=int,
                            help='Checkpoint period (create after X reads)')

        parser.add_argument('--checkpoint-file', dest='checkpoint_file', default=self.CHECKPOINT_NAME,
                            help='Checkpoint file name')

        parser.add_argument('--delete-invalid', dest='delete_invalid', default=False, action='store_const', const=True,
                            help='Delete invalid results')

        parser.add_argument('folder', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='folder with test matrix resutls - result dir of testbed.py')
        return parser

    def should_checkpoint(self, idx):
        return (idx > 0 and idx % self.args.checkpoint_period == 0) or (time.time() - self.last_checkpoint > 10*60)

    def save_checkpoint(self):
        with self.time_checkpoint:
            try:
                self.checkpoint.test_records = self.test_records
                self.checkpoint.total_functions = self.total_functions
                self.checkpoint.timing_bins = self.timing_bins
                self.checkpoint.invalid_results = self.invalid_results
                self.checkpoint.invalid_results_num = self.invalid_results_num
                self.checkpoint.skipped = self.skipped
                self.checkpoint.ref_bins = self.ref_bins
                self.checkpoint.recache()

                tmpfile1 = self.args.checkpoint_file + '.backup1'
                tmpfile2 = self.args.checkpoint_file + '.backup2'

                logger.info('Creating checkpoint %s ...' % self.args.checkpoint_file)
                shutil.copyfile(self.args.checkpoint_file, tmpfile1)
                json.dump(self.checkpoint.to_json(), open(tmpfile2, 'w+'), cls=common.AutoJSONEncoder, indent=2)

                shutil.copyfile(tmpfile2, self.args.checkpoint_file)
                os.remove(tmpfile2)

                logger.info('Checkpoint saved %s' % self.args.checkpoint_file)
                self.last_checkpoint = time.time()
                self.last_checkpoint_new_rec = 0
                self.last_checkpoint_cached_rec = 0

            except Exception as e:
                logger.exception('Could not create a checkpoint %s' % self.args.checkpoint_file, exc_info=e)

    def load_checkpoint(self):
        try:
            logger.info('Loading checkpoint %s ...' % self.args.checkpoint_file)
            if not os.path.exists(self.args.checkpoint_file):
                return False

            js = json.load(open(self.args.checkpoint_file))
            self.checkpoint = Checkpoint()
            self.checkpoint.from_json(js)
            self.checkpointed_files = self.checkpoint.get_cached_keys()
            return True
        except Exception as e:
            logger.exception('Could not load a checkpoint %s' % self.args.checkpoint_file, exc_info=e)

        return False

    def move_invalid(self, fname=None):
        if fname is None or self.args.tar or not self.args.delete_invalid:
            return
        try:
            os.rename(fname, '%s.invalid' % fname)
        except Exception as e:
            logger.exception('Could not move invalid file', exc_info=e)

    def accepting_file(self, tfile, bname):
        if self.args.static and ('static' not in bname and self.REF_NAME not in bname):
            return False

        if self.args.aes_ref and self.REF_NAME not in bname:
            return False

        if self.args.narrow and not is_narrow(bname):
            return False

        if self.args.narrow2 and not is_narrow(bname, 1):
            return False

        if self.args.benchmark and not is_narrow(bname, 2):
            return False

        return True

    def read_file(self, tfile, bname):
        js = None
        data = None
        stats = None
        try:
            if self.args.tar:
                with self.tf.extractfile(tfile) as fh:
                    data = fh.read()
                    js = json.loads(data)

            else:
                fd = os.open(tfile.path, os.O_RDONLY)
                try:
                    fh = os.fdopen(fd, 'r')
                    stats = os.fstat(fd)
                    data = fh.read()
                    js = json.loads(data)
                finally:
                    os.close(fd)

        except Exception as e:
            logger.error('Exception during processing %s: %s' % (tfile, e))
            logger.debug(traceback.format_exc())

        return js, data, stats

    def process_tr(self, tr, tfile, bname):
        if tr.zscore is None or tr.data == 0:
            self.invalid_results_num += 1
            self.invalid_results.append(bname)
            self.move_invalid(tfile.path if not self.args.tar else None)
            return False

        if self.REF_NAME in bname:
            tr.ref = True
            ref_cat = tr.ref_category()
            ref_cat_unhw = tr.ref_category_unhw()

            self.ref_bins[ref_cat].append(tr)
            if ref_cat != ref_cat_unhw:
                self.ref_bins[ref_cat_unhw].append(tr)

        self.test_records.append(tr)
        self.total_functions.add(tr.function)
        if tr.time_process:
            self.timing_bins[tr.bool_category()].append(tr.time_process)
        return True

    def read_file_tr(self, tfile, bname):
        # File read & parse
        js, data, stats = self.read_file(tfile, bname)

        # File process
        if js is None:
            self.move_invalid(tfile.path if not self.args.tar else None)
            return False

        try:
            hasher = hashlib.sha1()
            if isinstance(data, str):
                hasher.update(data.encode())
            else:
                hasher.update(data)

            tr = process_file(js, bname, self.args)
            tr.fhash = hasher.hexdigest()
            tr.mtime = stats.st_mtime if stats else None

            return tr
        except Exception as e:
            logger.exception('Could not process file', exc_info=e)

        return None

    def main(self):
        """
        testbed.py results processor

        "best_zscore"
        "blocklen": 256,
        "degree": 2,
        "comb_degree": 2,
        "top_k": 128,
        config.config.spec.fnc
        config.config.spec.c_round
        config.config.spec.data_size
        config.config.spec.strategy
        config.config.spec.gen_cfg.stream.scode
        config.config.spec.gen_cfg.stream.type
        config.config.spec.gen_cfg.stream.source.type
        :return:
        """
        parser = self.get_parser()
        args = parser.parse_args()
        self.args = args

        tstart = time.time()

        # Process the input
        if len(args.folder) == 0:
            print('Error; no input given')
            sys.exit(1)

        ctr = -1
        main_dir = args.folder[0]
        self.tf = None

        self.checkpoint = Checkpoint()
        self.checkpoint.args = args

        pval_db = PvalDb(args.pval_data)
        pval_db.load()

        if args.tar:
            import tarfile
            logger.info('Loading tar file: %s' % main_dir)

            self.tf = tarfile.open(main_dir, 'r')
            test_files = [x for x in self.tf.getmembers() if x.isfile()]
            logger.info('Totally %d files found in the tar file' % len(test_files))

        else:
            # Read all files in the folder.
            logger.info('Reading all testfiles list')
            # test_files = [f for f in os.listdir(main_dir) if os.path.isfile(os.path.join(main_dir, f))]
            test_files = os.scandir(main_dir)
            # logger.info('Totally %d tests were performed, parsing...' % len(test_files))

        # Test matrix definition
        total_block = [128, 256, 384, 512]
        total_deg = [1, 2, 3]
        total_comb_deg = [1, 2, 3]
        total_sizes = [10, 100]
        total_cases = [total_block, total_deg, total_comb_deg]
        total_cases_size = total_cases + [total_sizes]

        # Load checkpoint, restore state
        if args.checkpoint and self.load_checkpoint():
            logger.info('Checkpoint loaded, files read: %s' % len(self.checkpointed_files))

        num_cached = 0
        for idx, tfile in enumerate(test_files):
            bname = os.path.basename(tfile.name)
            if not args.tar and not tfile.is_file():
                continue

            if idx % 1000 == 0:
                logger.debug('Progress: %d, cur: %s skipped: %s, time: %.2f, #rec: %s, #fnc: %s, #cachedr: %s, #lcc: %s, lcr: %s'
                             % (idx, tfile.name, self.skipped, time.time() - tstart,
                                len(self.test_records), len(self.total_functions), num_cached,
                                self.last_checkpoint_cached_rec, self.last_checkpoint_new_rec))

            is_file_checkpointed = bname in self.checkpointed_files
            num_cached += 1 if is_file_checkpointed else 0

            if args.checkpoint and self.should_checkpoint(idx) and self.last_checkpoint_cached_rec < self.last_checkpoint_new_rec:
                self.save_checkpoint()

            if args.num_inp is not None and args.num_inp < idx:
                break

            if not bname.endswith('json'):
                continue

            if not self.accepting_file(tfile, bname):
                self.skipped += 1
                continue

            tr = self.read_file_tr(tfile, bname) if not is_file_checkpointed else self.checkpoint.get_cached(bname)
            if tr is None:
                self.move_invalid(tfile.path if not args.tar else None)
                continue

            try:
                if not self.process_tr(tr, tfile, bname):
                    continue

                self.last_checkpoint_cached_rec += 1 if is_file_checkpointed else 0
                self.last_checkpoint_new_rec += 1

            except Exception as e:
                logger.error('Exception during processing %s: %s' % (tfile, e))
                logger.debug(traceback.format_exc())

        logger.info('Invalid results: %s' % self.invalid_results_num)
        logger.info('Num records: %s' % len(self.test_records))
        logger.info('Num functions: %s' % len(self.total_functions))
        logger.info('Num ref bins: %s' % len(self.ref_bins.keys()))
        logger.info('Post processing')

        self.test_records.sort(key=lambda x: (x.function, x.round, x.method, x.data, x.block, x.deg, x.comb_deg))

        if not args.json:
            print(args.delim.join(['function', 'round', 'data'] +
                                  ['%s-%s-%s' % (x[0], x[1], x[2]) for x in itertools.product(*total_cases)]))

        # Reference statistics.
        ref_avg = {}
        for mthd in list(self.ref_bins.keys()):
            samples = self.ref_bins[mthd]
            ref_avg[mthd] = sum([abs(x.zscore) for x in samples]) / float(len(samples))

        # Stats files.
        fname_narrow = 'nw_' if args.narrow else ''
        if args.narrow2:
            fname_narrow = 'nw2_'
        elif args.benchmark:
            fname_narrow = 'bench_'

        fname_time = int(time.time())
        fname_ref_json = os.path.join(args.out_dir, 'ref_%s%s.json' % (fname_narrow, fname_time))
        fname_ref_csv = os.path.join(args.out_dir, 'ref_%s%s.csv' % (fname_narrow, fname_time))
        fname_results_json = os.path.join(args.out_dir, 'results_%s%s.json' % (fname_narrow, fname_time))
        fname_results_bat_json = os.path.join(args.out_dir, 'results_bat_%s%s.json' % (fname_narrow, fname_time))
        fname_results_csv = os.path.join(args.out_dir, 'results_%s%s.csv' % (fname_narrow, fname_time))
        fname_results_rf_csv = os.path.join(args.out_dir, 'results_rf_%s%s.csv' % (fname_narrow, fname_time))
        fname_results_rfd_csv = os.path.join(args.out_dir, 'results_rfd_%s%s.csv' % (fname_narrow, fname_time))
        fname_results_rfr_csv = os.path.join(args.out_dir, 'results_rfr_%s%s.csv' % (fname_narrow, fname_time))
        fname_timing_csv = os.path.join(args.out_dir, 'results_time_%s%s.csv' % (fname_narrow, fname_time))

        # Reference bins
        ref_keys = sorted(list(self.ref_bins.keys()))
        with open(fname_ref_csv, 'w+') as fh_csv, open(fname_ref_json, 'w+') as fh_json:
            fh_json.write('[\n')
            for rf_key in ref_keys:
                method, block, deg, comb_deg, data = rf_key
                ref_cur = self.ref_bins[rf_key]

                csv_line = args.delim.join([
                    method, fls(block), fls(deg), fls(comb_deg), fls(data), fls(ref_avg[rf_key])
                ] + [fls(x.zscore) for x in ref_cur])

                fh_csv.write(csv_line+'\n')
                js_cur = collections.OrderedDict()
                js_cur['method'] = method
                js_cur['block'] = block
                js_cur['deg'] = deg
                js_cur['comb_deg'] = comb_deg
                js_cur['data_size'] = data
                js_cur['zscore_avg'] = ref_avg[rf_key]
                js_cur['zscores'] = [x.zscore for x in ref_cur]
                json.dump(js_cur, fh_json, indent=2)
                fh_json.write(', \n')
            fh_json.write('\n    null\n]\n')

        # Timing resuls
        with open(fname_timing_csv, 'w+') as fh:
            hdr = ['block', 'degree', 'combdeg', 'data', 'num_samples', 'avg', 'stddev', 'data']
            fh.write(args.delim.join(hdr) + '\n')
            for case in itertools.product(*total_cases_size):
                cur_data = list(case)
                time_arr = self.timing_bins[case]
                num_samples = len(time_arr)

                if num_samples == 0:
                    cur_data += [0, None, None, None]

                else:
                    cur_data.append(num_samples)
                    avg_ = sum(time_arr) / float(num_samples)
                    stddev = math.sqrt(sum([(x-avg_)**2 for x in time_arr])/(float(num_samples) - 1)) if num_samples > 1 else None
                    cur_data += [avg_, stddev]
                    cur_data += time_arr
                fh.write(args.delim.join([str(x) for x in cur_data]) + '\n')

        # Close old
        fh_json.close()
        fh_csv.close()

        # Result processing
        fh_json = open(fname_results_json, 'w+')
        fh_bat_json = open(fname_results_bat_json, 'w+')
        fh_csv = open(fname_results_csv, 'w+')
        fh_rf_csv = open(fname_results_rf_csv, 'w+')
        fh_rfd_csv = open(fname_results_rfd_csv, 'w+')
        fh_rfr_csv = open(fname_results_rfr_csv, 'w+')
        fh_json.write('[\n')
        fh_bat_json.write('[\n')

        # Headers
        hdr = ['fnc_name', 'fnc_round', 'method', 'data_mb']
        for cur_key in itertools.product(*total_cases):
            hdr.append('%s-%s-%s' % (cur_key[0], cur_key[1], cur_key[2]))
        fh_csv.write(args.delim.join(hdr) + '\n')
        fh_rf_csv.write(args.delim.join(hdr) + '\n')
        fh_rfd_csv.write(args.delim.join(hdr) + '\n')
        fh_rfr_csv.write(args.delim.join(hdr) + '\n')

        # Processing, one per group
        js_out = []
        ref_added = set()
        for k, g in itertools.groupby(self.test_records, key=lambda x: (x.function, x.round, x.method, x.data)):
            logger.info('Key: %s' % list(k))

            fnc_name = k[0]
            fnc_round = k[1]
            method = k[2]
            data_mb = k[3]
            prefix_cols = [fnc_name, fls(fnc_round), method, fls(data_mb)]

            # CSV grouping, avg all results
            csv_grouper = lambda x: (x.block, x.deg, x.comb_deg)
            group_expanded = sorted(list(g), key=csv_grouper)  # type: list[TestRecord]
            results_map = {}
            for ssk, ssg in itertools.groupby(group_expanded, key=csv_grouper):
                ssg = list(ssg)
                if len(ssg) > 1:
                    cp = copy.deepcopy(ssg[0])
                    cp.zscore = average([x.zscore for x in ssg])
                else:
                    cp = ssg[0]
                results_map[ssk] = cp

            # Add line with reference values so one can compare
            if data_mb not in ref_added:
                ref_added.add(data_mb)
                results_list = []
                for cur_key in itertools.product(*total_cases):
                    results_list.append(get_ref_val_def(ref_avg, *cur_key, data=data_mb))

                csv_line = args.delim.join(['ref-AES', '10', REFERENCE_METHOD, fls(data_mb)]
                                           + [(fls(x) if x is not None else '-') for x in results_list])
                fh_csv.write(csv_line + '\n')

            # Grid list for booltest params
            results_list = []
            for cur_key in itertools.product(*total_cases):
                if cur_key in results_map:
                    results_list.append(results_map[cur_key])
                else:
                    results_list.append(None)

            # CSV result
            csv_line = args.delim.join(prefix_cols + [(fls(x.zscore) if x is not None else '-') for x in results_list])
            fh_csv.write(csv_line+'\n')

            # CSV only if above threshold
            def zscoreref(x, retmode=0):
                if x is None:
                    return '-'

                is_over = False
                thr = 0
                if args.pval_data:
                    is_over = pval_db.eval(x.block, x.deg, x.comb_deg, x.zscore)
                    if is_over is None:
                        return '?'

                else:
                    thr = get_ref_value(ref_avg, x)
                    if thr is None:
                        return '?'
                    is_over = is_over_threshold(ref_avg, x)

                if is_over:
                    if retmode == 0:
                        return fls(x.zscore)
                    elif retmode == 1:
                        return fls(abs(x.zscore) - abs(thr))
                    elif retmode == 2:
                        thr = thr if thr != 0 else 1.
                        return fls(abs(x.zscore) / abs(thr))
                return '.'

            csv_line_rf = args.delim.join(
                prefix_cols + [zscoreref(x) for x in results_list])
            fh_rf_csv.write(csv_line_rf + '\n')

            csv_line_rfd = args.delim.join(
                prefix_cols + [zscoreref(x, 1) for x in results_list])
            fh_rfd_csv.write(csv_line_rfd + '\n')

            csv_line_rfr = args.delim.join(
                prefix_cols + [zscoreref(x, 2) for x in results_list])
            fh_rfr_csv.write(csv_line_rfr + '\n')

            # JSON result
            cur_js = collections.OrderedDict()
            cur_js['function'] = fnc_name
            cur_js['round'] = fnc_round
            cur_js['method'] = method
            cur_js['data_mb'] = data_mb
            cur_js['tests'] = [[x.block, x.deg, x.comb_deg, x.zscore] for x in group_expanded]
            json.dump(cur_js, fh_json, indent=2)
            fh_json.write(',\n')

            # JSON battery format result
            for cur_res in group_expanded:
                cur_js = collections.OrderedDict()
                cur_js['battery'] = 'booltest'
                cur_js['function'] = fnc_name
                cur_js['round'] = fnc_round
                cur_js['method'] = method
                cur_js['data_bytes'] = cur_res.data_bytes
                cur_js['deg'] = cur_res.deg
                cur_js['k'] = cur_res.comb_deg
                cur_js['m'] = cur_res.block
                cur_js['data_file'] = cur_res.cfg_file_name
                cur_js['zscore'] = cur_res.zscore
                cur_js['pval0_rej'] = pval_db.eval(cur_res.block, cur_res.deg, cur_res.comb_deg, cur_res.zscore) if args.pval_data else None
                json.dump(cur_js, fh_bat_json, indent=2)
                fh_bat_json.write(',\n')

            if not args.json:
                print(csv_line)

            else:
                js_out.append(cur_js)

        fh_json.write('\nnull\n]\n')
        fh_bat_json.write('\nnull\n]\n')
        if args.json:
            print(json.dumps(js_out, indent=2))

        fh_json.close()
        fh_bat_json.close()
        fh_csv.close()
        fh_rf_csv.close()
        fh_rfd_csv.close()
        fh_rfr_csv.close()

        logger.info(fname_ref_json)
        logger.info(fname_ref_csv)
        logger.info(fname_results_json)
        logger.info(fname_results_bat_json)
        logger.info(fname_results_csv)
        logger.info(fname_results_rf_csv)
        logger.info(fname_results_rfd_csv)
        logger.info(fname_results_rfr_csv)
        logger.info(fname_timing_csv)
        logger.info('Processing finished in %s s' % (time.time() - tstart))


def main():
    p = Processor()
    p.main()


if __name__ == '__main__':
    main()




