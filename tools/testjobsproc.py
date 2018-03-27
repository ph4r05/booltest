#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dusanklinec'

from past.builtins import cmp
import argparse
import fileinput
import json
import time
import re
import os
import sys
import collections
import itertools
import traceback
import logging
import math
import coloredlogs

from booltest import common, egenerator

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


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
        self.elapsed = None
        self.iteration = 0
        self.strategy = None
        self.method = None
        self.ref = False

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

    def __repr__(self):
        return '%s-r%d-d%s_bl%d-deg%d-k%d' % (self.function, self.round, self.data, self.block, self.deg, self.comb_deg)

    def ref_category(self):
        return self.method, self.block, self.deg, self.comb_deg, self.data

    def ref_category_unhw(self):
        return self.method_unhw(), self.block, self.deg, self.comb_deg, self.data


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

    if tr.data:
        tr.data = int(math.ceil(math.ceil(tr.data/1024.0)/1024.0))

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


def is_over_threshold(ref_avg, tr):
    """
    Returns true of tr is over the reference threshold
    :param ref_bins:
    :param tr:
    :type tr: TestRecord
    :return:
    """
    ctg = tr.ref_category()
    if ctg in ref_avg:
        return abs(tr.zscore) >= ref_avg[ctg] + 1.0
    ctg_unhw = tr.ref_category_unhw()
    if ctg_unhw in ref_avg:
        return abs(tr.zscore) >= ref_avg[ctg_unhw] + 1.0
    return False


def is_narrow(fname, narrow_type=0):
    """
    Returns true if function is in the narrow set
    :param fname:
    :param narrow_type:
    :return:
    """
    return egenerator.is_narrow(fname, narrow_type)


def main():
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
    parser = argparse.ArgumentParser(description='Process battery of tests')

    parser.add_argument('--json', dest='json', default=False, action='store_const', const=True,
                        help='JSON output')

    parser.add_argument('--zscore-shape', dest='zscore_shape', default=False, action='store_const', const=True,
                        help='abs(round(zscore))')

    parser.add_argument('--out-dir', dest='out_dir', default='.',
                        help='dir for results')

    parser.add_argument('--delim', dest='delim', default=';',
                        help='CSV delimiter')

    parser.add_argument('--narrow', dest='narrow', default=False, action='store_const', const=True,
                        help='Process only smaller set of functions')

    parser.add_argument('--narrow2', dest='narrow2', default=False, action='store_const', const=True,
                        help='Process only smaller set of functions2')

    parser.add_argument('folder', nargs=argparse.ZERO_OR_MORE, default=[],
                        help='folder with test matrix resutls - result dir of testbed.py')

    args = parser.parse_args()

    # Process the input
    if len(args.folder) == 0:
        print('Error; no input given')
        sys.exit(1)

    ctr = -1
    main_dir = args.folder[0]

    # Read all files in the folder.
    logger.info('Reading all testfiles list')
    test_files = [f for f in os.listdir(main_dir) if os.path.isfile(os.path.join(main_dir, f))]
    total_files = len(test_files)
    logger.info('Totally %d tests were performed, parsing...' % total_files)

    # Test matrix definition
    total_functions = set()
    total_block = [128, 256, 384, 512]
    total_deg = [1, 2, 3]
    total_comb_deg = [1, 2, 3]
    total_sizes = [1, 10, 100]
    total_cases = [total_block, total_deg, total_comb_deg]

    # ref bins: method, bl, deg, comb, data
    ref_name = '-aAES-r10-'
    ref_bins = collections.defaultdict(lambda: [])

    test_records = []
    skipped = 0

    invalid_results = []
    invalid_results_num = 0
    for idx, tfile in enumerate(test_files):
        if idx % 1000 == 0:
            logger.debug('Progress: %d, cur: %s skipped: %s' % (idx, tfile, skipped))

        if tfile.endswith('csv'):
            continue

        if args.narrow and not is_narrow(tfile):
            skipped += 1
            continue

        if args.narrow2 and not is_narrow(tfile, 2):
            skipped += 1
            continue

        test_file = os.path.join(main_dir, tfile)
        try:
            with open(test_file, 'r') as fh:
                js = json.load(fh)

            tr = process_file(js, tfile, args)
            if tr.zscore is None or tr.data == 0:
                invalid_results_num += 1
                invalid_results.append(tfile)
                continue

            if ref_name in tfile:
                tr.ref = True
                ref_cat = tr.ref_category()
                ref_cat_unhw = tr.ref_category_unhw()

                ref_bins[ref_cat].append(tr)
                if ref_cat != ref_cat_unhw:
                    ref_bins[ref_cat_unhw].append(tr)

            test_records.append(tr)
            total_functions.add(tr.function)

        except Exception as e:
            logger.error('Exception during processing %s: %s' % (tfile, e))
            logger.debug(traceback.format_exc())

    logger.info('Invalid results: %s' % invalid_results_num)
    logger.info('Post processing')

    test_records.sort(key=lambda x: (x.function, x.round, x.method, x.data, x.block, x.deg, x.comb_deg))

    if not args.json:
        print(args.delim.join(['function', 'round', 'data'] +
                              ['%s-%s-%s' % (x[0], x[1], x[2]) for x in itertools.product(*total_cases)]))

    # Reference statistics.
    ref_avg = {}
    for mthd in list(ref_bins.keys()):
        samples = ref_bins[mthd]
        ref_avg[mthd] = sum([abs(x.zscore) for x in samples]) / float(len(samples))

    # Stats files.
    fname_narrow = 'nw_' if args.narrow else ''
    if args.narrow2:
        fname_narrow = 'nw2_'

    fname_time = int(time.time())
    fname_ref_json = os.path.join(args.out_dir, 'ref_%s%s.json' % (fname_narrow, fname_time))
    fname_ref_csv = os.path.join(args.out_dir, 'ref_%s%s.csv' % (fname_narrow, fname_time))
    fname_results_json = os.path.join(args.out_dir, 'results_%s%s.json' % (fname_narrow, fname_time))
    fname_results_csv = os.path.join(args.out_dir, 'results_%s%s.csv' % (fname_narrow, fname_time))
    fname_results_rf_csv = os.path.join(args.out_dir, 'results_rf_%s%s.csv' % (fname_narrow, fname_time))

    ref_keys = sorted(list(ref_bins.keys()))
    with open(fname_ref_csv, 'w+') as fh_csv, open(fname_ref_json, 'w+') as fh_json:
        fh_json.write('[\n')
        for rf_key in ref_keys:
            method, block, deg, comb_deg, data = rf_key
            ref_cur = ref_bins[rf_key]

            csv_line = args.delim.join([
                method, fls(block), fls(deg), fls(comb_deg), fls(data), fls(ref_avg[rf_key])
            ] + [fls(x.zscore) for x in ref_cur])

            fh_csv.write(csv_line+'\n')
            js_cur = collections.OrderedDict()
            js_cur['method'] = method
            js_cur['block'] = method
            js_cur['deg'] = deg
            js_cur['comb_deg'] = comb_deg
            js_cur['data_size'] = data
            js_cur['zscore_avg'] = ref_avg[rf_key]
            js_cur['zscores'] = [x.zscore for x in ref_cur]
            json.dump(js_cur, fh_json, indent=2)
            fh_json.write(', \n')
        fh_json.write('\n    null\n]\n')

    # Result processing
    fh_json = open(fname_results_json, 'w+')
    fh_csv = open(fname_results_csv, 'w+')
    fh_rf_csv = open(fname_results_rf_csv, 'w+')
    fh_json.write('[\n')

    # Headers
    fh_csv.write('function' + args.delim)
    fh_rf_csv.write('function' + args.delim)
    hdr = []
    for cur_key in itertools.product(*total_cases):
        hdr.append('%s-%s-%s' % (cur_key[0], cur_key[1], cur_key[2]))
    fh_csv.write(args.delim.join(hdr) + '\n')
    fh_rf_csv.write(args.delim.join(hdr) + '\n')

    # Processing
    js_out = []
    for k, g in itertools.groupby(test_records, key=lambda x: (x.function, x.round, x.method, x.data)):
        logger.info('Key: %s' % list(k))

        fnc_name = k[0]
        fnc_round = k[1]
        method = k[2]
        data_mb = k[3]

        group_expanded = list(g)
        results_map = {(x.block, x.deg, x.comb_deg): x for x in group_expanded}

        # Grid list for booltest params
        results_list = []
        for cur_key in itertools.product(*total_cases):
            if cur_key in results_map:
                results_list.append(results_map[cur_key])
            else:
                results_list.append(None)

        # CSV result
        csv_line = args.delim.join(
            [
                 fnc_name, fls(fnc_round), method, fls(data_mb)
            ] + [(fls(x.zscore) if x is not None else '-') for x in results_list])
        fh_csv.write(csv_line+'\n')

        # CSV only if above threshold
        def zscoreref(x):
            if x is None:
                return '-'
            if is_over_threshold(ref_avg, x):
                return fls(x.zscore)
            return '.'

        csv_line_rf = args.delim.join(
            [
                 fnc_name, fls(fnc_round), method, fls(data_mb)
            ] + [zscoreref(x) for x in results_list])
        fh_rf_csv.write(csv_line_rf + '\n')

        # JSON result
        cur_js = collections.OrderedDict()
        cur_js['function'] = fnc_name
        cur_js['round'] = fnc_round
        cur_js['method'] = method
        cur_js['data_mb'] = data_mb
        cur_js['tests'] = [[x.block, x.deg, x.comb_deg, x.zscore] for x in group_expanded]
        json.dump(cur_js, fh_json, indent=2)
        fh_json.write(',\n')

        if not args.json:
            print(csv_line)

        else:
            js_out.append(cur_js)

    fh_json.write(',\nNone\n]\n')
    if args.json:
        print(json.dumps(js_out, indent=2))

    fh_json.close()
    fh_csv.close()
    fh_rf_csv.close()


if __name__ == '__main__':
    main()




