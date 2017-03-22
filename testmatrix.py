#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dusanklinec'

from past.builtins import cmp
import argparse
import fileinput
import json
import os
import sys
import collections
import itertools
import traceback
import logging
import math
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


class TestRecord(object):
    """
    Represents one performed test and its result.
    """
    def __init__(self, function=None, round=None, block=None, deg=None, comb_deg=None, data=None,
                 elapsed=None, zscore=None, best_poly=None):
        self.function = None
        self.round = None
        self.block = None
        self.deg = None
        self.comb_deg = None
        self.data = None
        self.elapsed = None

        self.zscore = None
        self.best_poly = None

    def __cmp__(self, other):
        """
        Compare: function, round, data, block, deg, k.
        :param other:
        :return:
        """
        a = (self.function, self.round, self.data, self.block, self.deg, self.comb_deg)
        b = (other.function, other.round, other.data, other.block, other.deg, other.comb_deg)
        return cmp(a, b)

    def __repr__(self):
        return '%s-r%d-d%s_bl%d-deg%d-k%d' % (self.function, self.round, self.data, self.block, self.deg, self.comb_deg)


def process_file(js, args=None):
    """
    Process file json
    :param js:
    :return:
    """
    tr = TestRecord()
    tr.zscore = round(js['best_zscore'], 6)
    if args.zscore_shape:
        tr.zscore = int(abs(round(tr.zscore)))

    tr.best_poly = js['best_poly']

    if 'stream' in js['generator']:
        tr.function = js['generator']['stream']['algorithm']
        tr.round = js['generator']['stream']['round']
    else:
        tr.function = js['generator']['algorithm']
        tr.round = js['generator']['round']

    tr.block = js['blocklen']
    tr.deg = js['degree']
    tr.comb_deg = js['comb_degree']
    tr.data = int(math.ceil(math.ceil(js['data_read']/1024.0)/1024.0))

    if 'elapsed' in js:
        tr.elapsed = js['elapsed']

    return tr


def fls(x):
    """
    Converts float to string, replacing . with , - excel separator
    :param x:
    :return:
    """
    return str(x).replace('.', ',')


def main():
    """
    testbed.py results processor
    :return:
    """
    parser = argparse.ArgumentParser(description='Process battery of tests and produces CSV / JSON output')

    parser.add_argument('--json', dest='json', default=False, action='store_const', const=True,
                        help='JSON output')

    parser.add_argument('--zscore-shape', dest='zscore_shape', default=False, action='store_const', const=True,
                        help='abs(round(zscore))')

    parser.add_argument('--delim', dest='delim', default=';',
                        help='CSV delimiter')

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

    # Test matrix definition
    total_functions = set()
    total_block = [128, 256, 384, 512]
    total_deg = [1, 2, 3]
    total_comb_deg = [1, 2, 3]
    total_cases = [total_block, total_deg, total_comb_deg]

    test_records = []

    logger.info('Totally %d tests were performed, parsing...' % total_files)
    for idx, tfile in enumerate(test_files):
        if idx % 1000 == 0:
            logger.debug('Progress: %d, cur: %s' % (idx, tfile))

        test_file = os.path.join(main_dir, tfile)
        try:
            with open(test_file, 'r') as fh:
                js = json.load(fh)
                tr = process_file(js, args)
                test_records.append(tr)
                total_functions.add(tr.function)

        except Exception as e:
            logger.error('Exception during processing %s: %s' % (tfile, e))
            logger.debug(traceback.format_exc())

    logger.info('Post processing')
    test_records.sort()

    if not args.json:
        print(args.delim.join(['function', 'round', 'data'] +
                              ['%s-%s-%s' % (x[0], x[1], x[2]) for x in itertools.product(*total_cases)]))

    js_out = []
    for k, g in itertools.groupby(test_records, key=lambda x: (x.function, x.round, x.data)):
        logger.info('Key: %s' % list(k))

        function = k[0]
        round = k[1]
        data_mb = k[2]

        group_expanded = [x for x in g]
        results_map = {(x.block, x.deg, x.comb_deg): x for x in group_expanded}

        results_list = []
        for cur_key in itertools.product(*total_cases):
            if cur_key in results_map:
                results_list.append(results_map[cur_key])
            else:
                results_list.append(None)

        if not args.json:
            print(args.delim.join([
                function, fls(round), fls(data_mb)
            ] + [(fls(x.zscore) if x is not None else '-') for x in results_list]))

        else:
            cur_js = collections.OrderedDict()
            cur_js['function'] = function
            cur_js['round'] = round
            cur_js['data_mb'] = data_mb
            cur_js['tests'] = [[x.block, x.deg, x.comb_deg, x.zscore] for x in group_expanded]
            js_out.append(cur_js)

    if args.json:
        print(json.dumps(js_out, indent=2))

if __name__ == '__main__':
    main()




