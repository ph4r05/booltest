#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dusanklinec'


import argparse
import pprint
import json
import os
import sys
import math
import collections
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


def process_file(data, bl, deg, k, ctr):
    """
    Process file
    :param file_path:
    :param bl:
    :param deg:
    :param k:
    :param ctr:
    :return:
    """

    buff = []
    extended_parse = False
    delim_start_found = False

    for idx, line in enumerate(data.split('\n')):
        if idx == 0:
            if not line.startswith('{') and not line.startswith('['):
                extended_parse = True

        if extended_parse:
            if not delim_start_found and line.strip().startswith('-----BEGIN Z-SCORES-NORM-----'):
                delim_start_found = True
                continue
            if delim_start_found and line.strip().startswith('-----BEGIN'):
                break
            if delim_start_found:
                buff.append(line)
        else:
            buff.append(line)

    js = json.loads('\n'.join(buff))
    return js


def main():
    """
    AES process
    :return:
    """
    parser = argparse.ArgumentParser(description='process aes results')

    parser.add_argument('--start-char', dest='start_char', default='a',
                        help='Default character to start with - numbering')

    parser.add_argument('--split-key', dest='split_key', default=False, action='store_const', const=True,
                        help='Splits CSV key to multiple columns')

    parser.add_argument('--delim', dest='delim', default=',',
                        help='CSV delimiter')

    parser.add_argument('--alpha', dest='alpha', default=0.05, type=float,
                        help='Confidence')

    parser.add_argument('--avg', dest='avg', default=False, action='store_const', const=True,
                        help='Compute average')

    parser.add_argument('--json', dest='json', default=False, action='store_const', const=True,
                        help='JSON output')

    parser.add_argument('--py', dest='py', default=False, action='store_const', const=True,
                        help='Python output')

    parser.add_argument('files', nargs=argparse.ZERO_OR_MORE, default=[],
                        help='folder with aes results')

    args = parser.parse_args()

    # Process the input
    if len(args.files) == 0:
        print('Error; no input given')
        sys.exit(1)

    ctr = -1
    main_dir = args.files[0]
    db = collections.OrderedDict()

    for bl in [128, 256, 384, 512]:
        for deg in [1, 2, 3]:
            for k in [1, 2, 3]:
                for tests in [100, 1000]:
                    for basename in ['aestest', 'aessize']:
                        file_name = '%s-%dbl-%ddeg-%dk-%dB-%stests.out' % (basename, bl, deg, k, 1048576 * 10, tests)
                        file_path = os.path.join(main_dir, file_name)
                        ctr += 1
                        if not os.path.exists(file_path):
                            logger.error('File not found: %s' % file_path)
                            continue

                        with open(file_path, 'r') as fh:
                            try:
                                vals = process_file(fh.read(), bl, deg, k, ctr)
                                zscores = sorted([abs(x) for x in vals])
                                ln = len(vals)
                                zscores_avg = sum(zscores)/float(ln)

                                idx = min(ln-1, int(ln - math.floor(ln * args.alpha)))
                                dbkey = (bl, deg, k)

                                db[dbkey] = zscores[idx]
                                if args.avg:
                                    db[dbkey] = zscores_avg

                                # logging of the competed val, one can verify results 100 vs. 1000 tests
                                logger.info('%s t=%s, %s avg=%s' % (dbkey, tests, zscores[idx], zscores_avg))
                            except Exception as e:
                                logger.error('Exception in processing %s: %s' % (file_path, e))

    if args.py:
        pprint.pprint(dict(db), indent=2)
        return

    if args.json:
        js_db = collections.OrderedDict()
        for bl in [128, 256, 384, 512]:
            js_db[bl] = {}
            for deg in [1, 2, 3]:
                js_db[bl][deg] = {}
                for k in [1, 2, 3]:
                    js_db[bl][deg][k] = db[(bl, deg, k)]
        print(json.dumps(js_db, indent=2))
        return

    # CSV
    delim = '-'
    if args.split_key:
        delim = args.delim

    print('block%sdeg%sk%spval' % (delim, delim, args.delim))
    for bl in [128, 256, 384, 512]:
        for deg in [1, 2, 3]:
            for k in [1, 2, 3]:
                label = '%d%s%d%s%d' % (bl, delim, deg, delim, k)
                print('%s%s%s' % (label, args.delim, db[(bl,deg,k)]))


if __name__ == '__main__':
    main()




