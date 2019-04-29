#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dusanklinec'


import argparse
import fileinput
import json
import os
import sys
import collections
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


def process_file(data, bl, deg, k, ctr, split_key=False, delim=','):
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
    label = '%d-%d-%d' % (bl, deg, k)

    if split_key:
        label = '%d%s%d%s%d' % (bl, delim, deg, delim, k)

    for z in js:
        print('%s%s%s' % (label, delim, abs(z)))


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

    parser.add_argument('files', nargs=argparse.ZERO_OR_MORE, default=[],
                        help='folder with aes results')

    args = parser.parse_args()

    # Process the input
    if len(args.files) == 0:
        print('Error; no input given')
        sys.exit(1)

    ctr = -1
    main_dir = args.files[0]
    for k in [1, 2, 3]:
        for deg in [1, 2, 3]:
            for bl in [128, 256, 384, 512]:
                file_name = 'aestest-%dbl-%ddeg-%dk-10485760B-100tests.out' % (bl, deg, k)
                file_path = os.path.join(main_dir, file_name)
                ctr += 1
                if not os.path.exists(file_path):
                    logger.error('File not found: %s' % file_path)
                    continue

                with open(file_path, 'r') as fh:
                    try:
                        process_file(fh.read(), bl, deg, k, ctr, split_key=args.split_key, delim=args.delim)
                    except Exception as e:
                        logger.error('Exception in processing %s: %s' % (file_path, e))


if __name__ == '__main__':
    main()




