#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dusanklinec'


import argparse
import fileinput
import json
import sys
import collections


def main():
    """
    Process output of the randverif.py and produces CSV for graphing in R
    :return:
    """
    parser = argparse.ArgumentParser(description='json2csv')

    parser.add_argument('--start-idx', dest='start_idx', default=1, type=int,
                        help='Default index to start with - numbering')

    parser.add_argument('files', nargs=argparse.ZERO_OR_MORE, default=[],
                        help='files to process')

    args = parser.parse_args()

    # Process the input
    buff = []
    extended_parse = False
    delim_start_found = False
    for idx, line in enumerate(fileinput.input(args.files)):
        if idx == 0:
            if not line.startswith('{') and not line.startswith('['):
                extended_parse = True

        if extended_parse:
            if not delim_start_found and line.strip().startswith('-----BEGIN JSON-----'):
                delim_start_found = True
                continue
            if delim_start_found and line.strip().startswith('-----BEGIN'):
                break
            if delim_start_found:
                buff.append(line)
        else:
            buff.append(line)

    js = json.loads('\n'.join(buff))

    # counts - give first char to the most frequent distinguisher
    cnt = collections.Counter()
    for row in js:
        cnt[row['d']] += 1

    ctr = 0
    mapping = {}
    for elem in cnt.most_common():
        d = elem[0]
        mapping[d] = str(args.start_idx + ctr)
        ctr += 1

    sys.stderr.write(json.dumps(mapping) + '\n')

    print('label,variable,value')
    for row in js:
        z = row['z']
        d = row['d']
        sig = '+' if z >= 0 else '-'
        # var = '"f[%s]^%s"' % (mapping[d], sig)
        # var = '"f[%s]"' % (mapping[d])  # , sig)
        var = '"f[%s]^{{"%s{}"}}"' % (mapping[d], sig)
        print('%s,%s,%s' % (1, var, z))


if __name__ == '__main__':
    main()




