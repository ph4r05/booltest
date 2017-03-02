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
    Reads stdin jboss output, writes json on output
    :return:
    """
    parser = argparse.ArgumentParser(description='json2csv')

    parser.add_argument('--start-char', dest='start_char', default='a',
                        help='Default character to start with - numbering')

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
        mapping[d] = chr(ord(args.start_char) + ctr)
        ctr += 1

    sys.stderr.write(json.dumps(mapping) + '\n')

    print('label,variable,value')
    for row in js:
        z = row['z']
        d = row['d']
        sig = '+' if z >= 0 else '-'
        var = '%s%s' % (mapping[d], sig)
        print('%s,%s,%s' % (1, var, z))


if __name__ == '__main__':
    main()




