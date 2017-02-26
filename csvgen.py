#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dusanklinec'


import argparse
import fileinput
import json
import sys


def main():
    """
    Reads stdin jboss output, writes json on output
    :return:
    """
    # parser = argparse.ArgumentParser(description='json2csv')
    # parser.add_argument('files', dest='files', nargs=argparse.ZERO_OR_MORE, default=[],
    #                     help='files to process')
    # args = parser.parse_args()

    buff = ''
    for line in fileinput.input():
        buff += line

    js = json.loads(buff)

    ctr = 0
    mapping = {}
    for row in js:
        d = row['d']
        if d not in mapping:
            mapping[d] = chr(ord('a') + ctr)
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




