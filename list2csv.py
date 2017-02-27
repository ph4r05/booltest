#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'dusanklinec'


import argparse
import fileinput
import json
import sys
import os


def main():
    """
    Json list 2 csv
    * LABEL=4096 python list2csv.py
    * echo '' | LABEL=16 python list2csv.py
    * grep 'BEGIN Z-SCORES-ABS-----' testjava_x_1024k.txt -A 1 | tail -n 1 | LABEL=16 python list2csv.py
    * for xxx in 16 32 64 128 256 512 1024 2048 4096 8192; do grep 'BEGIN Z-SCORES-ABS-----' testjava_x_${xxx}k.txt -A 1 | tail -n 1 | LABEL=${xxx} python list2csv.py >> testjava_x_total.csv; done
    :return:
    """

    buff = ''
    for line in fileinput.input():
        buff += line

    js = json.loads(buff)

    for x in js:
        print('%s,%s' % (os.environ['LABEL'], abs(x)))


if __name__ == '__main__':
    main()




