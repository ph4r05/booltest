#!/usr/bin/env python
# -*- coding: utf-8 -*-

from future.utils import iteritems
from past.builtins import cmp

import base64, textwrap, time, random, datetime
import logging
import coloredlogs
import itertools
import json
from json import JSONEncoder
import decimal
import os
import sys
import collections
import argparse
import socket
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


logger = logging.getLogger(__name__)

coloredlogs.CHROOT_FILES = []
coloredlogs.install(level=logging.DEBUG, use_chroot=False)


def defval(val, default=None):
    """
    Returns val if is not None, default instead
    :param val:
    :param default:
    :return:
    """
    return val if val is not None else default


def defvalkey(js, key, default=None, take_none=True):
    """
    Returns js[key] if set, otherwise default. Note js[key] can be None.
    :param js:
    :param key:
    :param default:
    :param take_none:
    :return:
    """
    if js is None:
        return default
    if key not in js:
        return default
    if js[key] is None and not take_none:
        return default
    return js[key]


class Graphs(object):
    def __init__(self):
        self.args = None

    def main(self):
        """
        Main entry method
        :return:
        """
        parser = argparse.ArgumentParser(description='Graphs')

        parser.add_argument('--debug', dest='debug', default=False, action='store_const', const=True,
                            help='Debugging logging')

        parser.add_argument('--counts', dest='counts', default=False, action='store_const', const=True,
                            help='Job ordering analysis')

        parser.add_argument('--duplicities', dest='duplicities', default=False, action='store_const', const=True,
                            help='Job duplicity analysis')

        parser.add_argument('--avg', dest='avg', default=False, action='store_const', const=True,
                            help='Average jobs per second')

        parser.add_argument('--classic', dest='classic', default=False, action='store_const', const=True,
                            help='Classic graph')

        parser.add_argument('--data', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='data dirs with json files')

        self.args = parser.parse_args()
        self.work()

    def work(self):
        """
        Entry
        :return:
        """

        if self.args.avg:
            self.average()
            return

        if self.args.counts or self.args.duplicities:
            self.counts()
            return

    def counts(self):
        """
        Job ordering analysis
        :return:
        """
        dirs = self.args.data
        datasets = []

        for didx, dirname in enumerate(dirs):
            sub_rec = [f for f in os.listdir(dirname)]
            for fidx, fname in enumerate(sub_rec):
                full_path = os.path.join(dirname, fname)

                # dataset reset for each file.
                # for aggregation among directories use a dist key for test (conn, delMark, ...)
                datasets = []
                datasets_dupl = []
                if not os.path.isfile(full_path) or not fname.endswith('json'):
                    continue

                # if 'run_1513508909_mysql_conn1_dm1_dtsx0_dretry1_batch10000_cl0_window1_verify1' not in fname:
                #     continue

                with open(full_path, 'r') as fh:
                    js = json.load(fh)
                    if not js['settings']['verify']:
                        continue

                    sett = js['settings']
                    sett['db_conn'] = defvalkey(sett, 'db_conn', 'mysql')

                    #
                    # key, disp = Graphs.avg_dataname(js, self.args.classic)
                    # if key is None:
                    #     continue

                    bins = collections.defaultdict(lambda: 0)
                    bins_duplicities = collections.defaultdict(lambda: 0)
                    for x in js['runs']:
                        for counts in x['counts']:
                            bins[counts[0]] += counts[1]
                        for dupl in x['duplicities']:
                            bins_duplicities[dupl[0]] += dupl[1]

                    # binning 0,49, 50-100, 101-200, 200-1000, 1000-2000, 3000+
                    bin_merged = collections.defaultdict(lambda: 0)
                    for x, y in iteritems(bins):
                        bin_merged[Graphs.bin_name(x)] += y

                    for x, y in iteritems(bin_merged):
                        datasets.append({
                            'diff': x[0],
                            'ord': x[1],
                            'count': y / float(len(js['runs'])),
                            'env': dirname,
                            'conn': sett['conn'],
                            'db_conn': defvalkey(sett, 'db_conn'),
                            'delTsxFetch': sett['delTsxFetch'],
                            'delTsxRetry': sett['delTsxRetry'],
                            'deleteMark': sett['deleteMark'],
                            'windowStrategy': sett['windowStrategy'],
                        })

                    for x, y in iteritems(bins_duplicities):
                        datasets_dupl.append({
                            'reruns': x,
                            'counts': y / float(len(js['runs'])),
                            'env': dirname,
                        })

                    if len(bins) == 0 and self.args.counts:
                        continue
                    if len(bins_duplicities) == 0 and self.args.duplicities:
                        continue

                    sns.set_style("whitegrid")
                    datasets.sort(key=lambda x: x['ord'])
                    datasets_dupl.sort(key=lambda x: x['reruns'])
                    data = pd.DataFrame(datasets)
                    data_rerun = pd.DataFrame(datasets_dupl)

                    print(fname)
                    if self.args.counts:
                        print(bins)
                        for x in datasets:
                            print('  %s: %s' % (x['diff'], x['count']))

                    elif self.args.duplicities:
                        print(bins_duplicities)
                        for x in datasets_dupl:
                            print('  %s: %s' % (x['reruns'], x['counts']))

                        runs = js['runs']
                        ents = [x['protoEntries'] for x in runs]
                        uniq = [x['protoUnique'] for x in runs]

                        print('ProtoData: max(protoEntries): %s, '
                              'min(protoEntries): %s, '
                              'avg(protoEntries): %s, '
                              'max(protoUnique): %s, '
                              'min(protoUnique): %s,'
                              'avg(protoUnique): %s,' % (
                            max(ents), min(ents), sum(ents)/float(len(ents)),
                            max(uniq), min(uniq), sum(uniq)/float(len(uniq)),
                        ))

                    print('-' * 80)

                    if self.args.counts:
                        fig, ax = plt.subplots(figsize=(11.7, 8.27))
                        ax = sns.barplot(ax=ax, y='count', x='diff', hue='env', data=data, linewidth=0.5, errwidth=0)
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=-90)

                    elif self.args.duplicities:
                        fig, ax = plt.subplots(figsize=(11.7, 8.27))
                        ax = sns.barplot(ax=ax, y='counts', x='reruns', hue='env', data=data_rerun, linewidth=0.5, errwidth=0)
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=-90)

                    fprefix = 'counts' if self.args.counts else 'dupl'
                    plt.savefig('/tmp/%s_%s.svg' % (fprefix, fname), transparent=True)
                    # plt.show()

                    plt.cla()
                    plt.clf()
                    plt.close()

                # if fidx > 2:
                #     return

        # print(datasets)

    @staticmethod
    def bin_name(val):
        """
        Binning aggregate function
        :param val:
        :return:
        """
        val = int(val)
        if 0 <= val <= 49:
            return val, val
        elif val <= 100:
            return '%d-%d' % (int((val//10)*10), int((val//10)*10)+9), ((val//10)*10)
        elif val <= 200:
            return '100-199', 100
        elif val <= 1000:
            return '200-999', 200
        elif val > 1000:
            return '%d-%d' % (int((val // 1000) * 1000), int((val // 1000) * 1000) + 999), ((val // 1000) * 1000)

    def average(self):
        """
        Average box plot on jobs per second
        :return:
        """
        dirs = self.args.data
        datasets = []

        # Datasets, interesting data:
        #  - redis
        #  - beanstalk
        #  - pessimistic, default 5x, with mark,  mysql, pgsql
        #  - optimistic, window 0, window 1, window 2, mysql, pgsql

        # Read all jsons
        for didx, dirname in enumerate(dirs):
            sub_rec = [f for f in os.listdir(dirname)]
            for fname in sub_rec:
                full_path = os.path.join(dirname, fname)

                if not os.path.isfile(full_path) or not fname.endswith('json'):
                    continue

                with open(full_path, 'r') as fh:
                    js = json.load(fh)
                    if js['settings']['verify']:
                        continue

                    sett = js['settings']
                    sett['db_conn'] = defvalkey(sett, 'db_conn', 'mysql')

                    key, disp = Graphs.avg_dataname(js, self.args.classic)
                    if key is None:
                        continue

                    jps = [x['jps'] for x in js['runs']]
                    print('Avg %s %s: %s' % (dirname, key, sum(jps) / float(len(jps))))

                    for x in js['runs']:
                        datasets.append({
                            'key': key,
                            'method': disp,
                            'env': dirname,
                            'jps': x['jps'],
                            'conn': sett['conn'],
                            'db_conn': defvalkey(sett, 'db_conn'),
                            'delTsxFetch': sett['delTsxFetch'],
                            'delTsxRetry': sett['delTsxRetry'],
                            'deleteMark': sett['deleteMark'],
                            'windowStrategy': sett['windowStrategy'],
                    })

        def sort_key(x):
            if x['key'] == 'beanstalkd':
                return 1
            elif x['key'] == 'redis':
                return 2

            score = 100
            if 'sqlite' in x['key']:
                score += 200
            elif '_mysql' in x['key']:
                score += 300
            elif '_pgsql' in x['key']:
                score += 400
            if 'ph4DBOpt' in x['key']:
                score += 20
                score += x['windowStrategy']
            return score

        # sns.set()
        # sns.set_style("whitegrid")
        # sns.set_style("darkgrid")
        datasets.sort(key=sort_key)

        print('Env, method avg: ')
        avg_group_fnc = lambda x: (x['env'], x['method'])
        for g, k in itertools.groupby(sorted(datasets, key=avg_group_fnc), avg_group_fnc):
            k = list(k)
            avg_jps = sum(x['jps'] for x in k) / float(len(k))
            print('%s %s | %s' % (g[0], g[1], avg_jps))

        data = pd.DataFrame(datasets)

        # g = sns.FacetGrid(data, row='method', sharey=True, size=2)
        # g.map(sns.boxplot, 'jps', 'env', orient='h',)

        sns.axes_style({
            'axes.grid': True,
            'grid.color': '.8',
            'grid.linestyle': '--',

        })

        sns.set_style("whitegrid", {
            'xtick.major.size': 4.0,
            'ytick.major.size': 4.0,
            'grid.linestyle': '--',
        })

        ax = sns.boxplot(y='method', x='jps', hue='env', data=data, linewidth=0.5, orient='h',
                         fliersize=0.65)

        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        # ax.grid(b=True, which='major', color='black', axis='x', alpha=0.2, linewidth=0.5, linestyle='--')
        ax.grid(b=True, which='minor', color='black', axis='x', alpha=0.2, linewidth=0.5, linestyle='--')
        ax.grid(b=True, which='minor', color='black', axis='y', alpha=0.2, linewidth=0.5, linestyle='--')
        # ax.set_yticks([-1.25, -0.75, -0.25, 0.24, 0.75, 1.25], minor=True)

        # ax.yaxis.set_major_locator(ticker.MultipleLocator(1.))
        # ax = sns.factorplot(y='method', x='jps', hue='env', row='key',
        #                     data=data, linewidth=0.5, orient='h', kind='box')

        # ax.set_xticklabels(ax.get_xticklabels(), rotation=-30)
        # plt.gcf().subplots_adjust(bottom=0.15)

        plt.ylabel('')
        plt.legend(loc=4)
        plt.tight_layout()
        plt.savefig('/tmp/avg.svg', transparent=True)
        plt.show()

    @staticmethod
    def avg_dataname(json, classic_only=False, **kwargs):
        sett = json['settings']
        con = sett['conn']

        if con == 'redis':
            return ('redis', 'Redis') if classic_only else (None, None)
        elif con == 'beanstalkd':
            return ('beanstalkd', 'Beanstalkd') if classic_only else (None, None)

        if 'db_conn' not in sett:
            return None, None
        if 'sqlite' in sett['db_conn']:
            return None, None
        if classic_only and len(defvalkey(sett, 'key', '', take_none=True)) > 0:
            return None, None

        if con == 'ph4DBPess':
            params = (sett['deleteMark'], sett['delTsxFetch'], sett['delTsxRetry'])
            # if params not in [(False, False, 5), (False, True, 5), (False, True, 1), (True, False, 1)]:
            if classic_only and params not in [(False, False, 5)]:
                return None, None
            elif not classic_only and params not in [(False, False, 5), (True, False, 1)]:
                return None, None

            suffix = defvalkey(sett, 'key', '%d-%d-%s' % (int(sett['deleteMark']), int(sett['delTsxFetch']), int(sett['delTsxRetry'])), False)
            return '%s_%s_%s_%s_%s' % (con, sett['db_conn'], sett['deleteMark'], sett['delTsxFetch'], sett['delTsxRetry']),\
                   'DB-%s' % (sett['db_conn']) if classic_only else \
                   'DBP-%s-%s' % (sett['db_conn'], suffix)

        if con == 'ph4DBOptim':
            if classic_only:
                return None, None
            params = (sett['windowStrategy'],)
            if params not in [(0,), (1,), ]:
                return None, None
            return '%s_%s_%s' % (con, sett['db_conn'], sett['windowStrategy']), \
                   'DBO-%s-%d' % (sett['db_conn'], int(sett['windowStrategy']))
        return None


if __name__ == '__main__':
    app = Graphs()
    print(app.main())

