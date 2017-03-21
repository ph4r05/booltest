#!/usr/bin/env python
# -*- coding: utf-8 -*-

from past.builtins import basestring
import argparse
import logging
import coloredlogs
import traceback
import requests
import collections
import json
from lxml import html


logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


class RttFetch(object):
    """
    Fetches results from RTT
    """

    def __init__(self):
        self.args = None

    def load_page(self, url):
        """
        Loads given page with attempts
        :param url:
        :return:
        """
        for attempt in range(0, 3):
            try:
                res = requests.get(url)
                if res.status_code == 404:
                    logger.info('%s: 404 code' % url)
                    return None

                res.raise_for_status()
                data = res.content
                if data is None:
                    return None

                data = data.strip()
                if len(data) == 0:
                    return None

                return data

            except Exception as e:
                logger.error('Exception when fetching url %s: %s' % (url, e))
                logger.debug(traceback.format_exc())

        raise EnvironmentError('Could no fetch results: %s' % url)

    def work(self):
        """
        Main download job
        :return:
        """

        to_idx = self.args.to_idx
        if to_idx is None:
            to_idx = 50000

        records = []
        record_map = {}
        u01_tests = set()
        for idx in range(self.args.from_idx, to_idx):
            try:
                url = self.args.url + 'ViewResults/Experiment/%d/' % idx
                data = self.load_page(url)
                if data is None:
                    logger.error('Empty data for experiment %d' % idx)
                    continue

                tree = html.fromstring(data)
                tables = tree.xpath('//table')

                table_desc, table_res = None, None
                if len(tables) == 0:
                    raise ValueError('Invalid page')
                elif len(tables) == 1:
                    table_desc = tables[0]
                    table_res = []
                else:
                    table_desc, table_res = tables

                # Desc
                id = int(table_desc[0][1].text_content())
                name = table_desc[1][1].text_content()
                email = table_desc[2][1].text_content()
                status = table_desc[4][1].text_content()
                file_name = table_desc[6][1].text_content()

                if self.args.email is not None and email != self.args.email:
                    logger.info('Skipping experiment, different email: %s' % email)
                    continue

                exp_nist = None
                exp_die = None
                exp_u01 = []

                # Results
                for row_idx in range(1, len(table_res)):
                    row = table_res[row_idx]
                    exp_name = row[0].text_content()
                    exp_pass = int(row[1].text_content())
                    exp_total = int(row[2].text_content())
                    exp_more = row[3][0].attrib['href']

                    exp_rec = (exp_name, exp_pass, exp_total, exp_more)
                    if exp_name.lower().startswith('nist'):
                        exp_nist = exp_rec
                    elif exp_name.lower().startswith('die'):
                        exp_die = exp_rec
                    else:
                        exp_u01.append(exp_rec)
                        u01_tests.add(exp_name)

                exp_u01.sort()
                experiments = [exp_nist, exp_die] + exp_u01

                record = collections.OrderedDict()
                record['id'] = id
                record['name'] = name
                record['status'] = status
                record['file_name'] = file_name

                # Custom naming: FUNCTION_rround_0000MB
                if self.args.name_fmt:
                    parts = name.rsplit('_', 2)

                    function = parts[0]
                    cur_round = int(parts[1][1:])  # rINT
                    cur_data = int(parts[2][:-2])  # xyzMB

                    record['function'] = function
                    record['round'] = cur_round
                    record['data_mb'] = cur_data
                    record['tests'] = experiments
                    record_map[(function, cur_round, cur_data)] = record
                    logger.info('Test %02d: %s, round: %s, data: %d, %s. ' % (id, function, cur_round, cur_data, status))

                else:
                    record['tests'] = experiments
                    record_map[name] = record

                records.append(record)

            except Exception as e:
                logger.error('Download error: %s' % e)
                logger.debug(traceback.format_exc())
                break

        if self.args.json:
            print(json.dumps(records, indent=2))

        else:
            delim = self.args.delim
            hdr = []

            if self.args.name_fmt:
                hdr = ['function', 'round', 'data']
            else:
                hdr = ['name']

            u01_test_names = sorted(list(u01_tests))
            hdr += ['NIST_pass', 'NIST_total', 'NIST_succ', 'Dieharder_pass', 'Dieharder_total', 'Dieharder_succ']
            for u01 in u01_test_names:
                u01 = u01.replace(' ', '_')
                hdr += ['%s_pass' % u01, '%s_total' % u01, '%s_succ' % u01]
            print(delim.join(hdr))

            keys = sorted(record_map.keys())
            for key in keys:
                record = record_map[key]
                out = []

                if self.args.name_fmt:
                    out = [key[0], key[1], key[2]]
                else:
                    out = [key]

                tests = record['tests']
                u01_tests = sorted(tests[2:], key=lambda x: x[0])

                tests_sorted = tests[0:2] + u01_tests
                for cur_test in tests_sorted:
                    if cur_test is None or cur_test[1] is None:
                        out += ['', '', '']
                        continue

                    out += [cur_test[1], cur_test[2], 100.0*float(cur_test[1])/float(cur_test[2]) if cur_test[2] > 0 else '']

                print(delim.join([str(x) for x in out]))

    def main(self):
        logger.debug('App started')

        parser = argparse.ArgumentParser(description='Randomness testing toolking results fetcher')
        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')

        parser.add_argument('--verbose', dest='verbose', action='store_const', const=True,
                            help='enables verbose mode')

        parser.add_argument('--delim', dest='delim', default=',',
                            help='CSV delimiter')

        parser.add_argument('--json', dest='json', action='store_const', const=True,
                            help='json output mode')

        parser.add_argument('--name-fmt', dest='name_fmt', action='store_const', const=True,
                            help='if defined experiment name format is expected to be: Keccak_r4_1000MB')

        parser.add_argument('--url', dest='url', default='http://147.251.253.206/',
                            help='URL to main RTT web page with results')

        parser.add_argument('--from', dest='from_idx', default=0, type=int,
                            help='Result ID to start with')

        parser.add_argument('--to', dest='to_idx', default=None, type=int,
                            help='Result ID to end with ')

        parser.add_argument('--email', dest='email', default=None,
                            help='Only results with given email')

        self.args = parser.parse_args()
        if self.args.debug:
            coloredlogs.install(level=logging.DEBUG)

        self.work()


# Launcher
app = None
if __name__ == "__main__":
    app = RttFetch()
    app.main()

