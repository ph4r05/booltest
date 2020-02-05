#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import coloredlogs
import logging
import json
import itertools
import shlex
import time
import queue
import sys
import os
import collections
import tempfile
from jsonpath_ng import jsonpath, parse

from .runner import AsyncRunner
from .common import merge_pvals, booltest_pval
from . import common

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

"""
Config can look like this:

{
    "default-cli": "--no-summary --json-out --log-prints --top 128 --no-comb-and --only-top-comb --only-top-deg --no-term-map --topterm-heap --topterm-heap-k 256 --best-x-combs 512",
    "strategies": [
        {
            "name": "v1",
            "cli": "",
            "variations": [
                {
                    "bl": [128, 256, 384, 512],
                    "deg": [1],
                    "cdeg": [1],
                    "exclusions": []
                }
            ]
        },
        {
            "name": "halving",
            "cli": "--halving",
            "variations": [
                {
                    "bl": [128, 256, 384, 512],
                    "deg": [1, 2, 3],
                    "cdeg": [1, 2, 3],
                    "exclusions": []
                }
            ]
        }
    ]
}
"""


def jsonpath(path, obj, allow_none=False):
    r = [m.value for m in parse(path).find(obj)]
    return r[0] if not allow_none else (r[0] if r else None)


def listize(obj):
    return obj if (obj is None or isinstance(obj, list)) else [obj]


def get_runner(cli, cwd=None, rtt_env=None):
    async_runner = AsyncRunner(cli, cwd=cwd, shell=False, env=rtt_env)
    async_runner.log_out_after = False
    async_runner.preexec_setgrp = True
    return async_runner


class BoolParamGen:
    def __init__(self, cli, vals):
        self.cli = cli
        self.vals = vals if isinstance(vals, list) else [vals]


class BoolJob:
    def __init__(self, cli, name, vinfo='', idx=None):
        self.cli = cli
        self.name = name
        self.vinfo = vinfo
        self.idx = idx

    def is_halving(self):
        return '--halving' in self.cli


class BoolRes:
    def __init__(self, job, ret_code, js_res, is_halving, rejects=False, pval=None, alpha=None, stderr=None):
        self.job = job  # type: BoolJob
        self.ret_code = ret_code
        self.js_res = js_res
        self.is_halving = is_halving
        self.rejects = rejects
        self.alpha = alpha
        self.pval = pval
        self.stderr = stderr


class BoolRunner:
    def __init__(self):
        self.args = None
        self.bool_config = None
        self.parallel_tasks = None
        self.bool_wrapper = None
        self.job_queue = queue.Queue(maxsize=0)
        self.runners = []  # type: List[Optional[AsyncRunner]]
        self.comp_jobs = []  # type: List[Optional[BoolJob]]
        self.results = []

    def init_config(self):
        self.parallel_tasks = self.args.threads or 1
        self.bool_wrapper = self.args.booltest_bin
        try:
            if self.args.config:
                with open(self.args.config) as fh:
                    self.bool_config = json.load(fh)

                if not self.bool_wrapper:
                    self.bool_wrapper = jsonpath("$.wrapper", self.bool_config, True)
                if not self.args.threads:
                    self.parallel_tasks = jsonpath("$.threads", self.bool_config, True) or self.args.threads or 1

        except Exception as e:
            logger.error("Could not load the config %s" % (e,), exc_info=e)

        if not self.bool_wrapper:
            self.bool_wrapper = "\"%s\" -m booltest.booltest_main" % sys.executable

    def norm_methods(self, methods):
        res = set()
        for m in methods:
            if m == 'v1':
                res.add('1')
            elif m == '1':
                res.add(m)
            elif m == 'halving':
                res.add('2')
            elif m == 'v2':
                res.add('2')
            elif m == '2':
                res.add(m)
            else:
                raise ValueError("Unknown method %s" % m)
        return sorted(list(res))

    def norm_params(self, params, default):
        if params is None or len(params) == 0:
            return default
        return [int(x) for x in params]

    def generate_jobs(self):
        dcli = self.args.cli
        if dcli is None:
            dcli = jsonpath('$.default-cli', self.bool_config, True)
        if dcli is None:
            dcli = '--no-summary --json-out --log-prints --top 128 --no-comb-and --only-top-comb --only-top-deg ' \
                   '--no-term-map --topterm-heap --topterm-heap-k 256 --best-x-combs 512'
        if '--no-summary' not in dcli:
            dcli += ' --no-summary'
        if '--json-out' not in dcli:
            dcli += ' --json-out'
        if '--log-prints' not in dcli:
            dcli += ' --log-prints'

        strategies = jsonpath('$.strategies', self.bool_config, True)
        if strategies is None:
            strategies = []
            methods = self.norm_methods(self.args.methods or ["1", "2"])
            for mt in methods:
                strat = collections.OrderedDict()
                strat['name'] = "v%s" % mt
                strat['cli'] = "--halving" if mt == '2' else ''
                strat['variations'] = [collections.OrderedDict([
                    ('bl', self.norm_params(self.args.block, [128, 256, 384, 512])),
                    ('deg', self.norm_params(self.args.deg, [1, 2])),
                    ('cdeg', self.norm_params(self.args.comb_deg, [1, 2])),
                    ('exclusions', []),
                ])]
                strategies.append(strat)

        for st in strategies:
            name = st['name']
            st_cli = jsonpath('$.cli', st, True) or ''
            st_vars = jsonpath('$.variations', st, True) or []
            ccli = ('%s %s' % (dcli, st_cli)).strip()

            if not st_vars:
                yield BoolJob(ccli, name)
                continue

            for cvar in st_vars:
                blocks = listize(jsonpath('$.bl', cvar, True)) or [None]
                degs = listize(jsonpath('$.deg', cvar, True)) or [None]
                cdegs = listize(jsonpath('$.cdeg', cvar, True)) or [None]
                pcli = ['--block', '--degree', '--combine-deg']
                vinfo = ['', '', '']
                iterator = itertools.product(blocks, degs, cdegs)

                for el in iterator:
                    c = ' '.join([(('%s %s') % (pcli[ix], dt)) for (ix, dt) in enumerate(el) if dt is not None])
                    vi = '-'.join([(('%s%s') % (vinfo[ix], dt)).strip() for (ix, dt) in enumerate(el) if dt is not None])
                    ccli0 = ('%s %s' % (ccli, c)).strip()

                    yield BoolJob(ccli0, name, vi)

    def run_job(self, cli):
        async_runner = get_runner(shlex.split(cli))

        logger.info("Starting async command %s" % cli)
        async_runner.start()

        while async_runner.is_running:
            time.sleep(1)
        logger.info("Async command finished")

    def on_finished(self, job, runner, idx):
        if runner.ret_code != 0:
            logger.warning("Return code of job %s is %s" % (idx, runner.ret_code))
            stderr = ("\n".join(runner.err_acc)).strip()
            br = BoolRes(job, runner.ret_code, None, job.is_halving, stderr=stderr)
            self.results.append(br)
            return

        results = runner.out_acc
        buff = (''.join(results)).strip()
        try:
            js = json.loads(buff)

            is_halving = js['halving']
            br = BoolRes(job, 0, js, is_halving)

            if not is_halving:
                br.rejects = [m.value for m in parse('$.inputs[0].res[0].rejects').find(js)][0]
                br.alpha = [m.value for m in parse('$.inputs[0].res[0].ref_alpha').find(js)][0]
                logger.info('rejects: %s, at alpha %.5e' % (br.rejects, br.alpha))

            else:
                br.pval = [m.value for m in parse('$.inputs[0].res[1].halvings[0].pval').find(js)][0]
                logger.info('halving pval: %5e' % br.pval)

            self.results.append(br)

        except Exception as e:
            logger.error("Exception processing results: %s" % (e,), exc_info=e)
            logger.warning("[[[%s]]]" % buff)

    def on_results_ready(self):
        try:
            logger.info("="*80)
            logger.info("Results")
            ok_results = [r for r in self.results if r.ret_code == 0]
            nok_results = [r for r in self.results if r.ret_code != 0]
            bat_errors = ['Job %d (%s-%s), ret_code %d' % (r.job.idx, r.job.name, r.job.vinfo, r.ret_code)
                          for r in self.results if r.ret_code != 0]

            if nok_results:
                logger.warning("Some jobs failed with error: \n%s" % ("\n".join(bat_errors)))
            for r in nok_results:
                logger.info("Job %s, (%s-%s)" % (r.job.idx, r.job.name, r.job.vinfo))
                logger.info("Stderr: %s" % r.stderr)

            v1_jobs = [r for r in ok_results if not r.is_halving]
            v2_jobs = [r for r in ok_results if r.is_halving]

            v1_sum = collections.OrderedDict()
            v2_sum = collections.OrderedDict()

            if v1_jobs:
                rejects = [r for r in v1_jobs if r.rejects]
                v1_sum['alpha'] = max([x.alpha for x in v1_jobs])
                v1_sum['pvalue'] = booltest_pval(nfails=len(rejects), ntests=len(v1_jobs), alpha=v1_sum['alpha'])
                v1_sum['npassed'] = sum([1 for r in v1_jobs if not r.rejects])

            if v2_jobs:
                pvals = [r.pval for r in v2_jobs]
                v2_sum['npassed'] = sum([1 for r in v2_jobs if r.pval >= self.args.alpha])
                v2_sum['pvalue'] = merge_pvals(pvals)[0] if len(pvals) > 1 else -1

            if v1_jobs:
                logger.info("V1 results:")
                self.print_test_res(v1_jobs)

            if v2_jobs:
                logger.info("V2 results:")
                self.print_test_res(v2_jobs)

            logger.info("=" * 80)
            logger.info("Summary: ")
            if v1_jobs:
                logger.info("v1 tests: %s, #passed: %s, pvalue: %s"
                            % (len(v1_jobs), v1_sum['npassed'], v1_sum['pvalue']))

            if v2_jobs:
                logger.info("v2 tests: %s, #passed: %s, pvalue: %s"
                            % (len(v2_jobs), v2_sum['npassed'], v2_sum['pvalue']))

            if not self.args.json_out and not self.args.json_out_file:
                return

            jsout = collections.OrderedDict()
            jsout["nfailed_jobs"] = len(nok_results)
            jsout["failed_jobs_stderr"] = [r.stderr for r in nok_results]
            jsout["results"] = common.noindent_poly([r.js_res for r in ok_results])

            kwargs = {'indent': 2} if self.args.json_nice else {}
            if self.args.json_out:
                print(common.json_dumps(jsout, **kwargs))

            if self.args.json_out_file:
                with open(self.args.json_out_file, 'w+') as fh:
                    common.json_dump(jsout, fh, **kwargs)

            jsout = common.jsunwrap(jsout)
            return jsout

        except Exception as e:
            logger.warning("Exception in results processing: %s" % (e,), exc_info=e)

    def print_test_res(self, res):
        for rs in res:  # type: BoolRes
            passed = (rs.pval >= self.args.alpha if rs.is_halving else not rs.rejects) if rs.ret_code == 0 else None
            desc_str = ""
            if rs.is_halving:
                desc_str = "pvalue: %5e" % (rs.pval,)
            else:
                desc_str = "alpha: %5e" % (rs.alpha,)

            res = rs.js_res["inputs"][0]["res"]
            dist_poly = jsonpath('$[0].dists[0].poly', res, True)
            time_elapsed = jsonpath('$.time_elapsed', rs.js_res, True)
            best_dist_zscore = jsonpath('$[0].dists[0].zscore', res, True) or -1
            ref_zscore_min = jsonpath('$[0].ref_minmax[0]', res, True) or -1
            ref_zscore_max = jsonpath('$[0].ref_minmax[1]', res, True) or -1

            aux_str = ""
            if rs.is_halving:
                best_dist_zscore_halving = jsonpath('$[1].dists[0].zscore', res, True)
                aux_str = "Learn: (z-score: %.5f, acc. zscores: [%.5f, %.5f]), Eval: (z-score: %.5f)" \
                          % (best_dist_zscore, ref_zscore_min, ref_zscore_max, best_dist_zscore_halving)
            else:
                aux_str = "z-score: %.5f, acc. zscores: [%.5f, %.5f]" \
                          % (best_dist_zscore, ref_zscore_min, ref_zscore_max)

            logger.info(" - %s %s: passed: %s, %s, dist: %s\n   elapsed time: %6.2f s, %s"
                        % (rs.job.name, rs.job.vinfo, passed, desc_str, dist_poly,
                           time_elapsed, aux_str))

    def work(self):
        if len(self.args.files) != 1:
            raise ValueError("Provide exactly one file to test")

        ifile = self.args.files[0]
        if ifile != '-' and not os.path.exists(ifile):
            raise ValueError("Provided input file not found")

        tmp_file = None
        if ifile == '-':
            tmp_file = tempfile.NamedTemporaryFile(prefix="booltest-bat-inp", delete=True)
            while True:
                data = sys.stdin.read(4096) if sys.version_info < (3,) else sys.stdin.buffer.read(4096)
                if data is None or len(data) == 0:
                    break
                tmp_file.write(data)
            ifile = tmp_file.name

        jobs = [x for x in self.generate_jobs()]
        for i, j in enumerate(jobs):
            j.idx = i

        self.runners = [None] * self.parallel_tasks
        self.comp_jobs = [None] * self.parallel_tasks

        for j in jobs:
            self.job_queue.put_nowait(j)

        while not self.job_queue.empty() or sum([1 for x in self.runners if x is not None]) > 0:
            time.sleep(0.1)

            # Realloc work
            for i in range(len(self.runners)):
                if self.runners[i] is not None and self.runners[i].is_running:
                    continue

                was_empty = self.runners[i] is None
                if not was_empty:
                    self.job_queue.task_done()
                    logger.info("Task %d done, job queue size: %d, running: %s"
                                % (i, self.job_queue.qsize(), sum([1 for x in self.runners if x])))
                    self.on_finished(self.comp_jobs[i], self.runners[i], i)

                # Start a new task, if any
                try:
                    job = self.job_queue.get_nowait()  # type: BoolJob
                except queue.Empty:
                    self.runners[i] = None
                    continue

                cli = '%s %s "%s"' % (self.bool_wrapper, job.cli, ifile)
                self.comp_jobs[i] = job
                self.runners[i] = get_runner(shlex.split(cli))
                logger.info("Starting async command %s %s, %s" % (job.name, job.vinfo, cli))
                self.runners[i].start()

        return self.on_results_ready()

    def main(self):
        parser = self.argparser()
        self.args = parser.parse_args()
        self.init_config()
        return self.work()

    def argparser(self):
        parser = argparse.ArgumentParser(description='BoolTest Battery Runner')

        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')
        parser.add_argument('-c', '--config', default=None,
                            help='Test config')
        parser.add_argument('--alpha', dest='alpha', type=float, default=1e-4,
                            help='Alpha value for pass/fail')
        parser.add_argument('-t', dest='threads', type=int, default=1,
                            help='Maximum parallel threads')

        parser.add_argument('--block', dest='block', nargs=argparse.ZERO_OR_MORE,
                            default=None, type=int,
                            help='List of block sizes to test')

        parser.add_argument('--deg', dest='deg', nargs=argparse.ZERO_OR_MORE,
                            default=None, type=int,
                            help='List of degree to test')

        parser.add_argument('--comb-deg', dest='comb_deg', nargs=argparse.ZERO_OR_MORE,
                            default=None, type=int,
                            help='List of degree of combinations to test')

        parser.add_argument('--methods', dest='methods', nargs=argparse.ZERO_OR_MORE,
                            default=None,
                            help='List of methods to test, supported: 1, 2, halving')

        parser.add_argument('files', nargs=argparse.ONE_OR_MORE, default=[],
                            help='files to process')

        parser.add_argument('--stdin', dest='stdin', action='store_const', const=True, default=False,
                            help='Read from the stdin')

        parser.add_argument('--booltest-bin', dest='booltest_bin',
                            help='Specify BoolTest binary launcher. If not specified, autodetected.')

        parser.add_argument('--cli', dest='cli',
                            help='Specify common BoolTest CLI options')

        parser.add_argument('--json-out', dest='json_out', action='store_const', const=True, default=False,
                            help='Produce json result')

        parser.add_argument('--json-out-file', dest='json_out_file', default=None,
                            help='Produce json result to a file')

        parser.add_argument('--json-nice', dest='json_nice', action='store_const', const=True, default=False,
                            help='Nicely formatted json output')
        return parser


def main():
    br = BoolRunner()
    return br.main()


if __name__ == '__main__':
    main()
