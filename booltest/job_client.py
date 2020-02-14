#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import coloredlogs
import logging
import time
import json
import uuid
import collections
import websockets
import asyncio
import os
import shlex
import sys
from jsonpath_ng import jsonpath, parse
from booltest.runner import AsyncRunner


logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)

job_tpl = './generator-metacentrum.sh -c=%s | ./booltest-json-metacentrum.sh %s > "${LOGDIR}/%s.out" 2> "${LOGDIR}/%s.err"'
job_tpl_data_file = './booltest-json-metacentrum.sh %s > "${LOGDIR}/%s.out" 2> "${LOGDIR}/%s.err"'


def jsonpath(path, obj, allow_none=False):
    r = [m.value for m in parse(path).find(obj)]
    return r[0] if not allow_none else (r[0] if r else None)


def get_runner(cli, cwd=None, env=None, shell=True):
    async_runner = AsyncRunner(cli, cwd=cwd, shell=shell, env=env)
    async_runner.log_out_after = False
    async_runner.preexec_setgrp = True
    return async_runner


class JobWorker:
    def __init__(self):
        self.idx = 0
        self.uuid = str(uuid.uuid4())
        self.working_job = None  # type: Job
        self.finished = False
        self.res_code = None
        self.res_out = None
        self.res_err = None
        self.last_hb = 0
        self.runner = None  # type: AsyncRunner


class Job:
    def __init__(self, uid=None, obj=None):
        self.uuid = uid
        self.obj = obj


class JobClient:
    def __init__(self):
        self.args = None
        self.time_start = time.time()
        self.workers = []  # type: list[JobWorker]
        self.db_lock = asyncio.Lock()
        self.ws_conn = None

    def get_uri(self):
        return "ws://%s:%s" % (self.args.server, self.args.port)

    async def comm_msg(self, msg):
        async with websockets.connect(self.get_uri()) as websocket:
            await websocket.send(json.dumps(msg))
            resp = await websocket.recv()
            js = json.loads(resp)
            return js

    async def comm_get_job(self, worker: JobWorker):
        msg = collections.OrderedDict([
            ('action', 'acquire'),
            ('uuid', worker.uuid)
        ])
        return (await self.comm_msg(msg))['res']

    async def comm_hb(self, job: Job, worker: JobWorker):
        msg = collections.OrderedDict([
            ('action', 'heartbeat'),
            ('uuid', worker.uuid),
            ('jid', job.uuid),
        ])
        return await self.comm_msg(msg)

    async def comm_finished(self, job: Job, worker: JobWorker):
        msg = collections.OrderedDict([
            ('action', 'finished'),
            ('uuid', worker.uuid),
            ('jid', job.uuid),
        ])
        return await self.comm_msg(msg)

    def should_terminate(self):
        if not self.args.time:
            return False
        return (time.time() - self.time_start) + 10*60 >= self.args.time

    async def worker_hb(self, wx: JobWorker):
        logger.info("Worker %s:%s HB job %s" % (wx.idx, wx.uuid, wx.working_job.uuid))
        await self.comm_hb(wx.working_job, wx)
        wx.last_hb = time.time()

    def get_cli(self, job: Job):
        jo = job.obj
        args = ' --config-file %s' % jo['cfg_file_path']
        job_exec = ''
        if jo['gen_file_path']:
            job_exec = job_tpl % (jo['gen_file_path'], args, jo['res_file'], jo['res_file'])
        else:
            job_exec = job_tpl_data_file % (args, jo['res_file'], jo['res_file'])
        job_exec = job_exec.replace('${LOGDIR}', self.args.logdir)
        return job_exec

    async def worker_fetch(self, wx: JobWorker):
        job = await self.comm_get_job(wx)
        if job is None:
            return

        jb = Job(job['uuid'], job)
        wx.working_job = jb
        wx.last_hb = time.time()
        wx.finished = False
        wx.res_code = None
        wx.res_out = None
        wx.res_err = None

        cli = self.get_cli(jb)
        # cli = '/bin/bash -c "sleep 1.1"'
        wx.runner = get_runner(cli, shell=True, cwd=self.args.cwd)
        wx.runner.start(wait_running=False)
        # wx.runner.is_running = False

        logger.info("Worker %s:%s started job %s %s" % (wx.idx, wx.uuid, jb.uuid, cli))

    async def worker_check(self, wx: JobWorker):
        if wx.runner.is_running:
            return

        wx.res_code = wx.runner.ret_code
        wx.res_out = ''.join(wx.runner.out_acc) if wx.runner.out_acc else ''
        wx.res_err = ''.join(wx.runner.err_acc) if wx.runner.err_acc else ''
        wx.finished = True
        logger.info("Worker %s:%s finished job %s, code: %s" % (wx.idx, wx.uuid, wx.working_job.uuid, wx.res_code))
        await self.worker_finished(wx)

    async def worker_finished(self, wx: JobWorker):
        await self.comm_finished(wx.working_job, wx)
        wx.working_job = None
        wx.last_hb = None
        wx.finished = False
        wx.res_code = None
        wx.res_out = None
        wx.res_err = None

    async def process_worker(self, wx: JobWorker, ix):
        tt = time.time()
        if wx.working_job and not wx.finished and (tt - wx.last_hb) >= 180:
            await self.worker_hb(wx)
        elif wx.working_job is None:
            await self.worker_fetch(wx)
        elif not wx.finished:
            await self.worker_check(wx)
        elif wx.finished:
            await self.worker_finished(wx)
        else:
            return False
        return True

    async def process_workers(self):
        change = False
        for ix, wx in enumerate(self.workers):
            async with self.db_lock:
                try:
                    change |= await self.process_worker(wx, ix)
                except Exception as e:
                    logger.warning("Exc in worker %s: %s" % (ix, e), exc_info=e)
                    await asyncio.sleep(1.5)
        return change

    async def work(self):
        self.args.cwd = os.path.abspath(self.args.cwd)
        self.args.logdir = os.path.abspath(self.args.logdir)
        for ix in range(self.args.threads):
            wx = JobWorker()
            wx.idx = ix
            self.workers.append(wx)

        change = False
        while True:
            change = False
            if self.should_terminate():
                logger.info("Terminating")
                return

            try:
                change |= await self.process_workers()

            except Exception as e:
                logger.warning("Exception body: %s" % (e,), exc_info=e)
                await asyncio.sleep(2.0)

            if not change:
                await asyncio.sleep(0.4)

    async def main(self):
        parser = self.argparse()
        self.args = parser.parse_args()
        return await self.work()

    def argparse(self):
        parser = argparse.ArgumentParser(
            description='Job runner')

        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')
        parser.add_argument('--server', dest='server', default='localhost',
                            help='Server address to connect to')
        parser.add_argument('--port', dest='port', type=int, default=4688,
                            help='port to bind to')
        parser.add_argument('--time', dest='time', type=int, default=None,
                            help='time allocation, termination handling')
        parser.add_argument('--threads', dest='threads', type=int, default=1,
                            help='Concurrent jobs to acquire')
        parser.add_argument('--logdir', dest='logdir', default='.',
                            help='Log dir')
        parser.add_argument('--cwd', dest='cwd', default='.',
                            help='Working dir')

        return parser


async def amain():
    js = JobClient()
    await js.main()


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(amain())
    loop.close()


if __name__ == '__main__':
    main()


# import json
# async def hello():
#     uri = "ws://localhost:4688"
#     async with websockets.connect(uri) as websocket:
#         await websocket.send(json.dumps({'test':'test'}))
#         greeting = await websocket.recv()
#         print(f"< {greeting}")
