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
import hashlib
import shutil
import shlex
import sys
from jsonpath_ng import jsonpath, parse
from typing import Dict, List, Tuple, Optional, Any
from booltest.runner import AsyncRunner


logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)

job_tpl = './generator-metacentrum.sh -c=%s | ./booltest-json-metacentrum.sh %s > "${LOGDIR}/%s.log" 2>&1'
job_tpl_data_file = './booltest-json-metacentrum.sh %s > "${LOGDIR}/%s.log" 2>&1'


def try_rm(pth):
    if not pth:
        return
    try:
        os.unlink(pth)
    except:
        pass


def try_rm_tree(pth):
    if not pth:
        return
    try:
        shutil.rmtree(pth, True)
    except:
        pass


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
        self.working_job = None  # type: Optional[Job]
        self.finished = False
        self.res_code = None
        self.res_out = None
        self.res_err = None
        self.res_time = None
        self.last_hb = 0
        self.runner = None  # type: Optional[AsyncRunner]
        self.scratch_dir = None


class Job:
    def __init__(self, uid=None, obj=None):
        self.uuid = uid
        self.obj = obj  # type: Optional[Dict[str, Any]]
        self.scratch_dir = None
        self.log_file = None


class JobClient:
    def __init__(self):
        self.args = None
        self.time_start = time.time()
        self.workers = []  # type: List[JobWorker]
        self.db_lock = asyncio.Lock()
        self.ws_conn = None
        self.key = None
        self.last_server_ping = time.time()
        self.shutdown_flag = False
        self.last_empty_fetch = 0
        self.scratch_dir = None

    def get_uri(self):
        return "ws://%s:%s" % (self.args.server, self.args.port)

    async def comm_msg(self, msg):
        async with websockets.connect(self.get_uri()) as websocket:
            await websocket.send(json.dumps(msg))
            resp = await websocket.recv()
            js = json.loads(resp)
            self.last_server_ping = time.time()
            self.process_msg(js)
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
            ('ret_code', worker.res_code),
            ('time_elapsed', worker.res_time),
        ])
        return await self.comm_msg(msg)

    def should_terminate(self):
        if not self.args.time:
            return False
        return (time.time() - self.time_start) + 10*60 >= self.args.time

    def ext_key(self, resp, worker: JobWorker):
        if not self.key:
            return resp
        sha = hashlib.sha256()
        tt = int(time.time())
        sha.update(bytes(worker.uuid, "ascii"))
        sha.update(tt.to_bytes(8, byteorder='big'))
        resp['auth_time'] = tt
        resp['auth_token'] = sha.hexdigest()
        return resp

    def process_msg(self, msg):
        if self.args.epoch and 'only_epochs' in msg:
            try:
                ep = int(msg['only_epochs'])
                if ep > self.args.epoch:
                    self.shutdown_flag = True
            except Exception as e:
                logger.warning("Msg processing error: %s" % (e,), exc_info=e)

        if 'terminate' in msg and msg['terminate']:
            logger.info("Server commanded to terminate.")
            self.shutdown_flag = True

    async def worker_hb(self, wx: JobWorker):
        logger.info("Worker %s:%s HB job %s" % (wx.idx, wx.uuid, wx.working_job.uuid))
        await self.comm_hb(wx.working_job, wx)
        wx.last_hb = time.time()
        return True

    def scratch_dir_get(self, fallback='/tmp'):
        scratch = os.getenv('SCRATCHDIR')

        if not scratch:
            logger.warning('SCRATCHDIR dir not set')
            return fallback

        if not os.path.exists(scratch):
            logger.warning('SCRATCHDIR dir does not exist: %s' % (scratch,))
            return fallback
        return scratch

    def get_cli(self, job: Job):
        jo = job.obj
        cfg_file = jo['cfg_file_path']
        gen_file = jo['gen_file_path']
        res_file = jo['res_file']

        if 'gen_data' in jo and 'cfg_data' in jo:
            os.makedirs(job.scratch_dir, exist_ok=True)
            cfg_file = os.path.join(job.scratch_dir, 'cfg-%s.json' % res_file)
            gen_file = os.path.join(job.scratch_dir, 'gen-%s.json' % res_file)
            with open(cfg_file, 'w+') as fh:
                json.dump(jo['cfg_data'], fh)
            with open(gen_file, 'w+') as fh:
                json.dump(jo['gen_data'], fh)

        args = ' --config-file %s' % cfg_file
        job_exec = ''
        if gen_file:
            job_exec = job_tpl % (gen_file, args, jo['res_file'])
        else:
            job_exec = job_tpl_data_file % (args, jo['res_file'])

        logdir = self.args.logdir
        if self.args.log_to_scratch:
            job.log_file = os.path.join(job.scratch_dir, '%s.log' % jo['res_file'])
            logdir = job.scratch_dir

        job_exec = job_exec.replace('${LOGDIR}', logdir)
        return job_exec

    def try_cleanup_job_files(self, job: Job, exit_code: int):
        try:
            jo = job.obj
            files = [jo['cfg_file_path']]

            if self.args.logdir and self.args.log_to_scratch and job.log_file and exit_code != 0:
                dst_log = os.path.join(self.args.logdir, '%s.log' % jo['res_file'])
                try:
                    shutil.copy(dst_log, job.log_file)
                except Exception as e:
                    logger.warning("Could not copy log file %s to %s: %s" % (dst_log, job.log_file, e), exc_info=e)

            elif self.args.logdir and not self.args.log_to_scratch and exit_code == 0:
                files.append(os.path.join(self.args.logdir, '%s.log' % jo['res_file']))

            for fl in files:
                try_rm(fl)

        except Exception as e:
            logger.warning("Exception in job files cleanup: %s" % (e,), exc_info=e)

    async def worker_fetch(self, wx: JobWorker):
        job = await self.comm_get_job(wx)
        if job is None:
            self.last_empty_fetch = time.time()
            return False

        jb = Job(job['uuid'], job)
        jb.scratch_dir = os.path.join(wx.scratch_dir, 'jb-%s' % jb.uuid)
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
        return True

    async def worker_check(self, wx: JobWorker):
        if wx.runner.is_running:
            return False

        wx.res_code = wx.runner.ret_code
        wx.res_out = ''.join(wx.runner.out_acc) if wx.runner.out_acc else ''
        wx.res_err = ''.join(wx.runner.err_acc) if wx.runner.err_acc else ''
        wx.res_time = wx.runner.time_elapsed
        wx.finished = True
        logger.info("Worker %s:%s finished job %s, code: %s, time: %s"
                    % (wx.idx, wx.uuid, wx.working_job.uuid, wx.res_code, wx.res_time))
        if wx.res_code != 0:
            logger.warning("Non-zero return code, err: %s" % wx.res_err)
        await self.worker_finished(wx)
        return True

    async def worker_finished(self, wx: JobWorker):
        await self.comm_finished(wx.working_job, wx)
        if self.args.delete_on_success and wx.res_code == 0:
            self.try_cleanup_job_files(wx.working_job, wx.res_code)
        if wx.working_job.scratch_dir:
            try_rm_tree(wx.working_job.scratch_dir)
            wx.working_job.scratch_dir = None

        wx.working_job = None
        wx.last_hb = None
        wx.finished = False
        wx.res_code = None
        wx.res_out = None
        wx.res_err = None
        return True

    async def process_worker(self, wx: JobWorker, ix):
        tt = time.time()
        if wx.working_job and not wx.finished and (tt - wx.last_hb) >= 180:
            return await self.worker_hb(wx)
        elif wx.working_job is None and time.time() - self.last_empty_fetch > 30:
            return await self.worker_fetch(wx)
        elif not wx.finished:
            return await self.worker_check(wx)
        elif wx.finished:
            return await self.worker_finished(wx)
        else:
            return False

    async def process_workers(self):
        change = False
        for ix, wx in enumerate(self.workers):
            async with self.db_lock:
                try:
                    change |= await self.process_worker(wx, ix)
                except Exception as e:
                    logger.warning("Exc in worker %s: %s" % (ix, e), exc_info=e)
                    await asyncio.sleep(1.5)
                    time.sleep(0.1)
        return change

    async def work(self):
        self.args.cwd = os.path.abspath(self.args.cwd)
        self.args.logdir = os.path.abspath(self.args.logdir)
        self.scratch_dir = self.scratch_dir_get()
        for ix in range(self.args.threads):
            wx = JobWorker()
            wx.idx = ix
            wx.scratch_dir = os.path.join(self.scratch_dir, 'booltw-%s' % wx.uuid)
            os.makedirs(wx.scratch_dir, exist_ok=True)
            self.workers.append(wx)

        if self.args.key_file:
            with open(self.args.key_file, 'r+') as fh:
                kjs = json.load(fh)
            self.key = kjs['key']

        change = False
        while True:
            change = False
            if self.should_terminate():
                logger.info("Terminating")
                return

            if time.time() - self.last_server_ping >= 60*30:
                logger.info("Server down for 30 minutes, shutting down")
                return

            if self.shutdown_flag:
                logger.info("Shutdown flag set")
                return

            try:
                change |= await self.process_workers()

            except Exception as e:
                logger.warning("Exception body: %s" % (e,), exc_info=e)
                await asyncio.sleep(2.0)
                time.sleep(0.1)

            if not change:
                await asyncio.sleep(0.4)
                time.sleep(0.2)
            else:
                time.sleep(0.05)

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
        parser.add_argument('--key-file', dest='key_file', default=None,
                            help='Config file with auth keys')
        parser.add_argument('--epoch', dest='epoch', default=None, type=int,
                            help='Epoch ID for remote kill')
        parser.add_argument('--delete-on-success', dest='delete_on_success', default=0, type=int,
                            help='Delete job files on success')
        parser.add_argument('--log-to-scratch', dest='log_to_scratch', default=1, type=int,
                            help='Log to a scratch dir')

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
