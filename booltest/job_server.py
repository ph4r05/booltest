#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import zmq
import argparse
import coloredlogs
import logging
import time
import json
import websockets
import asyncio
import sys
import threading
import itertools
import random
import hashlib
import os
from jsonpath_ng import jsonpath, parse

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


def jsonpath(path, obj, allow_none=False):
    r = [m.value for m in parse(path).find(obj)]
    return r[0] if not allow_none else (r[0] if r else None)


class JobEntry:
    def __init__(self):
        self.unit = None
        self.uuid = None
        self.time_allocated = None
        self.time_ping = None
        self.worker_id = None
        self.idx = None
        self.finished = False
        self.failed = False
        self.retry_ctr = 0

    def desc(self):
        try:
            return self.unit['res_file']
        except:
            return self.uuid


class WorkerEntry:
    def __init__(self, uid=None):
        self.id = uid
        self.last_ping = 0


class JobServer:
    def __init__(self):
        self.args = None
        self.is_running = True
        self.job_entries = {}  # type: dict[str,JobEntry]
        self.job_queue = []  # type: list[str]

        # Mapping worker_id -> job_id
        self.worker_map = {}  # type: dict[str,str]
        self.workers = {}  # type: dict[str,WorkerEntry]
        self.db_lock = asyncio.Lock()
        self.db_lock_t = threading.Lock()
        self.thread_watchdog = None
        self.key = None

    def job_get(self, worker_id=None):
        if len(self.job_queue) == 0:
            return None
        uid = self.job_queue.pop(0)
        jb = self.job_entries[uid]
        jb.time_allocated = time.time()
        jb.time_ping = time.time()
        jb.worker_id = worker_id
        return jb

    def on_job_fail(self, jb, timeout=False):
        if jb.worker_id and jb.worker_id in self.worker_map and self.worker_map[jb.worker_id] == jb.uuid:
            self.worker_map[jb.worker_id] = None

        jb.retry_ctr += 1
        if jb.retry_ctr >= 6 and not timeout:
            jb.finished = True
            jb.failed = True
            logger.warning("Job %s failed, too many retries" % (jb.uuid[:13]))
        else:
            jb.failed = False
            jb.time_ping = None
            jb.time_allocated = None
            self.job_queue.append(jb.uuid)

    def on_job_alloc(self, jb, worker_id):
        if not jb:
            return
        # If worker had another job, finish it now. It failed probably
        if worker_id in self.worker_map:
            wold = self.worker_map[worker_id]
            if wold is not None:
                oldjb = self.job_entries[wold]
                self.on_job_fail(oldjb)

        self.worker_map[worker_id] = jb.uuid
        return jb

    def on_job_finished(self, uid, worker_id, jmsg):
        self.worker_map[worker_id] = None
        jb = self.job_entries[uid]
        if jmsg and 'ret_code' in jmsg:
            rcode = jmsg['ret_code']
            if rcode == 0 or rcode is None:
                jb.finished = True
            else:
                logger.warning("Job %s finished with error: %s, retry ctr: %s"
                               % (jb.uuid[:13], rcode, jb.retry_ctr))
                self.on_job_fail(jb, False)
        else:
            jb.finished = True

    def on_job_hb(self, uid, worker_id):
        jb = self.job_entries[uid]
        jb.time_ping = time.time()

    def on_worker_ping(self, worker_id):
        if worker_id not in self.workers:
            self.workers[worker_id] = WorkerEntry(worker_id)
        self.workers[worker_id].last_ping = time.time()

    def check_auth(self, msg):
        if not self.key:
            return True

        sha = hashlib.sha256()
        tt = int(msg['auth_time'])
        uid = str(msg['uuid'])
        sha.update(bytes(uid, "ascii"))
        sha.update(tt.to_bytes(8, byteorder='big'))
        our_token = sha.hexdigest()
        if our_token != msg['auth_token']:
            raise ValueError("Auth token invalid")
        return True

    def get_num_online_workers(self, timeout=60*5):
        tt = time.time()
        return sum([1 for x in self.workers.values() if tt - x.last_ping <= timeout])

    def get_num_jobs(self):
        return len(self.job_queue)

    def get_job_ret_code(self, job):
        return job['ret_code'] if 'ret_code' in job else None

    def run_watchdog(self):
        worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(worker_loop)
        worker_loop.run_until_complete(self.arun_watchdog())

    async def arun_watchdog(self):
        last_sweep = 0
        last_checkpoint = time.time()
        while self.is_running:
            try:
                await asyncio.sleep(3)
                tt = time.time()
                if tt - last_sweep < 180:
                    continue

                with self.db_lock_t:
                    for v in self.job_entries.values():
                        if v.finished or v.failed or v.time_allocated is None:
                            continue
                        if tt - v.time_ping >= 60*5:
                            logger.info("Expiring job %s for worker %s" % (v.uuid, v.worker_id))
                            self.on_job_fail(v, timeout=True)
                    pass

                if tt - last_checkpoint < 60*15 or not self.args.checkpoint:
                    continue

                with self.db_lock_t:
                    non_finished = [x.unit for x in self.job_entries.values() if not x.finished]
                    failed = [x.unit for x in self.job_entries.values() if x.failed]

                js = {'jobs': non_finished, 'jobs_failed': failed}
                checkp_tmp = self.args.checkpoint + '.tmp'
                with open(checkp_tmp, 'w+') as fh:
                    json.dump(js, fh, indent=2)
                os.rename(checkp_tmp, self.args.checkpoint)
                last_checkpoint = tt

            except Exception as e:
                logger.warning("Exception in watchdog: ")

    def buid_resp_job(self, jb):
        if jb is None:
            return {'res': None}
        return {'res': jb.unit}

    async def on_ws_msg(self, websocket, path):
        logger.debug("on_msg")
        msg = await websocket.recv()
        logger.debug("Msg recv: %s" % (msg, ))

        resp = await self.on_msg(msg)
        if self.args.epoch:
            resp['only_epochs'] = self.args.epoch
        if self.args.kill_all:
            resp['terminate'] = True

        resp_js = json.dumps(resp)
        await websocket.send(resp_js)

    async def on_msg(self, message):
        try:
            jmsg = json.loads(message)
            if 'action' not in jmsg:
                raise ValueError("Invalid message")

            act = jmsg['action']
            wid = jmsg['uuid']

            if act == 'acquire':
                if self.args.kill_all:
                    return self.buid_resp_job(None)

                with self.db_lock_t:
                    jb = self.job_get(wid)
                    if jb:
                        self.on_job_alloc(jb, wid)
                        self.on_worker_ping(wid)
                        numw = self.get_num_online_workers()
                        numj = self.get_num_jobs()
                        logger.info("Job acquired %s by wid %s, #w: %5d, #j: %7d, %s"
                                    % (jb.uuid[:13], wid[:13], numw, numj, jb.desc()))
                    resp = self.buid_resp_job(jb)
                return resp

            elif act == 'finished':
                jid = jmsg['jid']
                with self.db_lock_t:
                    self.on_job_finished(jid, wid, jmsg)
                    self.on_worker_ping(wid)
                    numw = self.get_num_online_workers()
                    numj = self.get_num_jobs()
                    rcode = self.get_job_ret_code(jmsg)
                logger.info("Job finished %s by wid %s, #w: %5d, #j: %7d, c: %s, %s"
                            % (jid[:13], wid[:13], numw, numj, rcode, self.job_entries[jid].desc()))
                return {'resp': 'ok'}

            elif act == 'heartbeat':
                jid = jmsg['jid']
                with self.db_lock_t:
                    self.on_job_hb(jid, wid)
                    self.on_worker_ping(wid)
                return {'resp': 'ok'}

            else:
                logger.info("Unknown action: [%s]" % (act,))

        except Exception as e:
            logger.warning("Exception in job handling: %s" % (e,), exc_info=e)
        return {'error': 'invalid'}

    async def work(self):
        # context = zmq.Context()
        # socket = context.socket(zmq.REP)
        # socket.bind("tcp://*:%s" % self.args.port)
        start_server = websockets.serve(self.on_ws_msg, "0.0.0.0", self.args.port)
        logger.info("Server started at 0.0.0.0:%s, python: %s" % (self.args.port, sys.version))
        await start_server

        # loop = asyncio.get_event_loop()
        # loop.run_forever()

    async def main(self):
        parser = self.argparse()
        self.args = parser.parse_args()

        if self.args.key_file:
            with open(self.args.key_file, 'r+') as fh:
                kjs = json.load(fh)
            self.key = kjs['key']

        logger.info("Reading job files")
        agg_jobs = []
        for fl in self.args.files:
            logger.info("Processing %s" % (fl,))
            with open(fl) as fh:
                js = json.load(fh)
            jobs = js['jobs']
            for j in jobs:
                je = JobEntry()
                je.unit = j
                je.uuid = j['uuid']
                je.idx = len(self.job_entries)
                agg_jobs.append(je)
                self.job_entries[je.uuid] = je
                if not self.args.sort_jobs and not self.args.rand_jobs:
                    self.job_queue.append(je.uuid)

        # Fast jobs first
        if self.args.sort_jobs:
            cplx_lambda = lambda x: (x['size_mb'], x['degree'], x['comb_deg'], x['block_size'])
            agg_jobs.sort(key=lambda x: cplx_lambda(x.unit))
            for k, g in itertools.groupby(agg_jobs, key=lambda x: cplx_lambda(x.unit)):
                subs = list(g)
                random.shuffle(subs)
                for je in subs:
                    self.job_queue.append(je.uuid)

        elif self.args.rand_jobs:
            random.shuffle(agg_jobs)
            for je in agg_jobs:
                self.job_queue.append(je.uuid)

        logger.info("Jobs in the queue: %s" % (len(self.job_queue),))

        self.thread_watchdog = threading.Thread(target=self.run_watchdog, args=())
        self.thread_watchdog.setDaemon(False)
        self.thread_watchdog.start()

        return await self.work()

    def argparse(self):
        parser = argparse.ArgumentParser(
            description='Job provider')

        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')
        parser.add_argument('--port', dest='port', type=int, default=4688,
                            help='port to bind to')
        parser.add_argument('--checkpoint', dest='checkpoint',
                            help='Job checkpointing file')
        parser.add_argument('--key-file', dest='key_file', default=None,
                            help='Config file with auth keys')
        parser.add_argument('--sort-jobs', dest='sort_jobs', action='store_const', const=True,
                            help='sort jobs by difficulty')
        parser.add_argument('--rand-jobs', dest='rand_jobs', action='store_const', const=True,
                            help='randomize jobs')
        parser.add_argument('--kill-all', dest='kill_all', action='store_const', const=True,
                            help='kill all clients, no job serving')
        parser.add_argument('--epoch', dest='epoch', type=int, default=None,
                            help='client epoch to require')
        parser.add_argument('files', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='job files')

        return parser


async def amain():
    js = JobServer()
    await js.main()


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(amain())
    loop.run_forever()
    loop.close()
    # asyncio.get_event_loop().run_until_complete(js.main)
    # asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
    main()
