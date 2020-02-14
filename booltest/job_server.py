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


class JobServer:
    def __init__(self):
        self.args = None
        self.is_running = True
        self.job_entries = {}  # type: dict[str,JobEntry]
        self.finished_jobs = set()  # type: set[str]
        self.job_queue = []  # type: list[str]

        # Mapping worker_id -> job_id
        self.worker_map = {}  # type: dict[str,str]
        self.db_lock = asyncio.Lock()
        self.thread_watchdog = None

    def job_get(self, worker_id=None):
        if len(self.job_queue) == 0:
            return 0
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
        if jb.retry_ctr >= 2 and not timeout:
            jb.finished = True
            jb.failed = True
        else:
            jb.failed = False
            jb.time_ping = None
            jb.time_allocated = None
            self.job_queue.append(jb.uuid)

    def on_job_alloc(self, jb, worker_id):
        # If worker had another job, finish it now. It failed probably
        if worker_id in self.worker_map:
            wold = self.worker_map[worker_id]
            if wold is not None:
                oldjb = self.job_entries[wold]
                self.on_job_fail(oldjb)

        self.worker_map[worker_id] = jb.uuid
        return jb

    def on_job_finished(self, uid, worker_id):
        jb = self.job_entries[uid]
        jb.finished = True
        self.worker_map[worker_id] = None

    def on_job_hb(self, uid, worker_id):
        jb = self.job_entries[uid]
        jb.time_ping = time.time()

    def run_watchdog(self):
        worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(worker_loop)
        worker_loop.run_until_complete(self.arun_watchdog())

    async def arun_watchdog(self):
        last_sweep = 0
        while self.is_running:
            await asyncio.sleep(3)
            tt = time.time()
            if tt - last_sweep < 180:
                continue

            async with self.db_lock:
                for v in self.job_entries.values():
                    if v.finished or v.failed or v.time_allocated is None:
                        continue
                    if tt - v.time_ping >= 60*5:
                        logger.info("Expiring job %s for worker %s" % (v.uuid, v.worker_id))
                        self.on_job_fail(v, timeout=True)
                pass

    def buid_resp_job(self, jb):
        if jb is None:
            return {'res': None}
        return {'res': jb.unit}

    async def on_ws_msg(self, websocket, path):
        logger.debug("on_msg")
        msg = await websocket.recv()
        logger.debug("Msg recv: %s" % (msg, ))

        resp = await self.on_msg(msg)

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
                async with self.db_lock:
                    jb = self.job_get(wid)
                    self.on_job_alloc(jb, wid)
                resp = self.buid_resp_job(jb)
                logger.info("Job acquired %s by wid %s, %s" % (jb.uuid, wid, jb.desc()))
                return resp

            elif act == 'finished':
                jid = jmsg['jid']
                async with self.db_lock:
                    self.on_job_finished(jid, wid)
                logger.info("Job finished %s by wid %s, %s" % (jid, wid, self.job_entries[jid].desc()))
                return {'resp': 'ok'}

            elif act == 'heartbeat':
                jid = jmsg['jid']
                async with self.db_lock:
                    self.on_job_hb(jid, wid)
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

        agg_jobs = []
        for fl in self.args.files:
            with open(fl) as fh:
                js = json.load(fh)
            jobs = js['jobs']
            for j in jobs:
                je = JobEntry()
                je.unit = j
                je.uuid = j['uuid']
                je.idx = len(self.job_entries)
                agg_jobs.append(je)

        # Fast jobs first
        cplx_lambda = lambda x: (x['size_mb'], x['degree'], x['comb_deg'], x['block_size'])
        agg_jobs.sort(key=lambda x: cplx_lambda(x.unit))
        for k, g in itertools.groupby(agg_jobs, key=lambda x: cplx_lambda(x.unit)):
            subs = list(g)
            random.shuffle(subs)
            for je in subs:
                self.job_entries[je.uuid] = je
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
